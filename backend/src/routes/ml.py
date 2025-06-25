from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.main import db
from src.models.ml_prediction import MLPrediction
from src.models.newsletter import Newsletter
from src.models.user import User
import requests
import os
from datetime import datetime

ml_bp = Blueprint("ml", __name__)

ML_ENGINE_URL = os.getenv("ML_ENGINE_URL", "http://localhost:8000")

@ml_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict_trade_probability():
    """Get probability prediction for a trade"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data or not data.get("ticker"):
            return jsonify({"error": "Ticker is required"}), 400

        ticker = data["ticker"].upper()

        # Prepare data for ML engine
        ml_request = {
            "ticker": ticker,
            "newsletter_data": data.get("newsletter_data", {}),
            "market_data": data.get("market_data", {}),
            "portfolio_data": data.get("portfolio_data", {}),
        }

        # Call ML engine
        try:
            response = requests.post(
                f"{ML_ENGINE_URL}/predict", json=ml_request, timeout=30
            )

            if response.status_code == 200:
                prediction_data = response.json()

                # Store prediction in database
                prediction = MLPrediction(
                    user_id=current_user_id,
                    ticker=ticker,
                    newsletter_id=data.get("newsletter_id"),
                    probability_score=prediction_data["probability"],
                    confidence_lower=prediction_data["confidence_interval"][0],
                    confidence_upper=prediction_data["confidence_interval"][1],
                    contributing_factors=prediction_data["feature_importance"],
                    model_version=prediction_data.get("model_version", "1.0.0"),
                )

                db.session.add(prediction)
                db.session.commit()

                return jsonify(
                    {
                        "prediction": prediction.to_dict(),
                        "recommendation": prediction_data["recommendation"],
                    }
                ), 200
            else:
                return (
                    jsonify({"error": "ML engine error", "details": response.text}),
                    500,
                )

        except requests.exceptions.RequestException as e:
            # Fallback to basic prediction if ML engine is unavailable
            fallback_prediction = create_fallback_prediction(
                current_user_id, ticker, data
            )
            return (
                jsonify(
                    {
                        "prediction": fallback_prediction.to_dict(),
                        "recommendation": "MONITOR",
                        "note": "Using fallback prediction - ML engine unavailable",
                    }
                ),
                200,
            )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@ml_bp.route("/predictions", methods=["GET"])
@jwt_required()
def get_predictions():
    """Get user's ML predictions"""
    try:
        current_user_id = get_jwt_identity()
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        ticker = request.args.get("ticker")

        query = MLPrediction.query.filter_by(user_id=current_user_id)

        if ticker:
            query = query.filter_by(ticker=ticker.upper())

        predictions = query.order_by(MLPrediction.prediction_timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify(
            {
                "predictions": [pred.to_dict() for pred in predictions.items],
                "total": predictions.total,
                "pages": predictions.pages,
                "current_page": page,
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ml_bp.route("/predictions/top", methods=["GET"])
@jwt_required()
def get_top_predictions():
    """Get top probability predictions for user"""
    try:
        current_user_id = get_jwt_identity()
        limit = request.args.get("limit", 10, type=int)
        min_probability = request.args.get("min_probability", 0.7, type=float)

        predictions = (
            MLPrediction.query.filter(
                MLPrediction.user_id == current_user_id,
                MLPrediction.probability_score >= min_probability,
            )
            .order_by(MLPrediction.probability_score.desc())
            .limit(limit)
            .all()
        )

        return jsonify(
            {"top_predictions": [pred.to_dict() for pred in predictions], "count": len(predictions)}
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ml_bp.route("/models/status", methods=["GET"])
@jwt_required()
def get_model_status():
    """Get ML model health status"""
    try:
        response = requests.get(f"{ML_ENGINE_URL}/health", timeout=10)

        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return (
                jsonify({"status": "unhealthy", "error": "ML engine not responding"}),
                503,
            )

    except requests.exceptions.RequestException:
        return (
            jsonify({"status": "unavailable", "error": "ML engine not reachable"}),
            503,
        )

@ml_bp.route("/models/retrain", methods=["POST"])
@jwt_required()
def retrain_model():
    """Trigger model retraining"""
    try:
        current_user_id = get_jwt_identity()

        # Only allow admin users to retrain (implement role check)
        user = User.query.get(current_user_id)
        if not user or user.preferences.get("role") != "admin":
            return jsonify({"error": "Insufficient permissions"}), 403

        response = requests.post(f"{ML_ENGINE_URL}/train", timeout=300)

        if response.status_code == 200:
            return (
                jsonify(
                    {
                        "message": "Model retraining initiated",
                        "details": response.json(),
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": "Failed to initiate retraining",
                        "details": response.text,
                    }
                ),
                500,
            )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"ML engine error: {str(e)}"}), 500

@ml_bp.route("/features/<ticker>", methods=["GET"])
@jwt_required()
def get_feature_importance(ticker):
    """Get feature importance for a specific ticker"""
    try:
        current_user_id = get_jwt_identity()

        # Get latest prediction for this ticker
        prediction = MLPrediction.query.filter_by(
            user_id=current_user_id, ticker=ticker.upper()
        ).order_by(MLPrediction.prediction_timestamp.desc()).first()

        if not prediction:
            return jsonify({"error": "No predictions found for this ticker"}), 404

        return jsonify(
            {
                "ticker": ticker.upper(),
                "feature_importance": prediction.contributing_factors,
                "prediction_timestamp": prediction.prediction_timestamp.isoformat(),
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ml_bp.route("/performance", methods=["GET"])
@jwt_required()
def get_model_performance():
    """Get model performance metrics"""
    try:
        response = requests.get(f"{ML_ENGINE_URL}/performance", timeout=10)

        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return (
                jsonify({"error": "Unable to fetch performance metrics"}), 500
            )

    except requests.exceptions.RequestException:
        return (
            jsonify({"error": "ML engine not available"}), 503
        )

@ml_bp.route("/batch-predict", methods=["POST"])
@jwt_required()
def batch_predict():
    """Batch prediction for multiple tickers"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data or not data.get("tickers"):
            return jsonify({"error": "Tickers list is required"}), 400

        tickers = [ticker.upper() for ticker in data["tickers"]]
        predictions = []

        for ticker in tickers:
            try:
                # Prepare individual prediction request
                ml_request = {
                    "ticker": ticker,
                    "newsletter_data": data.get("newsletter_data", {}),
                    "market_data": data.get("market_data", {}),
                    "portfolio_data": data.get("portfolio_data", {}),
                }

                response = requests.post(
                    f"{ML_ENGINE_URL}/predict", json=ml_request, timeout=10
                )

                if response.status_code == 200:
                    prediction_data = response.json()

                    # Store prediction
                    prediction = MLPrediction(
                        user_id=current_user_id,
                        ticker=ticker,
                        probability_score=prediction_data["probability"],
                        confidence_lower=prediction_data["confidence_interval"][0],
                        confidence_upper=prediction_data["confidence_interval"][1],
                        contributing_factors=prediction_data["feature_importance"],
                        model_version=prediction_data.get("model_version", "1.0.0"),
                    )

                    db.session.add(prediction)
                    predictions.append(prediction.to_dict())

            except Exception as e:
                # Continue with other tickers if one fails
                continue

        db.session.commit()

        return (
            jsonify(
                {
                    "predictions": predictions,
                    "processed_count": len(predictions),
                    "requested_count": len(tickers),
                }
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# Helper functions
def create_fallback_prediction(user_id, ticker, data):
    """Create a basic fallback prediction when ML engine is unavailable"""
    # Simple fallback logic based on newsletter sentiment
    newsletter_data = data.get("newsletter_data", {})
    sentiment_score = newsletter_data.get("sentiment_score", 0)

    # Basic probability calculation
    base_probability = 0.5
    sentiment_adjustment = sentiment_score * 0.2  # Adjust by sentiment

    probability = max(0.1, min(0.9, base_probability + sentiment_adjustment))

    prediction = MLPrediction(
        user_id=user_id,
        ticker=ticker,
        probability_score=probability,
        confidence_lower=probability - 0.1,
        confidence_upper=probability + 0.1,
        contributing_factors={
            "newsletter_sentiment": sentiment_score,
            "fallback_mode": True,
        },
        model_version="fallback-1.0",
    )

    db.session.add(prediction)
    db.session.commit()

    return prediction


