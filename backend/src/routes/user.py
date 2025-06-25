from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.main import db
from src.models.user import User
from src.models.user_alert import UserAlert
from src.models.ml_prediction import MLPrediction
from src.models.newsletter import Newsletter
from src.models.portfolio_position import PortfolioPosition

user_bp = Blueprint("user", __name__)

@user_bp.route("/", methods=["GET"])
@jwt_required()
def get_current_user():
    """Get current user information"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({"user": user.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route("/alerts", methods=["GET"])
@jwt_required()
def get_user_alerts():
    """Get user alerts"""
    try:
        current_user_id = get_jwt_identity()
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 20, type=int)
        unread_only = request.args.get("unread_only", "false").lower() == "true"

        query = UserAlert.query.filter_by(user_id=current_user_id)

        if unread_only:
            query = query.filter_by(is_read=False)

        alerts = query.order_by(UserAlert.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify(
            {
                "alerts": [alert.to_dict() for alert in alerts.items],
                "total": alerts.total,
                "pages": alerts.pages,
                "current_page": page,
                "unread_count": UserAlert.query.filter_by(
                    user_id=current_user_id, is_read=False
                ).count(),
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route("/alerts", methods=["POST"])
@jwt_required()
def create_alert():
    """Create a new user alert"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()

        if not data or not data.get("message"):
            return jsonify({"error": "Message is required"}), 400

        alert = UserAlert(
            user_id=current_user_id,
            alert_type=data.get("alert_type", "general"),
            ticker=data.get("ticker"),
            message=data["message"],
        )

        db.session.add(alert)
        db.session.commit()

        return jsonify(
            {"alert": alert.to_dict(), "message": "Alert created successfully"}
        ), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@user_bp.route("/alerts/<alert_id>/read", methods=["PUT"])
@jwt_required()
def mark_alert_read(alert_id):
    """Mark an alert as read"""
    try:
        current_user_id = get_jwt_identity()

        alert = UserAlert.query.filter_by(
            id=alert_id, user_id=current_user_id
        ).first()

        if not alert:
            return jsonify({"error": "Alert not found"}), 404

        alert.is_read = True
        db.session.commit()

        return jsonify(
            {"alert": alert.to_dict(), "message": "Alert marked as read"}
        ), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@user_bp.route("/alerts/mark-all-read", methods=["PUT"])
@jwt_required()
def mark_all_alerts_read():
    """Mark all alerts as read for the current user"""
    try:
        current_user_id = get_jwt_identity()

        UserAlert.query.filter_by(
            user_id=current_user_id, is_read=False
        ).update({"is_read": True})

        db.session.commit()

        return jsonify({"message": "All alerts marked as read"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@user_bp.route("/alerts/<alert_id>", methods=["DELETE"])
@jwt_required()
def delete_alert(alert_id):
    """Delete a user alert"""
    try:
        current_user_id = get_jwt_identity()

        alert = UserAlert.query.filter_by(
            id=alert_id, user_id=current_user_id
        ).first()

        if not alert:
            return jsonify({"error": "Alert not found"}), 404

        db.session.delete(alert)
        db.session.commit()

        return jsonify({"message": "Alert deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@user_bp.route("/preferences", methods=["PUT"])
@jwt_required()
def update_preferences():
    """Update user preferences"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)

        if not user:
            return jsonify({"error": "User not found"}), 404

        data = request.get_json()

        if not data:
            return jsonify({"error": "Preferences data is required"}), 400

        # Update preferences (merge with existing)
        current_preferences = user.preferences or {}
        current_preferences.update(data)
        user.preferences = current_preferences

        db.session.commit()

        return jsonify(
            {"preferences": user.preferences, "message": "Preferences updated successfully"}
        ), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@user_bp.route("/dashboard-stats", methods=["GET"])
@jwt_required()
def get_dashboard_stats():
    """Get dashboard statistics for the current user"""
    try:
        current_user_id = get_jwt_identity()

        # Get counts
        predictions_count = MLPrediction.query.filter_by(
            user_id=current_user_id
        ).count()
        newsletters_count = Newsletter.query.filter_by(
            user_id=current_user_id
        ).count()
        positions_count = PortfolioPosition.query.filter_by(
            user_id=current_user_id
        ).count()
        unread_alerts = UserAlert.query.filter_by(
            user_id=current_user_id, is_read=False
        ).count()

        # Get recent high-probability predictions
        recent_predictions = (
            MLPrediction.query.filter(
                MLPrediction.user_id == current_user_id,
                MLPrediction.probability_score >= 0.8,
            )
            .order_by(MLPrediction.prediction_timestamp.desc())
            .limit(5)
            .all()
        )

        return jsonify(
            {
                "stats": {
                    "predictions_count": predictions_count,
                    "newsletters_count": newsletters_count,
                    "positions_count": positions_count,
                    "unread_alerts": unread_alerts,
                },
                "recent_high_probability_predictions": [
                    pred.to_dict() for pred in recent_predictions
                ],
            }
        ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


