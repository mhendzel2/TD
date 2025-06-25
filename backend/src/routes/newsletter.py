from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.main import db
from src.models.newsletter import Newsletter
from src.services.newsletter_processor import NewsletterProcessor

newsletter_bp = Blueprint("newsletter", __name__, url_prefix="/api/newsletter")

@newsletter_bp.route("/analyze", methods=["POST"])
@jwt_required()
def analyze_newsletter():
    """Analyze a newsletter from provided content or URL"""
    user_id = get_jwt_identity()
    data = request.get_json()
    content = data.get("content")
    url = data.get("url")
    source = data.get("source", "Manual Input")
    title = data.get("title", "Untitled Newsletter")
    publish_date = data.get("publish_date")

    if not content and not url:
        return jsonify({"error": "Either content or URL must be provided"}), 400

    try:
        processor = NewsletterProcessor(user_id)
        if url:
            analysis_result = processor.process_from_url(url, source, title, publish_date)
        else:
            analysis_result = processor.process_from_content(content, source, title, publish_date)
        
        new_newsletter = Newsletter(
            user_id=user_id,
            title=analysis_result["title"],
            source=analysis_result["source"],
            publish_date=analysis_result["publish_date"],
            content=analysis_result["content"],
            processed_content=analysis_result["processed_content"],
            sentiment_score=analysis_result["sentiment_score"],
            sentiment_confidence=analysis_result["sentiment_confidence"],
            bullish_terms=analysis_result["bullish_terms"],
            bearish_terms=analysis_result["bearish_terms"],
            tickers=analysis_result["tickers"],
            key_phrases=analysis_result["key_phrases"],
            priority_score=analysis_result["priority_score"],
            analysis_status="Completed"
        )
        db.session.add(new_newsletter)
        db.session.commit()

        return jsonify(new_newsletter.to_dict()), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@newsletter_bp.route("/<newsletter_id>", methods=["GET"])
@jwt_required()
def get_newsletter(newsletter_id):
    """Get a specific newsletter by ID"""
    user_id = get_jwt_identity()
    newsletter = Newsletter.query.filter_by(id=newsletter_id, user_id=user_id).first()
    if not newsletter:
        return jsonify({"error": "Newsletter not found"}), 404
    return jsonify(newsletter.to_dict()), 200

@newsletter_bp.route("/", methods=["GET"])
@jwt_required()
def get_all_newsletters():
    """Get all newsletters for the current user"""
    user_id = get_jwt_identity()
    newsletters = Newsletter.query.filter_by(user_id=user_id).all()
    return jsonify([n.to_dict() for n in newsletters]), 200


