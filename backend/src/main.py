import os
import sys
import secrets
from datetime import timedelta

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_cors import CORS

from werkzeug.exceptions import HTTPException

from src.config.logging_config import setup_logging


# Initialize extensions outside of create_app
db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()
cors = CORS()

celery = None

def create_app(config_object=None):
    app = Flask(__name__)

    # Load configuration
    if config_object:
        app.config.from_object(config_object)
    else:
        app.config.from_mapping(
            SECRET_KEY=os.environ.get("SECRET_KEY", secrets.token_urlsafe(32)),
            SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///trading_dashboard.db"),        SQLALCHEMY_TRACK_MODIFICATIONS=False,
            JWT_SECRET_KEY=os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32)),
            JWT_ACCESS_TOKEN_EXPIRES=timedelta(hours=1),
            JWT_REFRESH_TOKEN_EXPIRES=timedelta(days=30),
            CORS_HEADERS=["Content-Type", "Authorization"],
            CORS_RESOURCES={
                r"/api/*": {
                    "origins": os.environ.get("CORS_ORIGINS", "*").split(","),
                    "supports_credentials": True
                }
            },

        )

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cors.init_app(app)

    # Setup logging
    setup_logging(app)

    # Setup Celery


    # Global error handler
    @app.errorhandler(HTTPException)
    def handle_exception(e):
        """Return JSON instead of HTML for HTTP errors."""
        # start with the correct headers and status code from the error
        response = e.get_response()
        # replace the body with JSON
        response.data = jsonify({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_exception(e):
        app.logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"code": 500, "name": "Internal Server Error", "description": "An unexpected error occurred."}), 500

    # Import models to ensure they are registered with SQLAlchemy
    from src.models import user, newsletter, ml_prediction, portfolio_position, user_alert, newsletter_source, market_data, trading_session

    # Register blueprints
    from src.routes.user import user_bp
    from src.routes.auth import auth_bp
    from src.routes.newsletter import newsletter_bp
    from src.routes.ml import ml_bp
    from src.routes.portfolio import portfolio_bp
    from src.routes.websocket import websocket_bp

    app.register_blueprint(user_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(newsletter_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(websocket_bp)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "ok", "message": "Trading Dashboard API is running"}), 200

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)


