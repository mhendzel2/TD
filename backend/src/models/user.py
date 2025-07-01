from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy import JSON, Text
from werkzeug.security import generate_password_hash, check_password_hash
import os

# Import other models for relationships
from src.models.newsletter import Newsletter
from src.models.ml_prediction import MLPrediction
from src.models.portfolio_position import PortfolioPosition
from src.models.user_alert import UserAlert
from src.models.trading_session import TradingSession
from src.models.newsletter_source import NewsletterSource
from src.models.market_data import MarketData

# Use JSON for SQLite compatibility, JSONB for PostgreSQL
JsonType = JSONB if 'postgresql' in os.getenv('DATABASE_URL', '') else JSON
ArrayType = ARRAY if 'postgresql' in os.getenv('DATABASE_URL', '') else Text

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(UUID(as_uuid=True) if 'postgresql' in os.getenv('DATABASE_URL', '') else db.String(36), 
                   primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    preferences = db.Column(JsonType, default={})
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # Note: Relationships are defined on the 'many' side to avoid circular imports here
    # NewsletterAnalysis, MLPrediction, PortfolioPosition, UserAlert, TradingSession
    # relationships are defined in their respective model files.

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.email}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'preferences': self.preferences,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


