from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import JSON
import os

# Use JSON for SQLite compatibility, JSONB for PostgreSQL
JsonType = JSONB if os.getenv("DATABASE_URL", "").startswith("postgresql") else JSON

class MLPrediction(db.Model):
    __tablename__ = "ml_predictions"
    
    id = db.Column(UUID(as_uuid=True) if os.getenv("DATABASE_URL", "").startswith("postgresql") else db.String(36), 
                   primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(UUID(as_uuid=True) if os.getenv("DATABASE_URL", "").startswith("postgresql") else db.String(36), 
                        db.ForeignKey("users.id"), nullable=False)
    newsletter_id = db.Column(UUID(as_uuid=True) if os.getenv("DATABASE_URL", "").startswith("postgresql") else db.String(36), 
                                      db.ForeignKey("newsletters.id")) # Corrected foreign key
    ticker = db.Column(db.String(10), nullable=False)
    prediction_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    probability_score = db.Column(db.Numeric(5, 4), nullable=False)
    confidence_lower = db.Column(db.Numeric(5, 4))
    confidence_upper = db.Column(db.Numeric(5, 4))
    contributing_factors = db.Column(JsonType)
    model_version = db.Column(db.String(50))
    
    def __repr__(self):
        return f"<MLPrediction {self.ticker}: {self.probability_score}>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "newsletter_id": str(self.newsletter_id) if self.newsletter_id else None,
            "ticker": self.ticker,
            "prediction_timestamp": self.prediction_timestamp.isoformat() if self.prediction_timestamp else None,
            "probability_score": float(self.probability_score),
            "confidence_lower": float(self.confidence_lower),
            "confidence_upper": float(self.confidence_upper),
            "contributing_factors": self.contributing_factors,
            "model_version": self.model_version
        }


