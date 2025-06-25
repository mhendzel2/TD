from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy import JSON, Text
import os

# Use JSON for SQLite compatibility, JSONB for PostgreSQL
JsonType = JSONB if os.getenv("DATABASE_URL", "").startswith("postgresql") else JSON
ArrayType = ARRAY if os.getenv("DATABASE_URL", "").startswith("postgresql") else Text

class Newsletter(db.Model):
    __tablename__ = "newsletters"
    
    id = db.Column(UUID(as_uuid=True) if os.getenv("DATABASE_URL", "").startswith("postgresql") else db.String(36), 
                   primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(UUID(as_uuid=True) if os.getenv("DATABASE_URL", "").startswith("postgresql") else db.String(36), 
                        db.ForeignKey("users.id"), nullable=True) # User ID can be null for public newsletters
    title = db.Column(db.String(255), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    publish_date = db.Column(db.DateTime, nullable=False)
    content = db.Column(db.Text, nullable=False)
    processed_content = db.Column(db.Text, nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    sentiment_confidence = db.Column(db.Float, nullable=True)
    bullish_terms = db.Column(ArrayType(db.String) if os.getenv("DATABASE_URL", "").startswith("postgresql") else Text, nullable=True)
    bearish_terms = db.Column(ArrayType(db.String) if os.getenv("DATABASE_URL", "").startswith("postgresql") else Text, nullable=True)
    tickers = db.Column(ArrayType(db.String) if os.getenv("DATABASE_URL", "").startswith("postgresql") else Text, nullable=True)
    key_phrases = db.Column(ArrayType(db.String) if os.getenv("DATABASE_URL", "").startswith("postgresql") else Text, nullable=True)
    priority_score = db.Column(db.Float, nullable=True)
    analysis_status = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Newsletter {self.title} from {self.source}>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "title": self.title,
            "source": self.source,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "content": self.content,
            "processed_content": self.processed_content,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "bullish_terms": self.bullish_terms if isinstance(self.bullish_terms, list) else (self.bullish_terms.split(",") if self.bullish_terms else []),
            "bearish_terms": self.bearish_terms if isinstance(self.bearish_terms, list) else (self.bearish_terms.split(",") if self.bearish_terms else []),
            "tickers": self.tickers if isinstance(self.tickers, list) else (self.tickers.split(",") if self.tickers else []),
            "key_phrases": self.key_phrases if isinstance(self.key_phrases, list) else (self.key_phrases.split(",") if self.key_phrases else []),
            "priority_score": self.priority_score,
            "analysis_status": self.analysis_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


