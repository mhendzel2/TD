from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy import Text

class Newsletter(db.Model):
    __tablename__ = "newsletters"
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), 
                        db.ForeignKey("users.id"), nullable=True) # User ID can be null for public newsletters
    source_id = db.Column(db.String(36), db.ForeignKey('newsletter_sources.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    publish_date = db.Column(db.DateTime, nullable=False)
    content = db.Column(db.Text, nullable=False)
    processed_content = db.Column(db.Text, nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    sentiment_confidence = db.Column(db.Float, nullable=True)
    bullish_terms = db.Column(Text, nullable=True)
    bearish_terms = db.Column(Text, nullable=True)
    tickers = db.Column(Text, nullable=True)
    key_phrases = db.Column(Text, nullable=True)
    priority_score = db.Column(db.Float, nullable=True)
    analysis_status = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    source = db.relationship('NewsletterSource')

    __table_args__ = (
        # Index for efficiently querying newsletters for a user, ordered by date.
        db.Index('ix_newsletters_user_id_publish_date_desc', 'user_id', db.desc('publish_date')),
    )

    def __repr__(self):
        return f"<Newsletter {self.title} from {self.source.name if self.source else 'Unknown'}>"

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "source_id": str(self.source_id),
            "source": self.source.name if self.source else None,
            "title": self.title,
            "publish_date": self.publish_date.isoformat(),
            "content": self.content,
            "processed_content": self.processed_content,
            "sentiment_score": self.sentiment_score,
            "sentiment_confidence": self.sentiment_confidence,
            "bullish_terms": self.bullish_terms.split(',') if self.bullish_terms else [],
            "bearish_terms": self.bearish_terms.split(',') if self.bearish_terms else [],
            "tickers": self.tickers.split(',') if self.tickers else [],
            "key_phrases": self.key_phrases.split(',') if self.key_phrases else [],
            "priority_score": self.priority_score,
            "analysis_status": self.analysis_status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
