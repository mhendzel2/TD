from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
import os

class NewsletterSource(db.Model):
    __tablename__ = 'newsletter_sources'
    
    id = db.Column(UUID(as_uuid=True) if 'postgresql' in os.getenv('DATABASE_URL', '') else db.String(36), 
                   primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    domain = db.Column(db.String(255), nullable=False)
    priority = db.Column(db.Integer, default=5)
    credibility_score = db.Column(db.Numeric(3, 2), default=0.50)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    newsletter_analyses = db.relationship('NewsletterAnalysis', backref='source', lazy=True)

    def __repr__(self):
        return f'<NewsletterSource {self.name}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'name': self.name,
            'domain': self.domain,
            'priority': self.priority,
            'credibility_score': float(self.credibility_score),
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


