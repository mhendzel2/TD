from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
import os

class UserAlert(db.Model):
    __tablename__ = 'user_alerts'
    
    id = db.Column(UUID(as_uuid=True) if 'postgresql' in os.getenv('DATABASE_URL', '') else db.String(36), 
                   primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(UUID(as_uuid=True) if 'postgresql' in os.getenv('DATABASE_URL', '') else db.String(36), 
                        db.ForeignKey('users.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # 'high_probability', 'portfolio_risk', etc.
    ticker = db.Column(db.String(10))
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<UserAlert {self.alert_type}: {self.ticker}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'alert_type': self.alert_type,
            'ticker': self.ticker,
            'message': self.message,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


