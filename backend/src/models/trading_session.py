from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID
import os

class TradingSession(db.Model):
    __tablename__ = 'trading_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), 
                        db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.String(36), db.ForeignKey('ml_predictions.id'))
    ticker = db.Column(db.String(10), nullable=False)
    entry_price = db.Column(db.Numeric(10, 4))
    exit_price = db.Column(db.Numeric(10, 4))
    quantity = db.Column(db.Numeric(10, 2))
    pnl = db.Column(db.Numeric(12, 2))
    trade_success = db.Column(db.Boolean)  # For ML training data
    entry_timestamp = db.Column(db.DateTime)
    exit_timestamp = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<TradingSession {self.ticker}: {self.pnl}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'prediction_id': str(self.prediction_id) if self.prediction_id else None,
            'ticker': self.ticker,
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'exit_price': float(self.exit_price) if self.exit_price else None,
            'quantity': float(self.quantity) if self.quantity else None,
            'pnl': float(self.pnl) if self.pnl else None,
            'trade_success': self.trade_success,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
