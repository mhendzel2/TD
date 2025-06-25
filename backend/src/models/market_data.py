from src.main import db
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import JSON
import os

# Use JSON for SQLite compatibility, JSONB for PostgreSQL
JsonType = JSONB if 'postgresql' in os.getenv('DATABASE_URL', '') else JSON

class MarketData(db.Model):
    __tablename__ = 'market_data'
    
    id = db.Column(UUID(as_uuid=True) if 'postgresql' in os.getenv('DATABASE_URL', '') else db.String(36), 
                   primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker = db.Column(db.String(10), nullable=False)
    data_type = db.Column(db.String(50), nullable=False)  # 'options_flow', 'volume', 'gex', etc.
    data_value = db.Column(JsonType, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<MarketData {self.ticker}: {self.data_type}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'ticker': self.ticker,
            'data_type': self.data_type,
            'data_value': self.data_value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


