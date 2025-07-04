from src.main import db
from datetime import datetime
import uuid
from sqlalchemy import JSON

class MarketData(db.Model):
    __tablename__ = 'market_data'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    ticker = db.Column(db.String(10), nullable=False)
    data_type = db.Column(db.String(50), nullable=False)  # 'options_flow', 'volume', 'gex', etc.
    data_value = db.Column(JSON, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Index for efficiently querying data for a specific ticker, ordered by time.
        db.Index('ix_market_data_ticker_timestamp_desc', 'ticker', db.desc('timestamp')),
    )

    def __repr__(self):
        return f'<MarketData {self.ticker}: {self.data_type}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'ticker': self.ticker,
            'data_type': self.data_type,
            'data_value': self.data_value,
            'timestamp': self.timestamp.isoformat()
        }
