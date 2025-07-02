from src.main import db
from datetime import datetime
import uuid

class PortfolioPosition(db.Model):
    __tablename__ = 'portfolio_positions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), 
                        nullable=False, index=True)
    ticker = db.Column(db.String(10), nullable=False)
    position_type = db.Column(db.String(20))  # 'STOCK', 'CALL', 'PUT'
    quantity = db.Column(db.Numeric(10, 2))
    avg_cost = db.Column(db.Numeric(10, 4))
    current_price = db.Column(db.Numeric(10, 4))
    market_value = db.Column(db.Numeric(12, 2))
    unrealized_pnl = db.Column(db.Numeric(12, 2))
    delta = db.Column(db.Numeric(8, 4))
    gamma = db.Column(db.Numeric(8, 4))
    theta = db.Column(db.Numeric(8, 4))
    vega = db.Column(db.Numeric(8, 4))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        # Index for efficiently querying all positions for a user.
        db.Index('ix_portfolio_positions_user_id_ticker', 'user_id', 'ticker'),
        # Ensure a user can only have one position entry per ticker.
        db.UniqueConstraint('user_id', 'ticker', name='uq_user_ticker_position'),
    )

    def __repr__(self):
        return f'<PortfolioPosition {self.ticker}: {self.quantity}>'

    def to_dict(self):
        return {
            'id': str(self.id),
            'user_id': str(self.user_id),
            'ticker': self.ticker,
            'position_type': self.position_type,
            'quantity': float(self.quantity) if self.quantity else None,
            'avg_cost': float(self.avg_cost) if self.avg_cost else None,
            'current_price': float(self.current_price) if self.current_price else None,
            'market_value': float(self.market_value) if self.market_value else None,
            'unrealized_pnl': float(self.unrealized_pnl) if self.unrealized_pnl else None,
            'delta': float(self.delta) if self.delta else None,
            'gamma': float(self.gamma) if self.gamma else None,
            'theta': float(self.theta) if self.theta else None,
            'vega': float(self.vega) if self.vega else None,
            'updated_at': self.updated_at.isoformat()
        }
