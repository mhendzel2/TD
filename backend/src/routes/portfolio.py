from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.models.user import db, PortfolioPosition, User
from datetime import datetime
import numpy as np

portfolio_bp = Blueprint('portfolio', __name__)

@portfolio_bp.route('/', methods=['GET'])
@jwt_required()
def get_portfolio():
    """Get current portfolio summary"""
    try:
        current_user_id = get_jwt_identity()
        
        positions = PortfolioPosition.query.filter_by(user_id=current_user_id).all()
        
        # Calculate portfolio summary
        total_value = sum(float(pos.market_value or 0) for pos in positions)
        total_pnl = sum(float(pos.unrealized_pnl or 0) for pos in positions)
        
        # Calculate portfolio Greeks
        portfolio_greeks = calculate_portfolio_greeks(positions)
        
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(positions)
        
        return jsonify({
            'positions': [pos.to_dict() for pos in positions],
            'summary': {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'position_count': len(positions),
                'greeks': portfolio_greeks,
                'risk_metrics': risk_metrics
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/positions', methods=['GET'])
@jwt_required()
def get_positions():
    """Get all portfolio positions"""
    try:
        current_user_id = get_jwt_identity()
        ticker = request.args.get('ticker')
        position_type = request.args.get('position_type')
        
        query = PortfolioPosition.query.filter_by(user_id=current_user_id)
        
        if ticker:
            query = query.filter_by(ticker=ticker.upper())
        
        if position_type:
            query = query.filter_by(position_type=position_type.upper())
        
        positions = query.order_by(PortfolioPosition.updated_at.desc()).all()
        
        return jsonify({
            'positions': [pos.to_dict() for pos in positions]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/positions', methods=['POST'])
@jwt_required()
def add_position():
    """Add or update a portfolio position"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or not data.get('ticker'):
            return jsonify({'error': 'Ticker is required'}), 400
        
        ticker = data['ticker'].upper()
        position_type = data.get('position_type', 'STOCK').upper()
        
        # Check if position already exists
        existing_position = PortfolioPosition.query.filter_by(
            user_id=current_user_id,
            ticker=ticker,
            position_type=position_type
        ).first()
        
        if existing_position:
            # Update existing position
            update_position_data(existing_position, data)
            position = existing_position
        else:
            # Create new position
            position = PortfolioPosition(
                user_id=current_user_id,
                ticker=ticker,
                position_type=position_type
            )
            update_position_data(position, data)
            db.session.add(position)
        
        db.session.commit()
        
        return jsonify({
            'position': position.to_dict(),
            'message': 'Position updated successfully'
        }), 201 if not existing_position else 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/positions/<position_id>', methods=['PUT'])
@jwt_required()
def update_position(position_id):
    """Update a specific position"""
    try:
        current_user_id = get_jwt_identity()
        
        position = PortfolioPosition.query.filter_by(
            id=position_id,
            user_id=current_user_id
        ).first()
        
        if not position:
            return jsonify({'error': 'Position not found'}), 404
        
        data = request.get_json()
        update_position_data(position, data)
        
        db.session.commit()
        
        return jsonify({
            'position': position.to_dict(),
            'message': 'Position updated successfully'
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/positions/<position_id>', methods=['DELETE'])
@jwt_required()
def delete_position(position_id):
    """Delete a portfolio position"""
    try:
        current_user_id = get_jwt_identity()
        
        position = PortfolioPosition.query.filter_by(
            id=position_id,
            user_id=current_user_id
        ).first()
        
        if not position:
            return jsonify({'error': 'Position not found'}), 404
        
        db.session.delete(position)
        db.session.commit()
        
        return jsonify({'message': 'Position deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/greeks', methods=['GET'])
@jwt_required()
def get_portfolio_greeks():
    """Get portfolio Greeks summary"""
    try:
        current_user_id = get_jwt_identity()
        
        positions = PortfolioPosition.query.filter_by(user_id=current_user_id).all()
        greeks = calculate_portfolio_greeks(positions)
        
        return jsonify({'greeks': greeks}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/risk', methods=['GET'])
@jwt_required()
def get_risk_metrics():
    """Get portfolio risk metrics"""
    try:
        current_user_id = get_jwt_identity()
        
        positions = PortfolioPosition.query.filter_by(user_id=current_user_id).all()
        risk_metrics = calculate_risk_metrics(positions)
        
        return jsonify({'risk_metrics': risk_metrics}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/correlation', methods=['POST'])
@jwt_required()
def calculate_correlation():
    """Calculate correlation between portfolio and a ticker"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or not data.get('ticker'):
            return jsonify({'error': 'Ticker is required'}), 400
        
        ticker = data['ticker'].upper()
        
        # Get portfolio positions
        positions = PortfolioPosition.query.filter_by(user_id=current_user_id).all()
        
        # Calculate correlation (simplified implementation)
        correlation = calculate_portfolio_ticker_correlation(positions, ticker)
        
        return jsonify({
            'ticker': ticker,
            'correlation': correlation,
            'interpretation': interpret_correlation(correlation)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/exposure', methods=['GET'])
@jwt_required()
def get_sector_exposure():
    """Get portfolio sector exposure"""
    try:
        current_user_id = get_jwt_identity()
        
        positions = PortfolioPosition.query.filter_by(user_id=current_user_id).all()
        
        # Group positions by ticker and calculate exposure
        ticker_exposure = {}
        total_value = 0
        
        for position in positions:
            ticker = position.ticker
            value = float(position.market_value or 0)
            
            if ticker not in ticker_exposure:
                ticker_exposure[ticker] = 0
            
            ticker_exposure[ticker] += value
            total_value += value
        
        # Calculate percentages
        exposure_percentages = {}
        for ticker, value in ticker_exposure.items():
            exposure_percentages[ticker] = round((value / total_value * 100), 2) if total_value > 0 else 0
        
        return jsonify({
            'ticker_exposure': exposure_percentages,
            'total_value': total_value,
            'position_count': len(ticker_exposure)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper functions
def update_position_data(position, data):
    """Update position with provided data"""
    if 'quantity' in data:
        position.quantity = data['quantity']
    if 'avg_cost' in data:
        position.avg_cost = data['avg_cost']
    if 'current_price' in data:
        position.current_price = data['current_price']
    if 'market_value' in data:
        position.market_value = data['market_value']
    if 'unrealized_pnl' in data:
        position.unrealized_pnl = data['unrealized_pnl']
    if 'delta' in data:
        position.delta = data['delta']
    if 'gamma' in data:
        position.gamma = data['gamma']
    if 'theta' in data:
        position.theta = data['theta']
    if 'vega' in data:
        position.vega = data['vega']
    
    # Calculate market value if not provided
    if position.quantity and position.current_price and not position.market_value:
        position.market_value = float(position.quantity) * float(position.current_price)
    
    # Calculate unrealized PnL if not provided
    if (position.quantity and position.current_price and position.avg_cost 
        and not position.unrealized_pnl):
        position.unrealized_pnl = (float(position.current_price) - float(position.avg_cost)) * float(position.quantity)

def calculate_portfolio_greeks(positions):
    """Calculate portfolio-level Greeks"""
    total_delta = sum(float(pos.delta or 0) * float(pos.quantity or 0) for pos in positions)
    total_gamma = sum(float(pos.gamma or 0) * float(pos.quantity or 0) for pos in positions)
    total_theta = sum(float(pos.theta or 0) * float(pos.quantity or 0) for pos in positions)
    total_vega = sum(float(pos.vega or 0) * float(pos.quantity or 0) for pos in positions)
    
    return {
        'delta': round(total_delta, 4),
        'gamma': round(total_gamma, 4),
        'theta': round(total_theta, 4),
        'vega': round(total_vega, 4)
    }

def calculate_risk_metrics(positions):
    """Calculate portfolio risk metrics"""
    if not positions:
        return {
            'var_95': 0,
            'max_drawdown': 0,
            'concentration_risk': 0,
            'beta': 1.0
        }
    
    # Calculate basic risk metrics
    values = [float(pos.market_value or 0) for pos in positions]
    total_value = sum(values)
    
    # Concentration risk (Herfindahl index)
    if total_value > 0:
        weights = [value / total_value for value in values]
        concentration_risk = sum(w ** 2 for w in weights)
    else:
        concentration_risk = 0
    
    # Simplified VaR calculation (would need historical data in practice)
    portfolio_volatility = 0.15  # Assume 15% volatility
    var_95 = total_value * 1.645 * portfolio_volatility  # 95% VaR
    
    return {
        'var_95': round(var_95, 2),
        'max_drawdown': 0,  # Would need historical data
        'concentration_risk': round(concentration_risk, 4),
        'beta': 1.0,  # Would need market data
        'total_exposure': round(total_value, 2)
    }

def calculate_portfolio_ticker_correlation(positions, ticker):
    """Calculate correlation between portfolio and a specific ticker"""
    # Simplified correlation calculation
    # In practice, would need historical price data
    
    # Check if ticker is already in portfolio
    ticker_positions = [pos for pos in positions if pos.ticker == ticker]
    
    if ticker_positions:
        ticker_value = sum(float(pos.market_value or 0) for pos in ticker_positions)
        total_value = sum(float(pos.market_value or 0) for pos in positions)
        
        if total_value > 0:
            weight = ticker_value / total_value
            # Higher weight = higher correlation
            return min(0.95, weight * 2)  # Cap at 0.95
    
    # Default correlation for tickers not in portfolio
    return 0.1

def interpret_correlation(correlation):
    """Interpret correlation value"""
    if correlation >= 0.7:
        return "High positive correlation"
    elif correlation >= 0.3:
        return "Moderate positive correlation"
    elif correlation >= -0.3:
        return "Low correlation"
    elif correlation >= -0.7:
        return "Moderate negative correlation"
    else:
        return "High negative correlation"

