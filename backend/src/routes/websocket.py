from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from src.services.websocket_manager import get_websocket_manager
from datetime import datetime

websocket_bp = Blueprint('websocket', __name__)

@websocket_bp.route('/broadcast/prediction', methods=['POST'])
@jwt_required()
def broadcast_prediction():
    """Broadcast prediction update to user"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Data is required'}), 400
        
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        # Send prediction update
        websocket_manager.send_prediction_update(current_user_id, data)
        
        return jsonify({
            'message': 'Prediction update broadcasted',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@websocket_bp.route('/broadcast/portfolio', methods=['POST'])
@jwt_required()
def broadcast_portfolio():
    """Broadcast portfolio update to user"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Data is required'}), 400
        
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        # Send portfolio update
        websocket_manager.send_portfolio_update(current_user_id, data)
        
        return jsonify({
            'message': 'Portfolio update broadcasted',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@websocket_bp.route('/broadcast/alert', methods=['POST'])
@jwt_required()
def broadcast_alert():
    """Broadcast alert to user"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Alert data is required'}), 400
        
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        # Send alert
        websocket_manager.send_alert(current_user_id, data)
        
        return jsonify({
            'message': 'Alert broadcasted',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@websocket_bp.route('/broadcast/market', methods=['POST'])
@jwt_required()
def broadcast_market():
    """Broadcast market update to all users"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Market data is required'}), 400
        
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        # Send market update
        websocket_manager.send_market_update(data)
        
        return jsonify({
            'message': 'Market update broadcasted',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@websocket_bp.route('/broadcast/newsletter', methods=['POST'])
@jwt_required()
def broadcast_newsletter():
    """Broadcast newsletter update to user"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Newsletter data is required'}), 400
        
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        # Send newsletter update
        websocket_manager.send_newsletter_update(current_user_id, data)
        
        return jsonify({
            'message': 'Newsletter update broadcasted',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@websocket_bp.route('/connections', methods=['GET'])
@jwt_required()
def get_connections():
    """Get WebSocket connection statistics"""
    try:
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        stats = websocket_manager.get_connection_stats()
        
        return jsonify({
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@websocket_bp.route('/test', methods=['POST'])
@jwt_required()
def test_websocket():
    """Test WebSocket functionality"""
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        websocket_manager = get_websocket_manager()
        if not websocket_manager:
            return jsonify({'error': 'WebSocket not available'}), 503
        
        # Send test message
        test_data = {
            'message': data.get('message', 'Test message from WebSocket'),
            'user_id': current_user_id,
            'test_timestamp': datetime.utcnow().isoformat()
        }
        
        websocket_manager.broadcast_to_user(current_user_id, 'test_message', test_data)
        
        return jsonify({
            'message': 'Test message sent',
            'data': test_data
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

