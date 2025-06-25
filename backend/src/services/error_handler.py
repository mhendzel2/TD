import logging
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from functools import wraps
from flask import jsonify, request, current_app
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/error.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Centralized error handling and reporting service.
    Provides consistent error responses and logging across the application.
    """
    
    # Error codes and messages
    ERROR_CODES = {
        'VALIDATION_ERROR': {
            'code': 'E001',
            'message': 'Input validation failed',
            'status_code': 400
        },
        'AUTHENTICATION_ERROR': {
            'code': 'E002',
            'message': 'Authentication failed',
            'status_code': 401
        },
        'AUTHORIZATION_ERROR': {
            'code': 'E003',
            'message': 'Insufficient permissions',
            'status_code': 403
        },
        'NOT_FOUND_ERROR': {
            'code': 'E004',
            'message': 'Resource not found',
            'status_code': 404
        },
        'RATE_LIMIT_ERROR': {
            'code': 'E005',
            'message': 'Rate limit exceeded',
            'status_code': 429
        },
        'DATABASE_ERROR': {
            'code': 'E006',
            'message': 'Database operation failed',
            'status_code': 500
        },
        'EXTERNAL_API_ERROR': {
            'code': 'E007',
            'message': 'External service unavailable',
            'status_code': 502
        },
        'ML_ENGINE_ERROR': {
            'code': 'E008',
            'message': 'ML prediction service error',
            'status_code': 503
        },
        'IBKR_CONNECTION_ERROR': {
            'code': 'E009',
            'message': 'IBKR connection failed',
            'status_code': 503
        },
        'DISCORD_BOT_ERROR': {
            'code': 'E010',
            'message': 'Discord bot service error',
            'status_code': 503
        },
        'INTERNAL_SERVER_ERROR': {
            'code': 'E999',
            'message': 'Internal server error',
            'status_code': 500
        }
    }
    
    @staticmethod
    def create_error_response(error_type: str, 
                            details: Optional[str] = None,
                            data: Optional[Dict] = None,
                            request_id: Optional[str] = None) -> Tuple[Dict, int]:
        """
        Create standardized error response.
        
        Args:
            error_type: Error type from ERROR_CODES
            details: Additional error details
            data: Additional error data
            request_id: Request ID for tracking
        
        Returns:
            Tuple of (response_dict, status_code)
        """
        error_info = ErrorHandler.ERROR_CODES.get(error_type, ErrorHandler.ERROR_CODES['INTERNAL_SERVER_ERROR'])
        
        response = {
            'error': True,
            'error_code': error_info['code'],
            'message': error_info['message'],
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        if details:
            response['details'] = details
        
        if data:
            response['data'] = data
        
        if request_id:
            response['request_id'] = request_id
        
        # In development, include more debugging info
        if current_app and current_app.config.get('DEBUG', False):
            response['debug'] = {
                'endpoint': request.endpoint if request else None,
                'method': request.method if request else None,
                'url': request.url if request else None
            }
        
        return response, error_info['status_code']
    
    @staticmethod
    def log_error(error_type: str,
                  error: Exception,
                  context: Optional[Dict] = None,
                  user_id: Optional[str] = None,
                  request_id: Optional[str] = None):
        """
        Log error with context information.
        
        Args:
            error_type: Error type from ERROR_CODES
            error: Exception object
            context: Additional context information
            user_id: User ID if available
            request_id: Request ID for tracking
        """
        error_info = {
            'error_type': error_type,
            'error_message': str(error),
            'error_class': error.__class__.__name__,
            'timestamp': datetime.utcnow().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        if context:
            error_info['context'] = context
        
        if user_id:
            error_info['user_id'] = user_id
        
        if request_id:
            error_info['request_id'] = request_id
        
        if request:
            error_info['request_info'] = {
                'method': request.method,
                'url': request.url,
                'endpoint': request.endpoint,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent')
            }
        
        logger.error(f"Application Error: {error_type}", extra=error_info)
    
    @staticmethod
    def handle_database_error(error: Exception, operation: str = None) -> Tuple[Dict, int]:
        """Handle database-related errors."""
        ErrorHandler.log_error('DATABASE_ERROR', error, {'operation': operation})
        
        # Don't expose internal database errors in production
        details = str(error) if current_app and current_app.config.get('DEBUG', False) else None
        
        return ErrorHandler.create_error_response('DATABASE_ERROR', details)
    
    @staticmethod
    def handle_external_api_error(error: Exception, service: str = None) -> Tuple[Dict, int]:
        """Handle external API errors."""
        ErrorHandler.log_error('EXTERNAL_API_ERROR', error, {'service': service})
        
        details = f"Service '{service}' is currently unavailable" if service else None
        
        return ErrorHandler.create_error_response('EXTERNAL_API_ERROR', details)
    
    @staticmethod
    def handle_ml_engine_error(error: Exception, operation: str = None) -> Tuple[Dict, int]:
        """Handle ML engine errors."""
        ErrorHandler.log_error('ML_ENGINE_ERROR', error, {'operation': operation})
        
        return ErrorHandler.create_error_response('ML_ENGINE_ERROR', 
                                                'ML prediction service is temporarily unavailable')
    
    @staticmethod
    def handle_validation_error(error: Exception, field_errors: Dict = None) -> Tuple[Dict, int]:
        """Handle validation errors."""
        ErrorHandler.log_error('VALIDATION_ERROR', error, {'field_errors': field_errors})
        
        return ErrorHandler.create_error_response('VALIDATION_ERROR', 
                                                str(error), 
                                                {'field_errors': field_errors})

def handle_exceptions(f):
    """
    Decorator to handle exceptions in Flask routes.
    Provides consistent error handling and logging.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            response, status_code = ErrorHandler.handle_validation_error(e)
            return jsonify(response), status_code
        except ConnectionError as e:
            response, status_code = ErrorHandler.handle_external_api_error(e)
            return jsonify(response), status_code
        except Exception as e:
            # Log unexpected errors
            ErrorHandler.log_error('INTERNAL_SERVER_ERROR', e)
            
            # Return generic error response
            response, status_code = ErrorHandler.create_error_response('INTERNAL_SERVER_ERROR')
            return jsonify(response), status_code
    
    return wrapper

def register_error_handlers(app):
    """
    Register global error handlers for Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(400)
    def bad_request(error):
        response, status_code = ErrorHandler.create_error_response('VALIDATION_ERROR', 
                                                                 'Bad request')
        return jsonify(response), status_code
    
    @app.errorhandler(401)
    def unauthorized(error):
        response, status_code = ErrorHandler.create_error_response('AUTHENTICATION_ERROR')
        return jsonify(response), status_code
    
    @app.errorhandler(403)
    def forbidden(error):
        response, status_code = ErrorHandler.create_error_response('AUTHORIZATION_ERROR')
        return jsonify(response), status_code
    
    @app.errorhandler(404)
    def not_found(error):
        response, status_code = ErrorHandler.create_error_response('NOT_FOUND_ERROR')
        return jsonify(response), status_code
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        response, status_code = ErrorHandler.create_error_response('RATE_LIMIT_ERROR')
        return jsonify(response), status_code
    
    @app.errorhandler(500)
    def internal_server_error(error):
        response, status_code = ErrorHandler.create_error_response('INTERNAL_SERVER_ERROR')
        return jsonify(response), status_code
    
    @app.errorhandler(502)
    def bad_gateway(error):
        response, status_code = ErrorHandler.create_error_response('EXTERNAL_API_ERROR')
        return jsonify(response), status_code
    
    @app.errorhandler(503)
    def service_unavailable(error):
        response, status_code = ErrorHandler.create_error_response('EXTERNAL_API_ERROR', 
                                                                 'Service temporarily unavailable')
        return jsonify(response), status_code

# Custom exception classes
class TradingDashboardError(Exception):
    """Base exception for trading dashboard errors."""
    pass

class ValidationError(TradingDashboardError):
    """Raised when input validation fails."""
    pass

class AuthenticationError(TradingDashboardError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(TradingDashboardError):
    """Raised when user lacks required permissions."""
    pass

class ExternalServiceError(TradingDashboardError):
    """Raised when external service calls fail."""
    pass

class MLEngineError(TradingDashboardError):
    """Raised when ML engine operations fail."""
    pass

class IBKRConnectionError(TradingDashboardError):
    """Raised when IBKR connection fails."""
    pass

class DiscordBotError(TradingDashboardError):
    """Raised when Discord bot operations fail."""
    pass

# Global error handler instance
error_handler = ErrorHandler()

