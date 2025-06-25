from marshmallow import Schema, fields, validate, ValidationError, pre_load
from typing import Dict, Any, List
import re
import logging

logger = logging.getLogger(__name__)

class ValidationService:
    """
    Centralized data validation and sanitization service.
    Provides comprehensive input validation for all API endpoints.
    """
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000, allow_html: bool = False) -> str:
        """
        Sanitize string input by removing dangerous characters and limiting length.
        """
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes and control characters
        value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
        
        # Remove HTML tags if not allowed
        if not allow_html:
            value = re.sub(r'<[^>]*>', '', value)
        
        # Limit length
        if len(value) > max_length:
            value = value[:max_length]
        
        # Strip whitespace
        return value.strip()
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate stock ticker format."""
        if not isinstance(ticker, str):
            return False
        
        # Basic ticker validation: 1-5 uppercase letters
        pattern = r'^[A-Z]{1,5}$'
        return bool(re.match(pattern, ticker.upper()))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """
        Validate password strength and return detailed feedback.
        """
        result = {
            'valid': True,
            'score': 0,
            'feedback': []
        }
        
        if len(password) < 8:
            result['valid'] = False
            result['feedback'].append('Password must be at least 8 characters long')
        else:
            result['score'] += 1
        
        if not re.search(r'[a-z]', password):
            result['valid'] = False
            result['feedback'].append('Password must contain at least one lowercase letter')
        else:
            result['score'] += 1
        
        if not re.search(r'[A-Z]', password):
            result['valid'] = False
            result['feedback'].append('Password must contain at least one uppercase letter')
        else:
            result['score'] += 1
        
        if not re.search(r'\d', password):
            result['valid'] = False
            result['feedback'].append('Password must contain at least one number')
        else:
            result['score'] += 1
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result['feedback'].append('Consider adding special characters for stronger security')
        else:
            result['score'] += 1
        
        if len(password) >= 12:
            result['score'] += 1
        
        return result

# Marshmallow Schemas for API validation

class UserRegistrationSchema(Schema):
    """Schema for user registration validation."""
    email = fields.Email(required=True, validate=validate.Length(max=255))
    password = fields.Str(required=True, validate=validate.Length(min=8, max=128))
    first_name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    last_name = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        """Sanitize all string inputs."""
        if 'email' in data:
            data['email'] = ValidationService.sanitize_string(data['email'], 255).lower()
        if 'first_name' in data:
            data['first_name'] = ValidationService.sanitize_string(data['first_name'], 100)
        if 'last_name' in data:
            data['last_name'] = ValidationService.sanitize_string(data['last_name'], 100)
        return data

class UserLoginSchema(Schema):
    """Schema for user login validation."""
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=validate.Length(min=1, max=128))
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'email' in data:
            data['email'] = ValidationService.sanitize_string(data['email'], 255).lower()
        return data

class NewsletterAnalysisSchema(Schema):
    """Schema for newsletter analysis validation."""
    subject = fields.Str(validate=validate.Length(max=500))
    content = fields.Str(required=True, validate=validate.Length(min=10, max=50000))
    source = fields.Str(validate=validate.Length(max=255))
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'subject' in data:
            data['subject'] = ValidationService.sanitize_string(data['subject'], 500)
        if 'content' in data:
            data['content'] = ValidationService.sanitize_string(data['content'], 50000, allow_html=True)
        if 'source' in data:
            data['source'] = ValidationService.sanitize_string(data['source'], 255)
        return data

class MLPredictionSchema(Schema):
    """Schema for ML prediction request validation."""
    ticker = fields.Str(required=True, validate=validate.Length(min=1, max=10))
    newsletter_data = fields.Dict(missing={})
    market_data = fields.Dict(missing={})
    portfolio_data = fields.Dict(missing={})
    prediction_horizon = fields.Str(validate=validate.OneOf(['1h', '1d', '1w', '1m']), missing='1d')
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'ticker' in data:
            ticker = ValidationService.sanitize_string(data['ticker'], 10).upper()
            if not ValidationService.validate_ticker(ticker):
                raise ValidationError('Invalid ticker format')
            data['ticker'] = ticker
        return data

class PortfolioPositionSchema(Schema):
    """Schema for portfolio position validation."""
    ticker = fields.Str(required=True, validate=validate.Length(min=1, max=10))
    position_type = fields.Str(required=True, validate=validate.OneOf(['STOCK', 'CALL', 'PUT']))
    quantity = fields.Decimal(required=True, validate=validate.Range(min=-999999, max=999999))
    avg_cost = fields.Decimal(validate=validate.Range(min=0, max=999999))
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'ticker' in data:
            ticker = ValidationService.sanitize_string(data['ticker'], 10).upper()
            if not ValidationService.validate_ticker(ticker):
                raise ValidationError('Invalid ticker format')
            data['ticker'] = ticker
        return data

class IBKRConnectionSchema(Schema):
    """Schema for IBKR connection validation."""
    host = fields.Str(validate=validate.Length(max=255), missing='localhost')
    port = fields.Int(validate=validate.Range(min=1, max=65535), missing=7497)
    client_id = fields.Int(validate=validate.Range(min=1, max=999), missing=1)
    account_id = fields.Str(validate=validate.Length(max=50))
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'host' in data:
            data['host'] = ValidationService.sanitize_string(data['host'], 255)
        if 'account_id' in data:
            data['account_id'] = ValidationService.sanitize_string(data['account_id'], 50)
        return data

class DiscordBotSchema(Schema):
    """Schema for Discord bot configuration validation."""
    bot_token = fields.Str(required=True, validate=validate.Length(min=50, max=100))
    server_id = fields.Str(required=True, validate=validate.Length(min=10, max=30))
    channel_id = fields.Str(required=True, validate=validate.Length(min=10, max=30))
    auto_start = fields.Bool(missing=True)
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'bot_token' in data:
            data['bot_token'] = ValidationService.sanitize_string(data['bot_token'], 100)
        if 'server_id' in data:
            data['server_id'] = ValidationService.sanitize_string(data['server_id'], 30)
        if 'channel_id' in data:
            data['channel_id'] = ValidationService.sanitize_string(data['channel_id'], 30)
        return data

class UWFlowRequestSchema(Schema):
    """Schema for Unusual Whales flow request validation."""
    ticker = fields.Str(required=True, validate=validate.Length(min=1, max=10))
    filters = fields.Dict(missing={})
    priority = fields.Str(validate=validate.OneOf(['low', 'medium', 'high']), missing='medium')
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'ticker' in data:
            ticker = ValidationService.sanitize_string(data['ticker'], 10).upper()
            if not ValidationService.validate_ticker(ticker):
                raise ValidationError('Invalid ticker format')
            data['ticker'] = ticker
        return data

class UserPreferencesSchema(Schema):
    """Schema for user preferences validation."""
    email_notifications = fields.Bool(missing=True)
    push_notifications = fields.Bool(missing=True)
    trading_alerts = fields.Bool(missing=True)
    portfolio_alerts = fields.Bool(missing=True)
    newsletter_frequency = fields.Str(validate=validate.OneOf(['real_time', 'hourly', 'daily']), missing='real_time')
    risk_tolerance = fields.Str(validate=validate.OneOf(['conservative', 'moderate', 'aggressive']), missing='moderate')
    
class AlertCreateSchema(Schema):
    """Schema for creating user alerts."""
    alert_type = fields.Str(required=True, validate=validate.OneOf(['high_probability', 'portfolio_risk', 'price_target', 'volume_spike']))
    ticker = fields.Str(validate=validate.Length(min=1, max=10))
    message = fields.Str(required=True, validate=validate.Length(min=1, max=1000))
    threshold = fields.Decimal(validate=validate.Range(min=0, max=1))
    
    @pre_load
    def sanitize_inputs(self, data, **kwargs):
        if 'ticker' in data:
            ticker = ValidationService.sanitize_string(data['ticker'], 10).upper()
            if not ValidationService.validate_ticker(ticker):
                raise ValidationError('Invalid ticker format')
            data['ticker'] = ticker
        if 'message' in data:
            data['message'] = ValidationService.sanitize_string(data['message'], 1000)
        return data

# Validation decorator for Flask routes
def validate_json(schema_class):
    """
    Decorator to validate JSON input against a Marshmallow schema.
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            
            try:
                schema = schema_class()
                validated_data = schema.load(request.get_json() or {})
                return f(validated_data, *args, **kwargs)
            except ValidationError as e:
                logger.warning(f"Validation error in {f.__name__}: {e.messages}")
                return jsonify({
                    'error': 'Validation failed',
                    'details': e.messages
                }), 400
            except Exception as e:
                logger.error(f"Unexpected validation error in {f.__name__}: {str(e)}")
                return jsonify({
                    'error': 'Invalid request data'
                }), 400
        
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

# Global validation service instance
validation_service = ValidationService()

