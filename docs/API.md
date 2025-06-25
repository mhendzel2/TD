# Trading Dashboard API Documentation (Simplified for Personal Use)

## Overview

The Trading Dashboard API provides endpoints for newsletter analysis, machine learning predictions, and portfolio management. All endpoints require authentication unless otherwise specified.

## Authentication

### Base URL
```
http://localhost:3001/api
```

### Authentication Headers
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

### Authentication Endpoints

#### Register User
```http
POST /auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe"
  },
  "access_token": "jwt_token",
  "refresh_token": "refresh_token"
}
```

#### Login User
```http
POST /auth/login
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "message": "Login successful",
  "user": {
    "id": "uuid",
    "email": "user@example.com",
    "first_name": "John",
    "last_name": "Doe"
  },
  "access_token": "jwt_token",
  "refresh_token": "refresh_token"
}
```

#### Refresh Token
```http
POST /auth/refresh
```

**Headers:**
```
Authorization: Bearer <refresh_token>
```

**Response:**
```json
{
  "access_token": "new_jwt_token"
}
```

#### Logout
```http
POST /auth/logout
```

**Response:**
```json
{
  "message": "Logout successful"
}
```

## Newsletter Endpoints

### Get User Newsletters
```http
GET /newsletters/
```

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 20)
- `source` (optional): Filter by newsletter source

**Response:**
```json
{
  "newsletters": [
    {
      "id": "uuid",
      "title": "Weekly Market Update",
      "source": "MarketWatch",
      "content": "Newsletter content...",
      "sentiment_score": 0.65,
      "priority_score": 8.2,
      "tickers_mentioned": ["AAPL", "GOOGL", "TSLA"],
      "created_at": "2024-01-15T10:30:00Z",
      "analyzed_at": "2024-01-15T10:35:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "pages": 8
  }
}
```

### Analyze Newsletter
```http
POST /newsletters/analyze
```

**Request Body:**
```json
{
  "title": "Weekly Market Update",
  "content": "This week we saw strong bullish momentum in tech stocks...",
  "source": "MarketWatch",
  "content_type": "html"
}
```

**Response:**
```json
{
  "id": "uuid",
  "analysis": {
    "sentiment_score": 0.65,
    "confidence": 0.82,
    "priority_score": 8.2,
    "tickers_mentioned": [
      {
        "ticker": "AAPL",
        "sentiment": 0.75,
        "confidence": 0.88,
        "mentions": 3,
        "context": ["bullish momentum", "strong earnings"]
      }
    ],
    "key_phrases": ["bullish momentum", "tech stocks", "strong earnings"],
    "summary": "Positive sentiment towards tech stocks with focus on Apple and Google."
  },
  "processed_at": "2024-01-15T10:35:00Z"
}
```

### Get Newsletter by ID
```http
GET /newsletters/{id}
```

**Response:**
```json
{
  "id": "uuid",
  "title": "Weekly Market Update",
  "source": "MarketWatch",
  "content": "Newsletter content...",
  "sentiment_score": 0.65,
  "priority_score": 8.2,
  "tickers_mentioned": ["AAPL", "GOOGL", "TSLA"],
  "analysis": {
    "sentiment_breakdown": {
      "AAPL": {"sentiment": 0.75, "confidence": 0.88},
      "GOOGL": {"sentiment": 0.55, "confidence": 0.72}
    },
    "key_phrases": ["bullish momentum", "tech stocks"],
    "summary": "Positive sentiment analysis summary"
  },
  "created_at": "2024-01-15T10:30:00Z",
  "analyzed_at": "2024-01-15T10:35:00Z"
}
```

### Get Ticker Mentions
```http
GET /newsletters/tickers/{ticker}
```

**Query Parameters:**
- `days` (optional): Number of days to look back (default: 30)
- `min_sentiment` (optional): Minimum sentiment score filter

**Response:**
```json
{
  "ticker": "AAPL",
  "mentions": [
    {
      "newsletter_id": "uuid",
      "newsletter_title": "Weekly Market Update",
      "source": "MarketWatch",
      "sentiment": 0.75,
      "confidence": 0.88,
      "context": "Strong bullish momentum with earnings beat",
      "mentioned_at": "2024-01-15T10:30:00Z"
    }
  ],
  "summary": {
    "total_mentions": 15,
    "average_sentiment": 0.68,
    "sentiment_trend": "positive",
    "top_sources": ["MarketWatch", "Seeking Alpha"]
  }
}
```

## Machine Learning Endpoints

### Get Predictions
```http
POST /ml/predict
```

**Request Body:**
```json
{
  "ticker": "AAPL",
  "features": {
    "sentiment_score": 0.75,
    "volume_ratio": 1.2,
    "price_change": 0.03,
    "market_cap": 3000000000000,
    "sector": "Technology"
  },
  "prediction_horizon": "1d"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "predictions": {
    "probability": 0.72,
    "direction": "bullish",
    "magnitude": 0.025,
    "confidence_interval": {
      "lower": 0.65,
      "upper": 0.79
    },
    "risk_score": 0.35
  },
  "model_info": {
    "version": "1.0.0",
    "accuracy": 0.68,
    "last_trained": "2024-01-10T15:00:00Z"
  },
  "generated_at": "2024-01-15T10:45:00Z"
}
```

### Analyze Sentiment
```http
POST /ml/sentiment
```

**Request Body:**
```json
{
  "text": "Apple stock showing strong bullish momentum after earnings beat",
  "context": "newsletter_analysis"
}
```

**Response:**
```json
{
  "sentiment_score": 0.75,
  "confidence": 0.88,
  "classification": "bullish",
  "key_phrases": ["strong bullish momentum", "earnings beat"],
  "tickers_detected": ["AAPL"],
  "analysis_details": {
    "positive_indicators": ["strong", "bullish", "beat"],
    "negative_indicators": [],
    "neutral_indicators": ["stock", "after"]
  }
}
```

### Get Model Status
```http
GET /ml/models/status
```

**Response:**
```json
{
  "models": {
    "sentiment_analyzer": {
      "status": "ready",
      "version": "1.0.0",
      "accuracy": 0.85,
      "last_trained": "2024-01-10T15:00:00Z",
      "training_samples": 50000
    },
    "trading_predictor": {
      "status": "training",
      "version": "1.1.0",
      "progress": 0.75,
      "eta": "2024-01-15T12:00:00Z"
    }
  },
  "system_status": "healthy"
}
```

### Train Models
```http
POST /ml/models/train
```

**Request Body:**
```json
{
  "model_type": "trading_predictor",
  "retrain": true,
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "validation_split": 0.2
  }
}
```

**Response:**
```json
{
  "message": "Model training started",
  "job_id": "uuid",
  "estimated_completion": "2024-01-15T12:00:00Z",
  "status": "training"
}
```

## Portfolio Endpoints

### Get Positions
```http
GET /portfolio/positions
```

**Response:**
```json
{
  "positions": [
    {
      "id": "uuid",
      "ticker": "AAPL",
      "quantity": 100,
      "average_cost": 150.25,
      "current_price": 155.30,
      "market_value": 15530.00,
      "unrealized_pnl": 505.00,
      "unrealized_pnl_percent": 3.36,
      "position_type": "long",
      "opened_at": "2024-01-10T09:30:00Z",
      "last_updated": "2024-01-15T16:00:00Z"
    }
  ],
  "summary": {
    "total_value": 125000.00,
    "total_pnl": 2500.00,
    "total_pnl_percent": 2.04,
    "cash_balance": 25000.00,
    "buying_power": 50000.00
  }
}
```

### Add Position
```http
POST /portfolio/positions
```

**Request Body:**
```json
{
  "ticker": "AAPL",
  "quantity": 100,
  "price": 150.25,
  "position_type": "long",
  "order_type": "market"
}
```

**Response:**
```json
{
  "id": "uuid",
  "ticker": "AAPL",
  "quantity": 100,
  "average_cost": 150.25,
  "position_type": "long",
  "opened_at": "2024-01-15T10:45:00Z",
  "message": "Position added successfully"
}
```

### Update Position
```http
PUT /portfolio/positions/{id}
```

**Request Body:**
```json
{
  "quantity": 150,
  "price": 152.00,
  "action": "add"
}
```

**Response:**
```json
{
  "id": "uuid",
  "ticker": "AAPL",
  "quantity": 150,
  "average_cost": 151.00,
  "message": "Position updated successfully"
}
```

### Delete Position
```http
DELETE /portfolio/positions/{id}
```

**Response:**
```json
{
  "message": "Position deleted successfully"
}
```

### Get Portfolio Analytics
```http
GET /portfolio/analytics
```

**Query Parameters:**
- `period` (optional): Analysis period (1d, 1w, 1m, 3m, 1y)

**Response:**
```json
{
  "performance": {
    "total_return": 0.0825,
    "annualized_return": 0.15,
    "sharpe_ratio": 1.25,
    "max_drawdown": -0.08,
    "volatility": 0.18
  },
  "risk_metrics": {
    "var_95": -2500.00,
    "beta": 1.15,
    "correlation_spy": 0.85,
    "concentration_risk": 0.35
  },
  "sector_allocation": {
    "Technology": 0.45,
    "Healthcare": 0.25,
    "Finance": 0.20,
    "Energy": 0.10
  }
}
```

## WebSocket Events

### Connection
```javascript
const socket = io("http://localhost:3001", {
  auth: {
    token: "jwt_token"
  }
});
```

### Subscribe to Channels
```javascript
socket.emit("subscribe", {
  channels: ["predictions", "portfolio", "alerts", "market"]
});
```

### Event Types

#### Prediction Updates
```javascript
socket.on("prediction_update", (data) => {
  console.log("New prediction:", data);
  // data.type: "prediction:update"
  // data.data: prediction object
  // data.timestamp: ISO timestamp
});
```

#### Portfolio Updates
```javascript
socket.on("portfolio_update", (data) => {
  console.log("Portfolio changed:", data);
  // data.type: "portfolio:update"
  // data.data: position changes
  // data.timestamp: ISO timestamp
});
```

#### New Alerts
```javascript
socket.on("new_alert", (data) => {
  console.log("New alert:", data);
  // data.type: "alert:new"
  // data.data: alert details
  // data.timestamp: ISO timestamp
});
```

#### Market Updates
```javascript
socket.on("market_update", (data) => {
  console.log("Market data:", data);
  // data.type: "market:update"
  // data.data: market information
  // data.timestamp: ISO timestamp
});
```

#### Newsletter Updates
```javascript
socket.on("newsletter_update", (data) => {
  console.log("Newsletter analyzed:", data);
  // data.type: "newsletter:update"
  // data.data: analysis results
  // data.timestamp: ISO timestamp
});
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "Specific field error"
  },
  "timestamp": "2024-01-15T10:45:00Z"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| INVALID_TOKEN | 401 | JWT token is invalid or expired |
| INSUFFICIENT_PERMISSIONS | 403 | User lacks required permissions |
| RESOURCE_NOT_FOUND | 404 | Requested resource doesn't exist |
| VALIDATION_ERROR | 400 | Request data validation failed |
| INTERNAL_ERROR | 500 | Server internal error |
| SERVICE_UNAVAILABLE | 503 | External service unavailable |

## Testing

### Example cURL Commands

#### Login
```bash
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'
```

#### Analyze Newsletter
```bash
curl -X POST http://localhost:3001/api/newsletters/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"title":"Market Update","content":"AAPL showing bullish momentum","source":"Test"}'
```

#### Get Predictions
```bash
curl -X POST http://localhost:3001/api/ml/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"ticker":"AAPL","features":{"sentiment_score":0.75}}'
```

## Support

For API support:
- 📧 Email: api-support@trading-dashboard.com
- 📖 Documentation: [api.trading-dashboard.com](https://api.trading-dashboard.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/trading-dashboard/issues)

---

**API Version: 1.0.0**  
**Last Updated: January 15, 2024**


