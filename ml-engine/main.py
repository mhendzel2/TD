from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.sentiment_analyzer import NewsletterSentimentAnalyzer
from src.models.trading_predictor import TradingPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trading Dashboard ML Engine",
    description="Machine Learning API for trading predictions and sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
sentiment_analyzer = NewsletterSentimentAnalyzer()
trading_predictor = TradingPredictor()

# Pydantic models for request/response
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Newsletter text to analyze")
    ticker: str = Field(..., description="Stock ticker to analyze sentiment for")

class BatchSentimentRequest(BaseModel):
    text: str = Field(..., description="Newsletter text to analyze")
    tickers: List[str] = Field(..., description="List of stock tickers to analyze")

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker")
    newsletter_sentiment_score: Optional[float] = Field(0.0, description="Newsletter sentiment score")
    newsletter_confidence: Optional[float] = Field(0.0, description="Newsletter confidence score")
    rsi_14: Optional[float] = Field(50.0, description="14-day RSI")
    macd_signal: Optional[float] = Field(0.0, description="MACD signal")
    volume_ratio: Optional[float] = Field(1.0, description="Volume ratio")
    price_momentum_5d: Optional[float] = Field(0.0, description="5-day price momentum")
    price_momentum_20d: Optional[float] = Field(0.0, description="20-day price momentum")
    vix_level: Optional[float] = Field(20.0, description="VIX level")
    market_sentiment: Optional[float] = Field(0.0, description="Overall market sentiment")
    sector_performance: Optional[float] = Field(0.0, description="Sector performance")
    position_size_ratio: Optional[float] = Field(0.05, description="Position size ratio")
    portfolio_correlation: Optional[float] = Field(0.5, description="Portfolio correlation")

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")

class TrainingRequest(BaseModel):
    retrain: bool = Field(True, description="Whether to retrain the model")
    data_source: Optional[str] = Field("database", description="Data source for training")

class SentimentResponse(BaseModel):
    sentiment_score: float
    confidence: float
    key_phrases: List[str]
    context_found: bool
    context_length: Optional[int] = None

class PredictionResponse(BaseModel):
    probability_score: float
    direction_probability: float
    confidence_lower: float
    confidence_upper: float
    contributing_factors: Dict[str, Any]
    model_version: str
    prediction_timestamp: str
    expected_return: Optional[float] = None

class ModelStatusResponse(BaseModel):
    model_loaded: bool
    model_version: str
    last_trained: Optional[str] = None
    performance_metrics: Dict[str, float]
    feature_count: int

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ml-engine",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# Sentiment analysis endpoints
@app.post("/sentiment/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment for a specific ticker in newsletter text."""
    try:
        result = sentiment_analyzer.analyze_sentiment(request.text, request.ticker)
        
        return SentimentResponse(
            sentiment_score=result['sentiment_score'],
            confidence=result['confidence'],
            key_phrases=result['key_phrases'],
            context_found=result['context_found'],
            context_length=result.get('context_length')
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/batch", response_model=Dict[str, SentimentResponse])
async def batch_analyze_sentiment(request: BatchSentimentRequest):
    """Analyze sentiment for multiple tickers in the same text."""
    try:
        results = sentiment_analyzer.batch_analyze(request.text, request.tickers)
        
        response = {}
        for ticker, result in results.items():
            response[ticker] = SentimentResponse(
                sentiment_score=result['sentiment_score'],
                confidence=result['confidence'],
                key_phrases=result['key_phrases'],
                context_found=result['context_found'],
                context_length=result.get('context_length')
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_trade(request: PredictionRequest):
    """Make a trading prediction for a single ticker."""
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Make prediction
        result = trading_predictor.predict(input_data)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
        
        prediction = result['predictions'][0]
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def batch_predict_trades(request: BatchPredictionRequest):
    """Make trading predictions for multiple tickers."""
    try:
        # Convert requests to DataFrame
        input_data = pd.DataFrame([pred.dict() for pred in request.predictions])
        
        # Make predictions
        result = trading_predictor.predict(input_data)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Batch prediction failed'))
        
        predictions = [PredictionResponse(**pred) for pred in result['predictions']]
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get the current status of the ML models."""
    try:
        model_loaded = 'probability' in trading_predictor.models
        
        return ModelStatusResponse(
            model_loaded=model_loaded,
            model_version=trading_predictor.model_version,
            performance_metrics=trading_predictor.performance_metrics,
            feature_count=len(trading_predictor.feature_importance.get('probability', {}))
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain the ML models."""
    try:
        if request.retrain:
            # In a real implementation, this would load data from the database
            # For now, we'll create mock training data
            background_tasks.add_task(train_models_background)
            
            return {
                "message": "Model training started in background",
                "status": "training",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "message": "Training not requested",
                "status": "idle"
            }
            
    except Exception as e:
        logger.error(f"Error starting model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/feature-importance")
async def get_feature_importance():
    """Get feature importance from the trained models."""
    try:
        if not trading_predictor.feature_importance:
            raise HTTPException(status_code=404, detail="No trained model found")
        
        return {
            "feature_importance": trading_predictor.feature_importance,
            "model_version": trading_predictor.model_version
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/performance")
async def get_model_performance():
    """Get performance metrics of the trained models."""
    try:
        if not trading_predictor.performance_metrics:
            raise HTTPException(status_code=404, detail="No performance metrics available")
        
        return {
            "performance_metrics": trading_predictor.performance_metrics,
            "model_version": trading_predictor.model_version
        }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for model training
async def train_models_background():
    """Background task to train models with mock data."""
    try:
        logger.info("Starting background model training...")
        
        # Create mock training data
        np.random.seed(42)
        n_samples = 1000
        
        training_data = pd.DataFrame({
            'ticker': np.random.choice(['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'], n_samples),
            'newsletter_sentiment_score': np.random.normal(0, 0.3, n_samples),
            'newsletter_confidence': np.random.uniform(0.5, 1.0, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'macd_signal': np.random.normal(0, 0.1, n_samples),
            'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
            'price_momentum_5d': np.random.normal(0, 0.02, n_samples),
            'price_momentum_20d': np.random.normal(0, 0.05, n_samples),
            'vix_level': np.random.uniform(12, 35, n_samples),
            'market_sentiment': np.random.normal(0, 0.2, n_samples),
            'sector_performance': np.random.normal(0, 0.03, n_samples),
            'position_size_ratio': np.random.uniform(0.01, 0.15, n_samples),
            'portfolio_correlation': np.random.uniform(0.2, 0.8, n_samples),
            'price_change_5d': np.random.normal(0, 0.05, n_samples)
        })
        
        # Create target variable (trade success)
        # Higher sentiment + good technicals = higher success probability
        success_prob = (
            0.5 + 
            0.2 * training_data['newsletter_sentiment_score'] +
            0.1 * (training_data['rsi_14'] - 50) / 50 +
            0.1 * training_data['price_momentum_5d'] +
            0.1 * np.random.normal(0, 0.1, n_samples)
        )
        success_prob = np.clip(success_prob, 0, 1)
        training_data['trade_success'] = np.random.binomial(1, success_prob)
        
        # Train the models
        result = trading_predictor.train_models(training_data)
        
        if result['success']:
            logger.info("Background model training completed successfully")
            
            # Save models
            model_path = "/tmp/trading_models"
            trading_predictor.save_models(model_path)
            
        else:
            logger.error(f"Background model training failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error in background model training: {str(e)}")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    try:
        model_path = "/tmp/trading_models"
        if os.path.exists(model_path):
            success = trading_predictor.load_models(model_path)
            if success:
                logger.info("Models loaded successfully on startup")
            else:
                logger.warning("Failed to load models on startup")
        else:
            logger.info("No saved models found. Training with mock data...")
            await train_models_background()
            
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

