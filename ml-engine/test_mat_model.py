"""
Test script for the Modality-aware Transformer (MAT) model.

This script demonstrates how to use the MAT model for financial time series forecasting
with both sentiment and technical data.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.mat_transformer import ModalityAwareTransformer, MATPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples=100, seq_length=10, pred_horizon=3):
    """
    Create sample financial data for testing the MAT model.
    
    Args:
        n_samples: Number of samples to generate
        seq_length: Length of input sequences
        pred_horizon: Number of future steps to predict
    
    Returns:
        Tuple of (txt_data, ts_data, target_data, metadata)
    """
    logger.info(f"Creating sample data: {n_samples} samples, {seq_length} sequence length")
    
    # Text/sentiment features (6 features)
    txt_features = 6
    txt_data = np.zeros((n_samples, seq_length, txt_features))
    
    # Time series/technical features (6 features)
    ts_features = 6
    ts_data = np.zeros((n_samples, seq_length, ts_features))
    
    # Target data (price changes)
    target_data = np.zeros((n_samples, pred_horizon, 1))
    
    # Generate realistic financial data
    tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
    metadata = []
    
    for i in range(n_samples):
        ticker = np.random.choice(tickers)
        
        # Generate sentiment features over time
        base_sentiment = np.random.normal(0, 0.3)  # Base sentiment for this sample
        sentiment_trend = np.random.normal(0, 0.1, seq_length)  # Sentiment evolution
        
        for t in range(seq_length):
            # Sentiment features
            txt_data[i, t, 0] = base_sentiment + sentiment_trend[t]  # Newsletter sentiment
            txt_data[i, t, 1] = np.random.uniform(0.5, 1.0)  # Confidence
            txt_data[i, t, 2] = abs(txt_data[i, t, 0])  # Sentiment magnitude
            txt_data[i, t, 3] = max(0, txt_data[i, t, 0]) * 10  # Bullish terms count
            txt_data[i, t, 4] = max(0, -txt_data[i, t, 0]) * 10  # Bearish terms count
            txt_data[i, t, 5] = np.random.uniform(0.8, 1.2)  # Amplifier impact
        
        # Generate technical features over time
        base_price = 100 + np.random.normal(0, 20)  # Starting price
        prices = [base_price]
        
        for t in range(seq_length):
            # Simulate price movement influenced by sentiment
            price_change = (
                0.01 * txt_data[i, t, 0] +  # Sentiment influence
                np.random.normal(0, 0.02)    # Random noise
            )
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
            
            # Technical indicators
            ts_data[i, t, 0] = np.random.uniform(0.5, 2.0)  # Volume ratio
            ts_data[i, t, 1] = price_change  # Price momentum
            ts_data[i, t, 2] = np.random.uniform(0, 1)  # Volatility rank
            ts_data[i, t, 3] = np.random.uniform(20, 80)  # RSI
            ts_data[i, t, 4] = np.random.normal(0, 0.1)  # MACD signal
            ts_data[i, t, 5] = np.random.uniform(-1, 1)  # Bollinger position
        
        # Generate target (future price changes)
        for h in range(pred_horizon):
            # Future price change influenced by recent sentiment and momentum
            recent_sentiment = np.mean(txt_data[i, -3:, 0])  # Recent sentiment
            recent_momentum = np.mean(ts_data[i, -3:, 1])  # Recent momentum
            
            future_change = (
                0.05 * recent_sentiment +
                0.03 * recent_momentum +
                np.random.normal(0, 0.02)
            )
            target_data[i, h, 0] = future_change
        
        metadata.append({
            'ticker': ticker,
            'sample_id': i,
            'base_sentiment': base_sentiment,
            'final_price': prices[-1]
        })
    
    return txt_data, ts_data, target_data, metadata


def test_mat_model():
    """Test the MAT model with sample data."""
    logger.info("Starting MAT model test...")
    
    # Model configuration
    config = {
        'mat_model': {
            'd_model': 128,  # Smaller for testing
            'n_heads': 4,
            'n_encoder_layers': 2,
            'n_decoder_layers': 1,
            'd_ff': 512,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'max_seq_length': 50
        }
    }
    
    # Create sample data
    txt_data, ts_data, target_data, metadata = create_sample_data(
        n_samples=200, seq_length=10, pred_horizon=3
    )
    
    logger.info(f"Data shapes - Text: {txt_data.shape}, TS: {ts_data.shape}, Target: {target_data.shape}")
    
    # Initialize MAT predictor
    mat_predictor = MATPredictor(config)
    
    # Split data for training and testing
    split_idx = int(0.8 * len(txt_data))
    
    train_txt = txt_data[:split_idx]
    train_ts = ts_data[:split_idx]
    train_target = target_data[:split_idx]
    
    test_txt = txt_data[split_idx:]
    test_ts = ts_data[split_idx:]
    test_target = target_data[split_idx:]
    
    logger.info(f"Training samples: {len(train_txt)}, Test samples: {len(test_txt)}")
    
    # Train the model
    logger.info("Training MAT model...")
    training_result = mat_predictor.train(
        train_txt, train_ts, train_target, epochs=20
    )
    
    if training_result['success']:
        logger.info(f"Training completed! Final loss: {training_result['final_loss']:.6f}")
        
        # Make predictions on test data
        logger.info("Making predictions on test data...")
        
        # Use the first part of target as decoder input (teacher forcing simulation)
        decoder_input = test_target[:, :1, :]  # Use first step as input
        
        predictions = mat_predictor.predict(test_txt, test_ts, decoder_input)
        
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Calculate simple metrics
        mse = np.mean((predictions[:, 0, :] - test_target[:, 0, :]) ** 2)
        mae = np.mean(np.abs(predictions[:, 0, :] - test_target[:, 0, :]))
        
        logger.info(f"Test MSE: {mse:.6f}")
        logger.info(f"Test MAE: {mae:.6f}")
        
        # Show some example predictions
        logger.info("\nExample predictions vs actual:")
        for i in range(min(5, len(predictions))):
            pred_val = predictions[i, 0, 0]
            actual_val = test_target[i, 0, 0]
            ticker = metadata[split_idx + i]['ticker']
            logger.info(f"{ticker}: Predicted={pred_val:.4f}, Actual={actual_val:.4f}")
        
        # Test model saving and loading
        logger.info("Testing model save/load...")
        model_path = "test_mat_model.pth"
        
        try:
            mat_predictor.save_model(model_path)
            logger.info("Model saved successfully")
            
            # Create new predictor and load model
            new_predictor = MATPredictor(config)
            new_predictor.load_model(model_path)
            logger.info("Model loaded successfully")
            
            # Test prediction with loaded model
            new_predictions = new_predictor.predict(test_txt[:1], test_ts[:1], decoder_input[:1])
            logger.info(f"Loaded model prediction: {new_predictions[0, 0, 0]:.4f}")
            
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info("Test model file cleaned up")
                
        except Exception as e:
            logger.error(f"Error in save/load test: {str(e)}")
        
    else:
        logger.error(f"Training failed: {training_result.get('error', 'Unknown error')}")


def test_standalone_model():
    """Test the standalone MAT model (without predictor wrapper)."""
    logger.info("Testing standalone MAT model...")
    
    # Model parameters
    txt_features = 6
    ts_features = 6
    d_model = 128
    n_heads = 4
    n_encoder_layers = 2
    n_decoder_layers = 1
    d_ff = 512
    dropout = 0.1
    
    # Create the model
    model = ModalityAwareTransformer(
        txt_features, ts_features, d_model, n_heads,
        n_encoder_layers, n_decoder_layers, d_ff, dropout
    )
    
    # Create sample input
    batch_size = 8
    seq_len_enc = 10
    seq_len_dec = 3
    
    x_txt = np.random.randn(batch_size, seq_len_enc, txt_features).astype(np.float32)
    x_ts = np.random.randn(batch_size, seq_len_enc, ts_features).astype(np.float32)
    x_dec = np.random.randn(batch_size, seq_len_dec, 1).astype(np.float32)
    
    # Convert to tensors
    import torch
    x_txt_tensor = torch.FloatTensor(x_txt)
    x_ts_tensor = torch.FloatTensor(x_ts)
    x_dec_tensor = torch.FloatTensor(x_dec)
    
    logger.info(f"Input shapes - Text: {x_txt_tensor.shape}, TS: {x_ts_tensor.shape}, Dec: {x_dec_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x_txt_tensor, x_ts_tensor, x_dec_tensor)
    
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different sequence lengths
    logger.info("Testing with different sequence lengths...")
    
    for seq_len in [5, 15, 20]:
        x_txt_test = torch.randn(4, seq_len, txt_features)
        x_ts_test = torch.randn(4, seq_len, ts_features)
        x_dec_test = torch.randn(4, 3, 1)
        
        with torch.no_grad():
            output_test = model(x_txt_test, x_ts_test, x_dec_test)
        
        logger.info(f"Seq len {seq_len}: Output shape {output_test.shape}")


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("MAT Model Testing Suite")
    logger.info("=" * 60)
    
    try:
        # Test standalone model first
        test_standalone_model()
        
        logger.info("\n" + "=" * 60)
        
        # Test full predictor with training
        test_mat_model()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
