# Modality-aware Transformer (MAT) Integration

This document describes the integration of the Modality-aware Transformer (MAT) model into the Trading Dashboard ML Engine for enhanced financial time series forecasting.

## Overview

The MAT model is a sophisticated deep learning architecture that combines textual (sentiment) and time-series (technical) data through novel attention mechanisms to improve financial forecasting accuracy. This implementation is based on the research paper "Modality-aware Transformer for Financial Time series Forecasting."

## Key Features

### 1. Multi-Modal Architecture
- **Text Modality**: Processes sentiment data, newsletter analysis, and market commentary
- **Time-Series Modality**: Handles technical indicators, price movements, and market data
- **Cross-Modal Attention**: Enables information exchange between modalities

### 2. Advanced Attention Mechanisms
- **Feature-Level Attention**: Focuses on important features within each modality
- **Intra-Modal Attention**: Self-attention within individual modalities
- **Inter-Modal Attention**: Cross-attention between different modalities
- **Target-Modal Attention**: Decoder attention to both modalities for prediction

### 3. Flexible Architecture
- Configurable model dimensions and layers
- Support for variable sequence lengths
- Scalable to different prediction horizons
- GPU acceleration support

## Model Architecture

```
Input Data:
├── Text Features (sentiment, confidence, key phrases, etc.)
└── Time-Series Features (RSI, MACD, volume, momentum, etc.)

Encoder:
├── Feature-Level Attention (per modality)
├── Embedding Layers
├── Positional Encoding
└── Multi-Layer Encoder
    ├── Intra-Modal Self-Attention
    ├── Inter-Modal Cross-Attention
    └── Feed-Forward Networks

Decoder:
├── Masked Self-Attention
├── Text-Target Attention
├── Time-Series-Target Attention
└── Final Prediction Layer

Output:
└── Multi-step Price Predictions
```

## Installation and Setup

### 1. Dependencies
The MAT model requires PyTorch and additional dependencies:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0
pip install -r requirements.txt
```

### 2. Configuration
Update `config/ml_config.json` to enable MAT:

```json
{
  "use_mat_model": true,
  "mat_model": {
    "d_model": 256,
    "n_heads": 8,
    "n_encoder_layers": 2,
    "n_decoder_layers": 1,
    "d_ff": 1024,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "max_seq_length": 50,
    "sequence_length": 10,
    "prediction_horizon": 3,
    "txt_features": 6,
    "ts_features": 6
  }
}
```

## Usage

### 1. API Endpoints

#### MAT Model Status
```http
GET /models/mat/status
```
Returns the current status of the MAT model including training state and device information.

#### Train MAT Model
```http
POST /models/mat/train
```
Initiates MAT model training in the background.

#### Model Information
```http
GET /models/info
```
Returns comprehensive information about all models including MAT status.

### 2. Direct Usage

```python
from src.models.mat_transformer import MATPredictor

# Initialize predictor
config = {
    'mat_model': {
        'd_model': 256,
        'n_heads': 8,
        'n_encoder_layers': 2,
        'n_decoder_layers': 1,
        'd_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 0.001
    }
}

predictor = MATPredictor(config)

# Train model
result = predictor.train(txt_data, ts_data, target_data, epochs=100)

# Make predictions
predictions = predictor.predict(txt_data, ts_data, decoder_input)
```

### 3. Data Format

#### Text Features (6 dimensions)
- `sentiment_score`: Newsletter sentiment (-1 to 1)
- `sentiment_confidence`: Confidence in sentiment (0 to 1)
- `sentiment_magnitude`: Absolute sentiment strength
- `bullish_terms_count`: Number of bullish terms
- `bearish_terms_count`: Number of bearish terms
- `amplifier_impact`: Impact of amplifying words

#### Time-Series Features (6 dimensions)
- `volume_ratio`: Trading volume ratio
- `price_momentum`: Price momentum indicator
- `volatility_rank`: Volatility percentile rank
- `rsi`: Relative Strength Index
- `macd_signal`: MACD signal line
- `bollinger_position`: Position within Bollinger Bands

#### Input Shapes
- Text Data: `(batch_size, sequence_length, txt_features)`
- Time-Series Data: `(batch_size, sequence_length, ts_features)`
- Target Data: `(batch_size, prediction_horizon, 1)`

## Testing

Run the comprehensive test suite:

```bash
cd ml-engine
python test_mat_model.py
```

This will test:
- Standalone model functionality
- Training and prediction pipeline
- Model saving and loading
- Different sequence lengths
- Performance metrics

## Performance Considerations

### 1. Hardware Requirements
- **CPU**: Multi-core processor recommended
- **Memory**: 8GB+ RAM for training
- **GPU**: CUDA-compatible GPU recommended for large datasets
- **Storage**: Sufficient space for model checkpoints

### 2. Training Parameters
- **Batch Size**: Adjust based on available memory
- **Learning Rate**: Start with 0.001, adjust based on convergence
- **Epochs**: 50-200 depending on dataset size
- **Sequence Length**: 10-50 time steps typically effective

### 3. Optimization Tips
- Use mixed precision training for faster training
- Implement gradient clipping to prevent exploding gradients
- Use learning rate scheduling for better convergence
- Monitor validation loss to prevent overfitting

## Integration with Trading Dashboard

### 1. Backend Integration
The MAT model is integrated into the existing ML engine and can be used alongside traditional models:

```python
# In trading_predictor.py
if self.use_mat and self._can_train_mat(training_data):
    mat_result = self._train_mat_model(training_data, features)
```

### 2. API Integration
MAT predictions can be accessed through the existing prediction endpoints with enhanced accuracy for multi-modal data.

### 3. Model Ensemble
The MAT model can be combined with traditional ML models for ensemble predictions:

```python
# Combine MAT predictions with traditional model predictions
final_prediction = (
    0.6 * mat_prediction + 
    0.4 * traditional_prediction
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce model dimensions
   - Use gradient checkpointing

2. **Training Not Converging**
   - Check learning rate
   - Verify data preprocessing
   - Ensure proper sequence alignment

3. **Poor Prediction Quality**
   - Increase training data
   - Tune hyperparameters
   - Check feature engineering

### Debug Mode
Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### 1. Planned Features
- Attention visualization for interpretability
- Multi-asset prediction support
- Real-time inference optimization
- Advanced ensemble methods

### 2. Research Directions
- Incorporation of additional modalities (news, social media)
- Adaptive attention mechanisms
- Meta-learning for quick adaptation
- Uncertainty quantification

## References

1. "Modality-aware Transformer for Financial Time series Forecasting" - Research Paper
2. "Attention Is All You Need" - Transformer Architecture
3. PyTorch Documentation - Deep Learning Framework
4. Financial Time Series Analysis - Domain Knowledge

## Support

For issues and questions:
1. Check the test suite output
2. Review configuration settings
3. Examine log files for detailed error messages
4. Consult the API documentation

---

**Note**: This implementation provides a solid foundation for multi-modal financial forecasting. Performance will improve with domain-specific data and proper hyperparameter tuning.
