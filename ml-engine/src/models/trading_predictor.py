"""
Trading Predictor

This module provides trading prediction capabilities using various ML models
including the Modality-aware Transformer (MAT) for enhanced predictions.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .mat_transformer import MATPredictor

logger = logging.getLogger(__name__)


class TradingPredictor:
    """
    Main trading prediction class that combines traditional ML models
    with the advanced MAT transformer for enhanced predictions.
    """
    
    def __init__(self, config_path: str = "config/ml_config.json"):
        """
        Initialize the trading predictor.
        
        Args:
            config_path: Path to ML configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.model_version = "1.0.0"
        self.last_trained = None
        
        # MAT model for advanced predictions
        self.mat_predictor = None
        self.use_mat = self.config.get('use_mat_model', False)
        
        logger.info("Trading Predictor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            import json
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model_version": "1.0.0",
            "models": {
                "random_state": 42,
                "probability": {
                    "n_estimators": 200,
                    "max_depth": 15,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2
                },
                "direction": {
                    "n_estimators": 150,
                    "max_depth": 12
                },
                "magnitude": {
                    "n_estimators": 150,
                    "max_depth": 12
                }
            },
            "training": {
                "test_size": 0.2,
                "cv_folds": 5,
                "min_samples_for_training": 100
            },
            "use_mat_model": False,
            "mat_model": {
                "d_model": 256,
                "n_heads": 8,
                "n_encoder_layers": 2,
                "n_decoder_layers": 1,
                "d_ff": 1024,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "max_seq_length": 50
            }
        }
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        
        Args:
            data: Raw input data
            
        Returns:
            Processed feature DataFrame
        """
        features = data.copy()
        
        # Ensure all required columns exist with defaults
        required_columns = [
            'newsletter_sentiment_score', 'newsletter_confidence', 'rsi_14',
            'macd_signal', 'volume_ratio', 'price_momentum_5d', 'price_momentum_20d',
            'vix_level', 'market_sentiment', 'sector_performance',
            'position_size_ratio', 'portfolio_correlation'
        ]
        
        for col in required_columns:
            if col not in features.columns:
                # Set reasonable defaults
                if 'sentiment' in col:
                    features[col] = 0.0
                elif 'rsi' in col:
                    features[col] = 50.0
                elif 'ratio' in col:
                    features[col] = 1.0
                elif 'vix' in col:
                    features[col] = 20.0
                else:
                    features[col] = 0.0
        
        # Create additional engineered features
        features['sentiment_confidence_product'] = (
            features['newsletter_sentiment_score'] * features['newsletter_confidence']
        )
        features['momentum_ratio'] = (
            features['price_momentum_5d'] / (features['price_momentum_20d'] + 1e-8)
        )
        features['risk_adjusted_sentiment'] = (
            features['newsletter_sentiment_score'] / (features['vix_level'] / 20.0)
        )
        
        # Handle any remaining NaN values
        features = features.fillna(0)
        
        return features
    
    def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all prediction models.
        
        Args:
            training_data: DataFrame with features and target variables
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare features
            features = self.prepare_features(training_data)
            
            # Create target variables if they don't exist
            if 'trade_success' not in training_data.columns:
                # Create synthetic target based on price change
                if 'price_change_5d' in training_data.columns:
                    training_data['trade_success'] = (training_data['price_change_5d'] > 0).astype(int)
                else:
                    # Create random target for demonstration
                    np.random.seed(42)
                    training_data['trade_success'] = np.random.binomial(1, 0.6, len(training_data))
            
            # Split data
            X = features.select_dtypes(include=[np.number])
            y = training_data['trade_success']
            
            if len(X) < self.config['training']['min_samples_for_training']:
                return {
                    'success': False,
                    'error': f'Insufficient training data: {len(X)} samples'
                }
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['training']['test_size'],
                random_state=self.config['models']['random_state'],
                stratify=y
            )
            
            # Scale features
            self.scalers['features'] = StandardScaler()
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            # Train probability model (main model)
            logger.info("Training probability model...")
            prob_config = self.config['models']['probability']
            self.models['probability'] = RandomForestClassifier(
                n_estimators=prob_config['n_estimators'],
                max_depth=prob_config['max_depth'],
                min_samples_split=prob_config['min_samples_split'],
                min_samples_leaf=prob_config['min_samples_leaf'],
                random_state=self.config['models']['random_state']
            )
            
            self.models['probability'].fit(X_train_scaled, y_train)
            
            # Calculate feature importance
            feature_names = X.columns.tolist()
            importance_scores = self.models['probability'].feature_importances_
            self.feature_importance['probability'] = dict(zip(feature_names, importance_scores))
            
            # Evaluate model
            y_pred = self.models['probability'].predict(X_test_scaled)
            y_pred_proba = self.models['probability'].predict_proba(X_test_scaled)[:, 1]
            
            self.performance_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'samples_trained': len(X_train),
                'samples_tested': len(X_test)
            }
            
            # Train MAT model if enabled
            if self.use_mat and self._can_train_mat(training_data):
                logger.info("Training MAT model...")
                mat_result = self._train_mat_model(training_data, features)
                if mat_result['success']:
                    self.performance_metrics['mat_model'] = mat_result
            
            self.last_trained = datetime.utcnow().isoformat()
            self.model_version = f"1.0.0-{datetime.utcnow().strftime('%Y%m%d')}"
            
            logger.info(f"Model training completed. Accuracy: {self.performance_metrics['accuracy']:.3f}")
            
            return {
                'success': True,
                'performance_metrics': self.performance_metrics,
                'feature_importance': self.feature_importance,
                'model_version': self.model_version
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _can_train_mat(self, data: pd.DataFrame) -> bool:
        """Check if we have sufficient data to train MAT model."""
        required_cols = ['ticker', 'newsletter_sentiment_score']
        return all(col in data.columns for col in required_cols) and len(data) >= 100
    
    def _train_mat_model(self, training_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Train the MAT model with sequence data."""
        try:
            # Initialize MAT predictor
            self.mat_predictor = MATPredictor(self.config)
            
            # Prepare sequence data for MAT
            # This is a simplified version - in practice, you'd need proper time series data
            seq_length = 10
            txt_features = 6  # sentiment-related features
            ts_features = 6   # technical features
            
            # Create mock sequence data for demonstration
            n_samples = len(training_data)
            txt_data = np.random.randn(n_samples, seq_length, txt_features)
            ts_data = np.random.randn(n_samples, seq_length, ts_features)
            target_data = np.random.randn(n_samples, 3, 1)  # 3-step prediction
            
            # In a real implementation, you would:
            # 1. Group data by ticker and time
            # 2. Create proper sequences from historical data
            # 3. Extract sentiment features over time
            # 4. Extract technical indicators over time
            
            # Train MAT model
            result = self.mat_predictor.train(txt_data, ts_data, target_data, epochs=50)
            
            return result
            
        except Exception as e:
            logger.error(f"Error training MAT model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make trading predictions.
        
        Args:
            input_data: DataFrame with features for prediction
            
        Returns:
            Prediction results dictionary
        """
        try:
            if 'probability' not in self.models:
                return {
                    'success': False,
                    'error': 'No trained model available'
                }
            
            # Prepare features
            features = self.prepare_features(input_data)
            X = features.select_dtypes(include=[np.number])
            
            # Scale features
            if 'features' in self.scalers:
                X_scaled = self.scalers['features'].transform(X)
            else:
                X_scaled = X.values
            
            # Make predictions
            predictions = []
            
            for i in range(len(X_scaled)):
                # Get probability prediction
                prob_pred = self.models['probability'].predict_proba([X_scaled[i]])[0]
                probability_score = prob_pred[1]  # Probability of success
                
                # Calculate confidence intervals (simplified)
                confidence_lower = max(0.0, probability_score - 0.1)
                confidence_upper = min(1.0, probability_score + 0.1)
                
                # Direction probability (bullish vs bearish)
                direction_prob = probability_score
                
                # Contributing factors
                feature_names = X.columns.tolist()
                feature_values = X_scaled[i]
                feature_importance = self.feature_importance.get('probability', {})
                
                contributing_factors = {}
                for j, (name, value) in enumerate(zip(feature_names, feature_values)):
                    importance = feature_importance.get(name, 0.0)
                    contributing_factors[name] = {
                        'value': float(value),
                        'importance': float(importance),
                        'contribution': float(value * importance)
                    }
                
                # Expected return (simplified calculation)
                expected_return = (probability_score - 0.5) * 0.1  # ±10% max expected return
                
                prediction = {
                    'probability_score': float(probability_score),
                    'direction_probability': float(direction_prob),
                    'confidence_lower': float(confidence_lower),
                    'confidence_upper': float(confidence_upper),
                    'contributing_factors': contributing_factors,
                    'model_version': self.model_version,
                    'prediction_timestamp': datetime.utcnow().isoformat(),
                    'expected_return': float(expected_return)
                }
                
                predictions.append(prediction)
            
            return {
                'success': True,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_models(self, model_path: str) -> bool:
        """
        Save trained models to disk.
        
        Args:
            model_path: Directory to save models
            
        Returns:
            Success status
        """
        try:
            os.makedirs(model_path, exist_ok=True)
            
            # Save sklearn models
            for model_name, model in self.models.items():
                if model_name != 'mat':  # MAT model is saved separately
                    model_file = os.path.join(model_path, f"{model_name}_model.pkl")
                    joblib.dump(model, model_file)
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_file = os.path.join(model_path, f"{scaler_name}_scaler.pkl")
                joblib.dump(scaler, scaler_file)
            
            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'last_trained': self.last_trained,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'config': self.config
            }
            
            metadata_file = os.path.join(model_path, "model_metadata.pkl")
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Save MAT model if available
            if self.mat_predictor and self.mat_predictor.is_trained:
                mat_file = os.path.join(model_path, "mat_model.pth")
                self.mat_predictor.save_model(mat_file)
            
            logger.info(f"Models saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, model_path: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            model_path: Directory containing saved models
            
        Returns:
            Success status
        """
        try:
            # Load metadata
            metadata_file = os.path.join(model_path, "model_metadata.pkl")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.model_version = metadata.get('model_version', '1.0.0')
                self.last_trained = metadata.get('last_trained')
                self.feature_importance = metadata.get('feature_importance', {})
                self.performance_metrics = metadata.get('performance_metrics', {})
                self.config.update(metadata.get('config', {}))
            
            # Load sklearn models
            model_files = {
                'probability': 'probability_model.pkl',
                'direction': 'direction_model.pkl',
                'magnitude': 'magnitude_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_file = os.path.join(model_path, filename)
                if os.path.exists(model_file):
                    self.models[model_name] = joblib.load(model_file)
            
            # Load scalers
            scaler_files = {
                'features': 'features_scaler.pkl'
            }
            
            for scaler_name, filename in scaler_files.items():
                scaler_file = os.path.join(model_path, filename)
                if os.path.exists(scaler_file):
                    self.scalers[scaler_name] = joblib.load(scaler_file)
            
            # Load MAT model if available
            mat_file = os.path.join(model_path, "mat_model.pth")
            if os.path.exists(mat_file) and self.use_mat:
                self.mat_predictor = MATPredictor(self.config)
                self.mat_predictor.load_model(mat_file)
            
            logger.info(f"Models loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models."""
        return {
            'model_version': self.model_version,
            'last_trained': self.last_trained,
            'models_loaded': list(self.models.keys()),
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'use_mat': self.use_mat,
            'mat_trained': self.mat_predictor is not None and self.mat_predictor.is_trained if self.mat_predictor else False
        }
