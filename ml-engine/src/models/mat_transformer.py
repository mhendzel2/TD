"""
Modality-aware Transformer (MAT) for Financial Time Series Forecasting

This implementation is based on the research paper:
"Modality-aware Transformer for Financial Time series Forecasting"

The MAT model combines textual and time-series data through novel attention mechanisms
to improve financial forecasting accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class FeatureLevelAttention(nn.Module):
    """
    Feature-level Attention Layer as described in the paper.
    This layer computes attention weights over the features of a modality.
    """
    def __init__(self, in_features: int):
        super(FeatureLevelAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_features)
        
        Returns:
            torch.Tensor: Attention-weighted features
        """
        # Permute to (batch_size, num_features, sequence_length) for Conv1d
        x_permuted = x.permute(0, 2, 1)
        attention_weights = F.softmax(self.conv1(x_permuted), dim=2)
        # Permute back to (batch_size, sequence_length, num_features)
        attention_weights = attention_weights.permute(0, 2, 1)
        return attention_weights * x


class IntraModalMultiHeadAttention(nn.Module):
    """
    Intra-modal Multi-Head Attention Layer.
    Performs multi-head self-attention within a single modality.
    """
    def __init__(self, d_model: int, n_heads: int):
        super(IntraModalMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q, k, v: Query, Key, Value tensors
            mask: Optional attention mask
        
        Returns:
            torch.Tensor: Attention output
        """
        batch_size = q.size(0)

        q = self.query(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(context)


class InterModalMultiHeadAttention(nn.Module):
    """
    Inter-modal Multi-Head Attention Layer.
    Performs multi-head attention between different modalities.
    """
    def __init__(self, d_model: int, n_heads: int):
        super(InterModalMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q_cross: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q_cross: Query tensor from cross-modality
            k, v: Key and Value tensors from source modality
            mask: Optional attention mask
        
        Returns:
            torch.Tensor: Cross-modal attention output
        """
        batch_size = q_cross.size(0)

        q = self.query(q_cross).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(context)


class EncoderLayer(nn.Module):
    """
    A single layer of the Modality-aware Encoder.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.intra_modal_attention = IntraModalMultiHeadAttention(d_model, n_heads)
        self.inter_modal_attention = InterModalMultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_other_modality: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from current modality
            x_other_modality: Input from other modality for cross-attention
        
        Returns:
            torch.Tensor: Encoded representation
        """
        # Intra-modal attention
        attn_output = self.intra_modal_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Inter-modal attention
        attn_output = self.inter_modal_attention(x_other_modality, x, x)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    A single layer of the Modality-aware Decoder.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = IntraModalMultiHeadAttention(d_model, n_heads)
        self.txt_target_attention = InterModalMultiHeadAttention(d_model, n_heads)
        self.ts_target_attention = InterModalMultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_txt: torch.Tensor, enc_ts: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Decoder input
            enc_txt: Encoded text features
            enc_ts: Encoded time-series features
            tgt_mask: Target mask for causal attention
        
        Returns:
            torch.Tensor: Decoded representation
        """
        # Masked multi-head attention (self-attention)
        attn_output = self.masked_multi_head_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Text-target attention
        attn_txt_output = self.txt_target_attention(x, enc_txt, enc_txt)
        x_txt = self.norm2(x + self.dropout(attn_txt_output))

        # Time-series-target attention
        attn_ts_output = self.ts_target_attention(x, enc_ts, enc_ts)
        x_ts = self.norm3(x + self.dropout(attn_ts_output))

        # Combine outputs from the two modalities
        x = x_txt + x_ts

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm4(x + self.dropout(ff_output))
        
        return x


class ModalityAwareTransformer(nn.Module):
    """
    The main Modality-aware Transformer (MAT) model for financial forecasting.
    
    This model processes both textual (sentiment) and time-series (technical) data
    to make more accurate financial predictions.
    """
    
    def __init__(self, 
                 txt_features: int, 
                 ts_features: int, 
                 d_model: int = 512, 
                 n_heads: int = 8, 
                 n_encoder_layers: int = 2, 
                 n_decoder_layers: int = 1, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        """
        Args:
            txt_features: Number of text/sentiment features
            ts_features: Number of time-series features
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(ModalityAwareTransformer, self).__init__()
        
        self.d_model = d_model
        self.txt_features = txt_features
        self.ts_features = ts_features
        
        # Feature-level attention layers
        self.feature_attention_txt = FeatureLevelAttention(txt_features)
        self.feature_attention_ts = FeatureLevelAttention(ts_features)

        # Encoder layers for each modality
        self.encoder_layers_txt = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_encoder_layers)
        ])
        self.encoder_layers_ts = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_decoder_layers)
        ])
        
        # Embedding layers
        self.txt_embedding = nn.Linear(txt_features, d_model)
        self.ts_embedding = nn.Linear(ts_features, d_model)
        self.target_embedding = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        # Output layers
        self.final_linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_txt: torch.Tensor, x_ts: torch.Tensor, 
                x_dec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MAT model.
        
        Args:
            x_txt: Text/sentiment features (batch_size, seq_len, txt_features)
            x_ts: Time-series features (batch_size, seq_len, ts_features)
            x_dec: Decoder input (batch_size, pred_len, 1)
        
        Returns:
            torch.Tensor: Predictions (batch_size, pred_len, 1)
        """
        batch_size, seq_len = x_txt.size(0), x_txt.size(1)
        device = x_txt.device
        
        # Apply feature-level attention
        x_txt_att = self.feature_attention_txt(x_txt)
        x_ts_att = self.feature_attention_ts(x_ts)
        
        # Embed inputs
        enc_txt_input = self.txt_embedding(x_txt_att)
        enc_ts_input = self.ts_embedding(x_ts_att)
        dec_input = self.target_embedding(x_dec)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:, :seq_len, :].to(device)
        enc_txt_input = enc_txt_input + pos_enc
        enc_ts_input = enc_ts_input + pos_enc
        
        dec_pos_enc = self.pos_encoding[:, :x_dec.size(1), :].to(device)
        dec_input = dec_input + dec_pos_enc
        
        # Apply dropout
        enc_txt_input = self.dropout(enc_txt_input)
        enc_ts_input = self.dropout(enc_ts_input)
        dec_input = self.dropout(dec_input)
        
        # Encoder
        enc_txt = enc_txt_input
        enc_ts = enc_ts_input
        for i in range(len(self.encoder_layers_txt)):
            enc_txt = self.encoder_layers_txt[i](enc_txt, enc_ts)
            enc_ts = self.encoder_layers_ts[i](enc_ts, enc_txt)

        # Decoder
        tgt_len = dec_input.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
        
        dec_output = dec_input
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_txt, enc_ts, tgt_mask)
            
        # Final output
        output = self.final_linear(dec_output)
        
        return output

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_attention_weights(self, x_txt: torch.Tensor, x_ts: torch.Tensor, 
                            x_dec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for interpretability.
        
        Returns:
            Dict containing attention weights from different layers
        """
        # This would require modifying the forward pass to return attention weights
        # Implementation depends on specific interpretability requirements
        pass


class MATPredictor:
    """
    Wrapper class for the MAT model to integrate with the trading dashboard.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MAT predictor.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler_txt = None
        self.scaler_ts = None
        self.scaler_target = None
        self.is_trained = False
        
        logger.info(f"MAT Predictor initialized on device: {self.device}")

    def build_model(self, txt_features: int, ts_features: int) -> None:
        """Build the MAT model with specified feature dimensions."""
        model_config = self.config.get('mat_model', {})
        
        self.model = ModalityAwareTransformer(
            txt_features=txt_features,
            ts_features=ts_features,
            d_model=model_config.get('d_model', 512),
            n_heads=model_config.get('n_heads', 8),
            n_encoder_layers=model_config.get('n_encoder_layers', 2),
            n_decoder_layers=model_config.get('n_decoder_layers', 1),
            d_ff=model_config.get('d_ff', 2048),
            dropout=model_config.get('dropout', 0.1),
            max_seq_length=model_config.get('max_seq_length', 100)
        ).to(self.device)
        
        logger.info(f"MAT model built with {sum(p.numel() for p in self.model.parameters())} parameters")

    def prepare_data(self, txt_data: np.ndarray, ts_data: np.ndarray, 
                    target_data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for training/inference.
        
        Args:
            txt_data: Text/sentiment features
            ts_data: Time-series features  
            target_data: Target values
            
        Returns:
            Tuple of prepared tensors
        """
        # Convert to tensors
        x_txt = torch.FloatTensor(txt_data).to(self.device)
        x_ts = torch.FloatTensor(ts_data).to(self.device)
        x_target = torch.FloatTensor(target_data).to(self.device)
        
        return x_txt, x_ts, x_target

    def train(self, txt_data: np.ndarray, ts_data: np.ndarray, 
             target_data: np.ndarray, epochs: int = 100) -> Dict[str, Any]:
        """
        Train the MAT model.
        
        Args:
            txt_data: Text/sentiment features (samples, seq_len, txt_features)
            ts_data: Time-series features (samples, seq_len, ts_features)
            target_data: Target values (samples, pred_len, 1)
            epochs: Number of training epochs
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.build_model(txt_data.shape[-1], ts_data.shape[-1])
        
        # Prepare data
        x_txt, x_ts, x_target = self.prepare_data(txt_data, ts_data, target_data)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.config.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(x_txt, x_ts, x_target)
            loss = criterion(predictions, x_target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            scheduler.step(loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
        
        self.is_trained = True
        
        return {
            'success': True,
            'final_loss': losses[-1],
            'training_losses': losses,
            'epochs_trained': epochs
        }

    def predict(self, txt_data: np.ndarray, ts_data: np.ndarray, 
               decoder_input: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained MAT model.
        
        Args:
            txt_data: Text/sentiment features
            ts_data: Time-series features
            decoder_input: Decoder input (e.g., lagged target values)
            
        Returns:
            Predictions array
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            x_txt, x_ts, x_dec = self.prepare_data(txt_data, ts_data, decoder_input)
            predictions = self.model(x_txt, x_ts, x_dec)
            return predictions.cpu().numpy()

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'txt_features': self.model.txt_features,
            'ts_features': self.model.ts_features,
            'is_trained': self.is_trained
        }, filepath)
        
        logger.info(f"MAT model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        txt_features = checkpoint['txt_features']
        ts_features = checkpoint['ts_features']
        
        self.build_model(txt_features, ts_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint['is_trained']
        
        logger.info(f"MAT model loaded from {filepath}")


# Example usage and testing
if __name__ == '__main__':
    # Model parameters
    txt_features = 10  # Example: sentiment features
    ts_features = 5    # Example: technical indicators
    d_model = 512
    n_heads = 8
    n_encoder_layers = 2
    n_decoder_layers = 1
    d_ff = 2048
    dropout = 0.1
    
    # Create the model
    model = ModalityAwareTransformer(
        txt_features, ts_features, d_model, n_heads, 
        n_encoder_layers, n_decoder_layers, d_ff, dropout
    )
    
    # Example input tensors
    batch_size = 16
    seq_len_enc = 9  # Lookback window of 9 periods
    seq_len_dec = 3  # Prediction horizon of 3 periods
    
    x_txt = torch.randn(batch_size, seq_len_enc, txt_features)
    x_ts = torch.randn(batch_size, seq_len_enc, ts_features)
    x_dec = torch.randn(batch_size, seq_len_dec, 1)

    # Get model output
    output = model(x_txt, x_ts, x_dec)
    
    print("Modality-aware Transformer created successfully.")
    print("Example output shape:", output.shape)  # Expected: (batch_size, seq_len_dec, 1)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
