"""
Temporal Sentence Classifier for 101-Class Indian Sign Language Recognition.

Module: ai.continuous.sentence_classifier
Architecture:
1. Feature Projection: Linear(678, 256) -> LayerNorm(256) -> GELU -> Dropout(0.2)
2. Temporal Conv1D Blocks: 2 residual blocks of Conv1d(256, 256, k=3, p=1), BatchNorm1d(256), GELU, Dropout(0.2)
3. Bidirectional GRU: nn.GRU(256, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2) -> Output (B, T, 256)
4. Temporal Attention Pooling: TemporalAttention compresses (B, T, 256) -> (B, 256) with mask support
5. Classification Head: Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 101)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention pooling with masking support for variable-length sequences.

    Applies projection with tanh to compute unnormalized attention scores, masks out
    padded frames with -inf (or -1e9) so they receive zero probability in softmax,
    and returns the pooled context vector and attention weights.
    """

    def __init__(self, hidden_dim: int, attn_dim: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if attn_dim is None:
            attn_dim = max(hidden_dim // 2, 64)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward attention pass.

        Args:
            x: Tensor of shape (B, T, hidden_dim)
            attention_mask: Optional boolean tensor of shape (B, T) where True
                            indicates valid frames and False indicates padding.

        Returns:
            context: Context vector of shape (B, hidden_dim)
            weights: Attention weights of shape (B, T)
        """
        scores = self.projection(x).squeeze(-1)  # (B, T)

        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, -1e9)

        weights = F.softmax(scores, dim=-1)  # (B, T)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Context vector: weighted sum over time
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, hidden_dim)
        return context, weights


class TemporalConv1DBlock(nn.Module):
    """Residual Conv1D block with LayerNorm for padding-safe temporal feature processing."""

    def __init__(self, channels: int = 256, kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # LayerNorm over channels per timestep: avoids BatchNorm padding pollution across variable lengths
        self.ln = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Input x: (B, T, C), Output: (B, T, C)."""
        residual = x
        h = x.transpose(1, 2)  # (B, C, T)
        h = self.conv(h)
        h = h.transpose(1, 2)  # (B, T, C)
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        out = residual + h
        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1)
        return out


class TemporalSentenceClassifier(nn.Module):
    """
    101-Class Continuous Indian Sign Language Sentence Classifier.

    Combines modality-group LayerNorm, temporal convolution residual blocks with LayerNorm,
    2-layer Bidirectional GRU, masked temporal attention pooling, and classification head.
    """

    def __init__(
        self,
        input_dim: int = 678,
        num_classes: int = 101,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Modality-group LayerNorm: balances spatial, velocity, acceleration independently
        self.spatial_ln = nn.LayerNorm(225)
        self.vel_ln = nn.LayerNorm(225)
        self.acc_ln = nn.LayerNorm(225)

        # 1. Input Projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Temporal Conv1D Residual Blocks
        self.conv_block1 = TemporalConv1DBlock(hidden_dim, kernel_size=3, dropout=dropout)
        self.conv_block2 = TemporalConv1DBlock(hidden_dim, kernel_size=3, dropout=dropout)

        # 3. Bidirectional GRU (hidden_size = hidden_dim // 2, so bidirectional output = hidden_dim)
        gru_hidden = hidden_dim // 2
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Temporal Attention Pooling
        self.attention = TemporalAttention(hidden_dim=hidden_dim)

        # 5. Classification Head: Linear(256, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 101)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, input_dim)
            lengths: Optional 1D tensor of sequence lengths (B,)
            attention_mask: Optional boolean tensor (B, T), True for valid frames

        Returns:
            logits: Output classification logits of shape (B, num_classes)
        """
        B, T, D = x.shape

        if attention_mask is None:
            if lengths is not None:
                device = x.device
                attention_mask = (
                    torch.arange(T, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)
                )
            else:
                attention_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)

        # 0. Modality-group normalization: scale spatial, velocity, acceleration to unit variance
        s = self.spatial_ln(x[:, :, :225])
        v = self.vel_ln(x[:, :, 225:450])
        a = self.acc_ln(x[:, :, 450:675])
        m = x[:, :, 675:678]
        x_norm = torch.cat([s, v, a, m], dim=-1)
        if attention_mask is not None:
            x_norm = x_norm * attention_mask.unsqueeze(-1)

        # 1. Projection
        h = self.proj(x_norm)  # (B, T, hidden_dim)
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)

        # 2. Temporal Conv1D blocks with attention mask re-zeroing
        h = self.conv_block1(h, attention_mask=attention_mask)
        h = self.conv_block2(h, attention_mask=attention_mask)

        # 3. Bidirectional GRU
        if lengths is not None and lengths.min() != lengths.max():
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.gru(h)  # (B, T, hidden_dim)

        # 4. Temporal Attention Pooling
        context, weights = self.attention(out, attention_mask=attention_mask)
        self._last_attn_weights = weights

        # 5. Classification Head
        logits = self.head(context)  # (B, num_classes)
        return logits

    @torch.no_grad()
    def predict(
        self,
        x: Union[torch.Tensor, np.ndarray],
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Perform inference prediction.

        Args:
            x: Input features (B, T, 678) or (T, 678)
            lengths: Optional sequence length tensor (B,)

        Returns:
            dict with:
                - 'logits': Logits tensor (B, num_classes)
                - 'probabilities': Softmax probability tensor (B, num_classes)
                - 'predicted_class': Int (if B==1) or Tensor/list
                - 'confidence': Float (if B==1) or Tensor/list
                - 'attention_weights': Attention weights tensor (B, T)
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        else:
            x_tensor = x.to(device=device, dtype=torch.float32)

        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        B, T, _ = x_tensor.shape

        if lengths is None:
            lengths_tensor = torch.full((B,), T, dtype=torch.long, device=device)
        else:
            lengths_tensor = lengths.to(device=device, dtype=torch.long)

        attention_mask = (
            torch.arange(T, device=device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
        )

        logits = self.forward(x_tensor, lengths=lengths_tensor, attention_mask=attention_mask)
        probs = F.softmax(logits, dim=-1)
        conf, pred_class = torch.max(probs, dim=-1)

        result: Dict[str, Any] = {
            "logits": logits,
            "probabilities": probs,
            "attention_weights": self._last_attn_weights,
        }

        if B == 1:
            result["predicted_class"] = int(pred_class[0].item())
            result["confidence"] = float(conf[0].item())
        else:
            result["predicted_class"] = pred_class
            result["confidence"] = conf

        return result
