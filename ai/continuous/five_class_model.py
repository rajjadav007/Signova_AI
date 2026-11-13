"""
Signova AI - Five-Class ISL Sentence Temporal Sequence Model.
BiGRU with Bahdanau Self-Attention Pooling for variable-length landmark sequences.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttentionPooling(nn.Module):
    """
    Bahdanau-style Additive Self-Attention Pooling with padding mask support.
    score_t = v^T tanh(W h_t + b)
    alpha_t = softmax(score_t, masked)
    c = sum(alpha_t * h_t)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Tensor of shape (B, T, hidden_dim)
            mask: BoolTensor of shape (B, T), True for valid frames, False for padded
        Returns:
            context: Tensor of shape (B, hidden_dim)
            alpha: Tensor of shape (B, T)
        """
        # (B, T, attn_dim)
        score = self.v(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T)

        if mask is not None:
            # Mask padded positions with large negative value
            score = score.masked_fill(~mask, -1e9)

        alpha = F.softmax(score, dim=-1)  # (B, T)
        # Handle all-masked edge cases if any
        alpha = torch.nan_to_num(alpha, nan=0.0)

        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, hidden_dim)
        return context, alpha


class FiveClassBiGRUAttentionModel(nn.Module):
    """
    Temporal sequence classifier for 5-class ISL sentences.
    - Input projection: 300 -> 128 (LayerNorm + ReLU + Dropout)
    - Temporal encoder: 1-layer Bidirectional GRU (hidden_size=64 -> 128 output)
    - Attention pooling: Bahdanau self-attention
    - Classification head: 128 -> 64 -> 5
    """

    def __init__(
        self,
        input_dim: int = 300,
        proj_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 1. Input Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        encoder_dim = hidden_dim * 2  # bidirectional

        # 3. Attention Pooling
        self.attention = BahdanauAttentionPooling(encoder_dim, attn_dim=64)

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, T, input_dim)
            mask: BoolTensor of shape (B, T)
        Returns:
            logits: Tensor of shape (B, num_classes)
            attn_weights: Tensor of shape (B, T)
        """
        # Project frame features
        proj = self.input_proj(x)  # (B, T, proj_dim)

        # Temporal GRU
        gru_out, _ = self.gru(proj)  # (B, T, 2 * hidden_dim)

        # Attention pooling with mask
        context, attn_weights = self.attention(gru_out, mask=mask)  # (B, 2 * hidden_dim)

        # Logits
        logits = self.head(context)  # (B, num_classes)
        return logits, attn_weights

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "FiveClassBiGRUAttentionModel",
            "input_dim": self.input_dim,
            "proj_dim": self.proj_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
