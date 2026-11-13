"""
Bi-GRU + CTC Continuous ISL Baseline Model and Dataset.

Phase 8.3.2 Implementation:
- Input feature dimension: (T, 300) [75 landmarks x 4 channels (x, y, z, conf)]
- Linear feature projection: 300 -> 128 + LayerNorm + Dropout(0.3) + ReLU
- Temporal subsampling: Conv1D(128, 128, kernel=3, stride=2, padding=1) + LayerNorm + ReLU
- Two-layer Bidirectional GRU: hidden_size=128 per direction (output dimension 256), dropout=0.3
- Linear classifier: 256 -> 5
- CTC Loss with blank index 0
- Target vocabulary: 0=<blank>, 1=I, 2=GO, 3=EAT, 4=FOOD

Ponytail design: Minimal dependencies (torch, numpy), direct implementation, no excess boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

GLOSS_TO_ID: Dict[str, int] = {
    "blank": 0,
    "i": 1,
    "go": 2,
    "eat": 3,
    "food": 4,
}

ID_TO_GLOSS: Dict[int, str] = {v: k for k, v in GLOSS_TO_ID.items()}
NUM_CLASSES = 5
BLANK_INDEX = 0


def compute_subsampled_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output sequence lengths after Conv1D(kernel=3, stride=2, padding=1).
    
    Formula: floor((T + 2*padding - kernel) / stride) + 1
           = floor((T + 2 - 3) / 2) + 1 = floor((T - 1) / 2) + 1 = (T + 1) // 2
    """
    return (input_lengths + 1) // 2


class BiGRUCTCModel(nn.Module):
    """Two-layer Bidirectional GRU with Stride-2 Conv1D temporal subsampling for CTC."""

    def __init__(
        self,
        input_dim: int = 300,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # 1. Linear Feature Projection
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_ln = nn.LayerNorm(proj_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.proj_relu = nn.ReLU()

        # 2. Temporal Subsampling (Stride-2 Conv1D)
        self.conv = nn.Conv1d(
            in_channels=proj_dim,
            out_channels=proj_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_ln = nn.LayerNorm(proj_dim)
        self.conv_relu = nn.ReLU()

        # 3. Two-layer Bidirectional GRU (output dim = hidden_dim * 2 = 256)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Classifier Head
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, T, input_dim)
            lengths: Optional 1D Tensor of shape (B,) with original lengths T_i
            
        Returns:
            log_probs: Tensor of shape (T', B, num_classes) ready for torch.nn.CTCLoss
            subsampled_lengths: 1D Tensor of shape (B,) with subsampled lengths T'_i
        """
        b, t, c = x.shape
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=x.device)

        # 1. Projection
        h = self.proj(x)
        h = self.proj_ln(h)
        h = self.proj_dropout(h)
        h = self.proj_relu(h)

        # 2. Temporal Subsampling: conv expects (B, C, T)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)  # back to (B, T', C)
        h = self.conv_ln(h)
        h = self.conv_relu(h)

        subsampled_lengths = compute_subsampled_lengths(lengths)

        # 3. Recurrent Encoding
        # ponytail: pack_padded_sequence if lengths vary, else direct pass
        if lengths.min() != lengths.max():
            packed = nn.utils.rnn.pack_padded_sequence(
                h, subsampled_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(h)

        # 4. Classifier
        out = self.head_dropout(out)
        logits = self.classifier(out)  # (B, T', num_classes)

        # CTC Loss expects (T', B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, subsampled_lengths


def decode_greedy(
    log_probs: torch.Tensor,
    subsampled_lengths: torch.Tensor,
    blank_idx: int = BLANK_INDEX,
) -> List[Dict[str, Any]]:
    """Greedy CTC decoder with blank-collapse and consecutive duplicate removal.
    
    Args:
        log_probs: Tensor of shape (T', B, num_classes)
        subsampled_lengths: Tensor of shape (B,)
        blank_idx: Index of blank token (0)
        
    Returns:
        List of dicts per sample with:
            - 'token_ids': List[int]
            - 'tokens': List[str]
            - 'raw_preds': List[int]
            - 'blank_percentage': float
            - 'total_frames': int
    """
    # Transpose to (B, T', num_classes)
    probs_b = log_probs.detach().transpose(0, 1)
    results = []

    for i in range(probs_b.shape[0]):
        t_len = int(subsampled_lengths[i].item())
        sample_probs = probs_b[i, :t_len]
        raw_preds = torch.argmax(sample_probs, dim=-1).cpu().tolist()

        blank_count = sum(1 for p in raw_preds if p == blank_idx)
        blank_pct = (blank_count / t_len * 100.0) if t_len > 0 else 0.0

        # CTC collapse: deduplicate consecutive identical tokens, then drop blank
        collapsed: List[int] = []
        prev = None
        for p in raw_preds:
            if p != prev:
                if p != blank_idx:
                    collapsed.append(p)
                prev = p

        tokens = [ID_TO_GLOSS.get(idx, f"unk_{idx}") for idx in collapsed]

        results.append({
            "token_ids": collapsed,
            "tokens": tokens,
            "raw_preds": raw_preds,
            "blank_percentage": blank_pct,
            "total_frames": t_len,
        })

    return results


def compute_levenshtein(seq1: List[Any], seq2: List[Any]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


class ContinuousISLDataset(Dataset):
    """Dataset for continuous ISL features extracted in Phase 8.2."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        camera_view: str = "all",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.camera_view = camera_view

        manifest_path = self.dataset_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            all_records = json.load(f)

        self.records = [
            r for r in all_records
            if (split == "all" or r["split"] == split)
            and (camera_view == "all" or r["camera_view"] == camera_view)
        ]

        if len(self.records) == 0:
            raise ValueError(
                f"No records matching split='{split}' and camera_view='{camera_view}'"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        npz_path = self.dataset_dir / rec["npz_relative_path"]

        with np.load(npz_path) as data:
            spatial = data["spatial_features"].astype(np.float32)

        target_ids = [GLOSS_TO_ID[g.lower()] for g in rec["target_glosses"]]

        return {
            "features": torch.from_numpy(spatial),  # (T, 300)
            "feature_length": spatial.shape[0],
            "target": torch.tensor(target_ids, dtype=torch.long),
            "target_length": len(target_ids),
            "target_glosses": rec["target_glosses"],
            "sentence_id": rec["sentence_id"],
            "video_id": rec["video_id"],
            "camera_view": rec["camera_view"],
            "split": rec["split"],
            "isl_gloss": rec["isl_gloss"],
            "english": rec["english"],
        }


def collate_continuous_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length feature sequences and targets."""
    batch_size = len(batch)
    lengths = [item["feature_length"] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
    feature_lengths = torch.tensor(lengths, dtype=torch.long)

    for i, item in enumerate(batch):
        f = item["features"]
        padded_features[i, : f.shape[0], :] = f

    target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)
    targets_concat = torch.cat([item["target"] for item in batch], dim=0)

    return {
        "features": padded_features,
        "feature_lengths": feature_lengths,
        "targets": targets_concat,
        "target_lengths": target_lengths,
        "items": batch,
    }
