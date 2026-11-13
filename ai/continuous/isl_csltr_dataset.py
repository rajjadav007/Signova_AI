"""
ISL-CSLTR Continuous Dataset and Collation Module.

Phase 8.3.6 Implementation.
Ponytail design: Simple, robust, standard PyTorch Dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class ISLCSLTRDataset(Dataset):
    """PyTorch Dataset loading extracted MediaPipe landmarks for ISL-CSLTR."""

    def __init__(
        self,
        manifest_path: str | Path,
        features_root: str | Path,
        vocab_path: Optional[str | Path] = None,
        canonical_only: bool = True,
        feature_key: str = "spatial_features",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.features_root = Path(features_root)
        self.canonical_only = canonical_only
        self.feature_key = feature_key

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if self.canonical_only:
            self.records = [r for r in records if r.get("is_canonical", True)]
        else:
            self.records = records

        # Load vocabulary
        if vocab_path is not None:
            vpath = Path(vocab_path)
            with open(vpath, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            self.gloss_to_id: Dict[str, int] = vdata["gloss_to_id"]
            self.blank_index: int = vdata.get("blank_index", 0)
            self.num_classes: int = vdata.get("num_classes", len(self.gloss_to_id))
        else:
            self.gloss_to_id = {}
            self.blank_index = 0
            self.num_classes = 171

        self.id_to_gloss: Dict[int, str] = {v: k for k, v in self.gloss_to_id.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        npz_rel = record["npz_relative_path"]
        # Handle relative path from dataset root vs features directory
        npz_path = self.features_root / Path(npz_rel).name
        if not npz_path.exists():
            # Try direct relative resolution from dataset root
            npz_path = self.features_root.parent / npz_rel
            if not npz_path.exists():
                raise FileNotFoundError(f"Feature archive missing: {npz_path}")

        data = np.load(npz_path)
        if self.feature_key in data:
            feat = data[self.feature_key]  # (T, 300)
        elif "landmarks" in data:
            lm = data["landmarks"]  # (T, 75, 4)
            feat = lm.reshape(lm.shape[0], -1)  # (T, 300)
        else:
            raise KeyError(f"Neither '{self.feature_key}' nor 'landmarks' in {npz_path}")

        # Ensure float32
        feat = feat.astype(np.float32)

        # Target token IDs
        token_ids = record["token_ids"]
        tokens = record["tokens"]

        return {
            "video_id": record["video_id"],
            "features": torch.from_numpy(feat),  # (T, 300)
            "length": feat.shape[0],
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "tokens": tokens,
            "signer_id": record["signer_id"],
            "sentence": record["sentence"],
            "resolution": record.get("resolution", "unknown"),
        }


def collate_isl_csltr_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable-length sequences with zero-padding and length tracking."""
    batch_size = len(batch)
    lengths = [sample["length"] for sample in batch]
    max_len = max(lengths)
    feature_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    input_lengths = torch.tensor(lengths, dtype=torch.long)

    target_list: List[torch.Tensor] = []
    target_lengths_list: List[int] = []
    tokens_list: List[List[str]] = []
    video_ids: List[str] = []
    signers: List[int] = []
    resolutions: List[str] = []

    for i, sample in enumerate(batch):
        t_len = sample["length"]
        padded_features[i, :t_len, :] = sample["features"]

        t_ids = sample["token_ids"]
        target_list.append(t_ids)
        target_lengths_list.append(len(t_ids))
        tokens_list.append(sample["tokens"])
        video_ids.append(sample["video_id"])
        signers.append(sample["signer_id"])
        resolutions.append(sample["resolution"])

    # Concatenate targets 1D for PyTorch CTCLoss
    targets = torch.cat(target_list, dim=0)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)

    return {
        "features": padded_features,          # (B, max_len, 300)
        "input_lengths": input_lengths,        # (B,)
        "targets": targets,                    # (sum(target_lengths),)
        "target_lengths": target_lengths,      # (B,)
        "tokens_list": tokens_list,            # List[List[str]]
        "video_ids": video_ids,                # List[str]
        "signers": signers,                    # List[int]
        "resolutions": resolutions,            # List[str]
    }
