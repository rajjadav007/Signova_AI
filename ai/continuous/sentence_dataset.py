"""
Dataset and Batch Collation for 101-Class ISL Continuous Sentence Classification.

Module: ai.continuous.sentence_dataset
Features:
- ISLCSLTRSentenceDataset: PyTorch Dataset loading .npz archives, extracting 678 temporal features
- collate_sentence_batch: Pads variable-length sequences to batch max length with attention masks
- compute_smoothed_class_weights: Inverse-frequency class weight calculation with exponential smoothing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ai.continuous.temporal_features import extract_temporal_features


def augment_temporal_sequence(
    features: torch.Tensor,
    speed_range: Tuple[float, float] = (0.85, 1.15),
    time_mask_max_frames: int = 5,
    num_time_masks: int = 2,
    hand_dropout_prob: float = 0.1,
) -> torch.Tensor:
    """
    Apply physics-safe temporal data augmentation to 678-dim feature sequence (T, 678).

    - Speed jitter: resamples sequence with physically consistent derivative scaling
      (v scales by 1/s, a scales by 1/s^2, boolean detection masks snapped to {0, 1}).
    - SpecAugment-style frame time masking: zeroes out entire frames across all channels
      simultaneously, simulating frame drop / occlusions without breaking inter-channel consistency.
    - Hand dropout: randomly masks out left hand to mitigate student single-hand bias.
    """
    T, D = features.shape
    if T <= 4:
        return features

    out = features.clone()

    # 1. Temporal speed scaling with physical derivative correction
    if speed_range is not None and speed_range[0] < speed_range[1]:
        speed = random.uniform(*speed_range)
        if abs(speed - 1.0) > 0.02:
            new_T = max(4, int(round(T * speed)))
            feat_t = out.transpose(0, 1).unsqueeze(0)  # (1, D, T)
            resampled = F.interpolate(feat_t, size=new_T, mode="linear", align_corners=False).squeeze(0).transpose(0, 1)
            out = resampled
            out[:, 225:450] /= speed       # dx/dt scales by 1/speed
            out[:, 450:675] /= (speed**2)  # dv/dt scales by 1/speed^2
            out[:, 675:678] = (out[:, 675:678] > 0.5).float()
            T = new_T

    # 2. SpecAugment-style frame time masking
    if num_time_masks > 0 and time_mask_max_frames > 0:
        for _ in range(num_time_masks):
            mask_len = random.randint(1, min(time_mask_max_frames, max(1, T // 4)))
            t0 = random.randint(0, max(0, T - mask_len))
            out[t0 : t0 + mask_len, :] = 0.0

    # 3. Hand dropout (simulates single-hand signing to bridge native vs student domain shift)
    if hand_dropout_prob > 0.0 and random.random() < hand_dropout_prob:
        # Left Hand (landmarks 33..53)
        lh_spatial = slice(33 * 3, 54 * 3)
        out[:, lh_spatial] = 0.0
        out[:, 225 + 33 * 3 : 225 + 54 * 3] = 0.0
        out[:, 450 + 33 * 3 : 450 + 54 * 3] = 0.0
        out[:, 676] = 0.0  # left hand mask flag

    return out


class ISLCSLTRSentenceDataset(Dataset):
    """
    PyTorch Dataset for 101-class Indian Sign Language Continuous Sentence Classification.

    Reads manifest JSON (train_manifest.json, validation_manifest.json, test_manifest.json)
    or manifest.csv, loads .npz landmarks, computes 678 temporal features, and yields
    sentence_id targets.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        features_root: Optional[Union[str, Path]] = None,
        canonical_only: bool = True,
        fps: float = 25.0,
        cache_features: bool = True,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.features_root = Path(features_root) if features_root is not None else None
        self.canonical_only = canonical_only
        self.fps = fps
        self.cache_features = cache_features
        self.augment = augment
        self._cache: Dict[int, Dict[str, Any]] = {}

        # Resolve dataset root: manifest directory's parent (e.g. datasets/isl_csltr_continuous)
        self.dataset_root = self.manifest_path.parent.parent

        # Load records
        if self.manifest_path.suffix.lower() == ".json":
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
        elif self.manifest_path.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(self.manifest_path)
            raw_records = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported manifest format: {self.manifest_path.suffix}")

        if self.canonical_only:
            self.records = [r for r in raw_records if bool(r.get("is_canonical", True))]
        else:
            self.records = raw_records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_npz_path(self, npz_rel: str) -> Path:
        """Resolve full filesystem path to .npz feature file."""
        candidates = []
        if self.features_root:
            candidates.append(self.features_root / npz_rel)
            candidates.append(self.features_root / Path(npz_rel).name)

        candidates.append(self.dataset_root / npz_rel)
        candidates.append(self.dataset_root / "features" / Path(npz_rel).name)
        candidates.append(Path(npz_rel))

        for cand in candidates:
            if cand is not None and cand.exists():
                return cand

        raise FileNotFoundError(
            f"Could not locate NPZ file '{npz_rel}'. Checked candidates: {[str(c) for c in candidates]}"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache_features and idx in self._cache:
            base_item = self._cache[idx]
        else:
            record = self.records[idx]
            npz_rel = str(record["npz_relative_path"])
            npz_path = self._resolve_npz_path(npz_rel)

            data = np.load(npz_path)
            landmarks = data["landmarks"].astype(np.float32)  # (T, 75, 4)
            detection_mask = data["detection_mask"]  # (T, 3)
            timestamps_ms = data["timestamps_ms"] if "timestamps_ms" in data else None

            # Extract 678-dimensional temporal features
            features_np = extract_temporal_features(
                landmarks=landmarks,
                detection_mask=detection_mask,
                timestamps_ms=timestamps_ms,
                fps=self.fps,
            )
            features = torch.from_numpy(features_np)  # (T, 678) float32
            length = int(features.shape[0])
            label = int(record["sentence_id"])
            sentence = str(record.get("sentence", ""))
            video_id = str(record.get("video_id", ""))
            signer_id = int(record.get("signer_id", 0))

            base_item = {
                "features": features,
                "length": length,
                "label": label,
                "sentence": sentence,
                "video_id": video_id,
                "signer_id": signer_id,
            }

            if self.cache_features:
                self._cache[idx] = base_item

        if self.augment:
            aug_features = augment_temporal_sequence(base_item["features"])
            return {
                **base_item,
                "features": aug_features,
                "length": int(aug_features.shape[0]),
            }

        return base_item


def collate_sentence_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of sentence dictionaries into a zero-padded batch with attention masks.

    Args:
        batch: List of dicts, each containing:
            - 'features': Tensor of shape (T, 678)
            - 'length': int
            - 'label': int
            - 'sentence': str
            - 'video_id': str
            - 'signer_id': int

    Returns:
        Dict containing:
            - 'features': Tensor of shape (B, max_len, 678) float32
            - 'lengths': Tensor of shape (B,) long
            - 'labels': Tensor of shape (B,) long
            - 'attention_mask': Tensor of shape (B, max_len) bool (True for valid frames, False for padding)
            - 'sentences': List of sentence strings
            - 'video_ids': List of video ID strings
            - 'signer_ids': List of signer IDs
    """
    B = len(batch)
    lengths = [int(item["length"]) for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[-1]

    padded_features = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)
    labels = []
    sentences = []
    video_ids = []
    signer_ids = []

    for i, item in enumerate(batch):
        f = item["features"]
        if not isinstance(f, torch.Tensor):
            f = torch.tensor(f, dtype=torch.float32)
        l = lengths[i]
        padded_features[i, :l] = f[:l]
        attention_mask[i, :l] = True
        labels.append(int(item.get("label", 0)))
        sentences.append(str(item.get("sentence", "")))
        video_ids.append(str(item.get("video_id", "")))
        signer_ids.append(int(item.get("signer_id", 0)))

    out_batch: Dict[str, Any] = {
        "features": padded_features,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": attention_mask,
        "sentences": sentences,
        "video_ids": video_ids,
        "signer_ids": signer_ids,
        # Singular aliases for caller compatibility
        "sentence": sentences,
        "video_id": video_ids,
        "signer_id": signer_ids,
    }
    return out_batch


def compute_smoothed_class_weights(
    manifest_path: Union[str, Path],
    num_classes: int = 101,
    smoothing: float = 0.5,
) -> torch.Tensor:
    """
    Compute smoothed inverse-frequency class weights for 101 ISL sentence classes.

    Formula:
        count_c = occurrences of class c in manifest
        effective_count = count_c if count_c > 0 else min(counts > 0)
        weight_c = 1.0 / (effective_count ** smoothing)
        normalized_weight = weight_c / mean(weights)

    Args:
        manifest_path: Path to manifest JSON or CSV file
        num_classes: Total number of classes (default: 101)
        smoothing: Exponential smoothing factor in [0.0, 1.0] (default: 0.5)

    Returns:
        Tensor of shape (num_classes,) float32 with mean = 1.0
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    counts = np.zeros(num_classes, dtype=np.float32)

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            if bool(r.get("is_canonical", True)):
                sid = int(r["sentence_id"])
                if 0 <= sid < num_classes:
                    counts[sid] += 1.0
    elif path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        if "is_canonical" in df.columns:
            df = df[df["is_canonical"] == True]
        for sid in df["sentence_id"]:
            sid = int(sid)
            if 0 <= sid < num_classes:
                counts[sid] += 1.0

    min_count = np.min(counts[counts > 0]) if np.any(counts > 0) else 1.0
    effective_counts = np.where(counts > 0, counts, min_count)

    raw_weights = 1.0 / (effective_counts ** float(smoothing))
    mean_w = np.mean(raw_weights)
    if mean_w > 0:
        normalized_weights = raw_weights / mean_w
    else:
        normalized_weights = np.ones(num_classes, dtype=np.float32)

    return torch.tensor(normalized_weights, dtype=torch.float32)
