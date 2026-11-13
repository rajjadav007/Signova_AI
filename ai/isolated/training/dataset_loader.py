"""
Dataset loading and batching utilities for Individual ISL Sign Recognition.
Loads pre-extracted .npz landmark files from datasets/landmarks/{split}/.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    FEATURES_DIR,
    LANDMARKS_DIR,
    SPLITS_DIR,
)
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample


def load_split_dataset(
    split: str = "train",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    include_motion: bool = True,
    apply_spatial_norm: bool = True,
    from_precomputed_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Union[str, int]]]]:
    """
    Loads landmark or feature arrays from disk for a given split (train/validation/test).
    Returns:
      X: np.ndarray of shape (N, sequence_length, feature_dim)
      y: np.ndarray of shape (N,) with integer class labels (0..9)
      metadata: List of metadata dicts per sample.
    """
    csv_path = SPLITS_DIR / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV file not found: {csv_path}")

    X_list = []
    y_list = []
    meta_list = []

    features_split_dir = FEATURES_DIR / split
    landmarks_split_dir = LANDMARKS_DIR / split

    use_precomputed = (
        from_precomputed_features
        and sequence_length == DEFAULT_SEQUENCE_LENGTH
        and apply_spatial_norm
        and features_split_dir.exists()
    )

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            class_label = row["class_label"]
            if class_label not in CLASS_TO_IDX:
                continue

            if use_precomputed:
                feat_npz = features_split_dir / f"{vid}.npz"
                if not feat_npz.exists():
                    continue
                data = np.load(feat_npz)
                if include_motion:
                    features = data["features"]  # (64, 600)
                else:
                    features = data["spatial_features"]  # (64, 300)
                orig_frames = int(data["normalized_landmarks"].shape[0])
            else:
                npz_path = landmarks_split_dir / f"{vid}.npz"
                if not npz_path.exists():
                    continue

                data = np.load(npz_path)
                landmarks = data["landmarks"]  # (T, 75, 4)

                if apply_spatial_norm:
                    landmarks = spatial_normalize(landmarks)

                # Uniform temporal resampling
                resampled = temporal_resample(landmarks, target_length=sequence_length)

                if include_motion:
                    features = extract_isolated_features(resampled)  # (target_length, 600)
                else:
                    features = resampled.reshape(sequence_length, -1)  # (target_length, 300)
                orig_frames = landmarks.shape[0]

            X_list.append(features)
            y_list.append(CLASS_TO_IDX[class_label])
            meta_list.append({
                "video_id": vid,
                "class_label": class_label,
                "signer_id": row.get("signer_id", ""),
                "source_dataset": row.get("source_dataset", ""),
                "original_frames": orig_frames,
            })

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    return X, y, meta_list
