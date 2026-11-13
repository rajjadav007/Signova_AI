"""
Generates 600-dimensional features (300 spatial + 300 masked velocity) for 20-class ISL dataset.
Reuses existing precomputed features and processes only new clips.
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample

LANDMARKS_DIR = PROJECT_ROOT / "datasets" / "landmarks"
FEATURES_DIR = PROJECT_ROOT / "datasets" / "features"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"

CLASSES_20 = [
    "food", "hello", "help", "no", "please", "sorry", "stop", "thank you", "water", "yes",
    "father", "mother", "sister", "teacher", "you", "friend", "student", "school", "market", "hospital"
]
CLASS_TO_IDX_20 = {cls: idx for idx, cls in enumerate(CLASSES_20)}

def process_features():
    t0 = time.perf_counter()
    splits = ["train", "validation", "test"]

    total_attempted = 0
    total_reused = 0
    total_computed = 0
    total_failed = 0

    index_rows = []

    for split in splits:
        split_feat_dir = FEATURES_DIR / split
        split_feat_dir.mkdir(parents=True, exist_ok=True)

        split_csv = SPLITS_DIR / f"{split}_20class.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        with open(split_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        print(f"Processing split {split}: {len(reader)} clips")

        for row in reader:
            total_attempted += 1
            vid = row["video_id"]
            class_label = row["class_label"]
            out_feat_npz = split_feat_dir / f"{vid}.npz"

            if out_feat_npz.exists():
                total_reused += 1
                feat_data = np.load(out_feat_npz)
                index_rows.append({
                    "video_id": vid,
                    "split": split,
                    "class_label": class_label,
                    "class_idx": CLASS_TO_IDX_20[class_label],
                    "feature_path": str(out_feat_npz.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "feature_shape": "(64, 600)",
                    "spatial_dim": 300,
                    "velocity_dim": 300,
                    "reused": True,
                })
                continue

            raw_npz_path = LANDMARKS_DIR / split / f"{vid}.npz"
            if not raw_npz_path.exists():
                print(f"Warning: missing landmarks for {vid} at {raw_npz_path}")
                total_failed += 1
                continue

            try:
                raw_data = np.load(raw_npz_path)
                landmarks = raw_data["landmarks"]
                det_mask = raw_data.get("detection_mask", None)
                orig_frames = landmarks.shape[0]

                # 1. Spatial normalization
                normed = spatial_normalize(landmarks)

                # 2. Temporal resampling
                resampled, res_mask, res_ts = temporal_resample(
                    normed,
                    target_length=64,
                    detection_mask=det_mask,
                    return_metadata=True,
                )

                # 3. Combined features (300 spatial + 300 velocity)
                comb_600, spat_300, vel_300 = extract_isolated_features(
                    resampled,
                    return_components=True,
                )

                np.savez_compressed(
                    out_feat_npz,
                    features=comb_600,
                    spatial_features=spat_300,
                    velocity_features=vel_300,
                    normalized_landmarks=resampled,
                    resampled_mask=res_mask,
                    resampled_timestamps_ms=res_ts,
                    original_frames=orig_frames,
                )
                total_computed += 1
                index_rows.append({
                    "video_id": vid,
                    "split": split,
                    "class_label": class_label,
                    "class_idx": CLASS_TO_IDX_20[class_label],
                    "feature_path": str(out_feat_npz.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "feature_shape": "(64, 600)",
                    "spatial_dim": 300,
                    "velocity_dim": 300,
                    "reused": False,
                })
            except Exception as e:
                print(f"Failed processing features for {vid}: {e}")
                total_failed += 1

    # Save index
    import pandas as pd
    idx_df = pd.DataFrame(index_rows)
    idx_df.to_csv(FEATURES_DIR / "feature_index_20class.csv", index=False)

    summary = {
        "version": "2.0.0",
        "total_attempted": total_attempted,
        "total_reused": total_reused,
        "total_computed": total_computed,
        "total_failed": total_failed,
        "classes_count": len(CLASSES_20),
        "sequence_length": 64,
        "feature_dim": 600,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    with open(FEATURES_DIR / "feature_summary_20class.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Features complete in {summary['elapsed_seconds']}s. Reused: {total_reused}, Computed: {total_computed}, Failed: {total_failed}")

if __name__ == "__main__":
    process_features()
