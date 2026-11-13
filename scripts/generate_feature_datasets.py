"""
F-021: Feature Engineering and Normalization Pipeline for Isolated ISL Sign Recognition.
Transforms raw landmark extractions into fixed-length (T=64), spatially-normalized,
600-dimensional feature sequences (300 spatial + 300 masked velocity).
Outputs:
  - datasets/features/{train,validation,test}/*.npz
  - datasets/features/feature_index.csv
  - datasets/features/feature_summary.json
  - docs/reports/figures/feature_*.png
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Set MPLCONFIGDIR before importing matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "C:/tmp")) / "matplotlib_signova"))
import matplotlib.pyplot as plt

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    FEATURES_DIR,
    ISOLATED_CLASSES,
    LANDMARKS_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
)
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample

FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"


def process_features():
    print("=" * 70)
    print("STARTING F-021: FEATURE DATASET GENERATION")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]
    index_rows: List[Dict[str, Any]] = []
    split_stats: Dict[str, Dict[str, Any]] = {}
    class_velocity_sums: Dict[str, np.ndarray] = {cls: np.zeros(DEFAULT_SEQUENCE_LENGTH, dtype=np.float64) for cls in ISOLATED_CLASSES}
    class_sample_counts: Dict[str, int] = {cls: 0 for cls in ISOLATED_CLASSES}

    total_samples = 0
    total_bytes = 0

    sample_for_temporal_plot = None
    sample_for_velocity_plot = None

    t_start = time.perf_counter()

    for split in splits:
        split_dir = FEATURES_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_csv = SPLITS_DIR / f"{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")

        with open(split_csv, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        print(f"\nProcessing split: {split.upper()} ({len(reader)} clips)")

        split_features_list = []
        split_class_counts: Dict[str, int] = {cls: 0 for cls in ISOLATED_CLASSES}
        split_bytes = 0

        for row in reader:
            vid = row["video_id"]
            class_label = row["class_label"]
            raw_npz_path = LANDMARKS_DIR / split / f"{vid}.npz"

            if not raw_npz_path.exists():
                raise FileNotFoundError(f"Landmark file missing: {raw_npz_path}")

            raw_data = np.load(raw_npz_path)
            landmarks = raw_data["landmarks"]  # (T, 75, 4)
            detection_mask = raw_data["detection_mask"] if "detection_mask" in raw_data else None
            orig_frames = landmarks.shape[0]

            # 1. Spatial Normalization
            normed_landmarks = spatial_normalize(landmarks)

            # 2. Fixed-length Temporal Resampling (T=64)
            resampled_landmarks, resampled_mask, resampled_ts = temporal_resample(
                normed_landmarks,
                target_length=DEFAULT_SEQUENCE_LENGTH,
                detection_mask=detection_mask,
                return_metadata=True,
            )

            # 3. Masked Velocity and Combined Feature Extraction
            combined_600, spatial_300, velocity_300 = extract_isolated_features(
                resampled_landmarks,
                return_components=True,
            )

            # Sanity checks
            if np.isnan(combined_600).any() or np.isinf(combined_600).any():
                raise ValueError(f"NaN or Inf detected in features for {vid}!")

            assert combined_600.shape == (DEFAULT_SEQUENCE_LENGTH, 600)
            assert spatial_300.shape == (DEFAULT_SEQUENCE_LENGTH, 300)
            assert velocity_300.shape == (DEFAULT_SEQUENCE_LENGTH, 300)
            assert resampled_landmarks.shape == (DEFAULT_SEQUENCE_LENGTH, 75, 4)
            assert resampled_mask.shape == (DEFAULT_SEQUENCE_LENGTH, 3)
            assert resampled_ts.shape == (DEFAULT_SEQUENCE_LENGTH,)

            # Save sample representations for plots
            if sample_for_temporal_plot is None and class_label == "hello":
                sample_for_temporal_plot = {
                    "orig_landmarks": landmarks,
                    "normed_landmarks": normed_landmarks,
                    "resampled_landmarks": resampled_landmarks,
                    "class_label": class_label,
                    "video_id": vid,
                }
            if sample_for_velocity_plot is None and class_label == "stop":
                sample_for_velocity_plot = {
                    "spatial": spatial_300,
                    "velocity": velocity_300,
                    "class_label": class_label,
                    "video_id": vid,
                }

            # Accumulate velocity magnitude profile across time for class
            # Right hand wrist is landmark 54 (or pose right wrist 16)
            rh_vel_mag = np.linalg.norm(velocity_300[:, 54 * 4 : 54 * 4 + 3], axis=1)
            lh_vel_mag = np.linalg.norm(velocity_300[:, 33 * 4 : 33 * 4 + 3], axis=1)
            hand_vel_mag = np.maximum(rh_vel_mag, lh_vel_mag)
            class_velocity_sums[class_label] += hand_vel_mag
            class_sample_counts[class_label] += 1

            # Save compressed .npz
            out_npz_path = split_dir / f"{vid}.npz"
            np.savez_compressed(
                out_npz_path,
                features=combined_600,
                spatial_features=spatial_300,
                velocity_features=velocity_300,
                normalized_landmarks=resampled_landmarks,
                detection_mask=resampled_mask,
                timestamps_ms=resampled_ts,
            )

            file_size = out_npz_path.stat().st_size
            split_bytes += file_size
            total_bytes += file_size
            total_samples += 1

            split_features_list.append(combined_600)
            split_class_counts[class_label] = split_class_counts.get(class_label, 0) + 1

            index_rows.append({
                "video_id": vid,
                "split": split,
                "class_label": class_label,
                "signer_id": row.get("signer_id", "unknown"),
                "source_dataset": row.get("source_dataset", "unknown"),
                "original_frames": orig_frames,
                "resampled_frames": DEFAULT_SEQUENCE_LENGTH,
                "feature_dim": 600,
                "file_size_bytes": file_size,
                "npz_path": str(out_npz_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            })

        split_all_features = np.concatenate(split_features_list, axis=0)  # (N * 64, 600)
        split_spatial = split_all_features[:, :300]
        split_velocity = split_all_features[:, 300:]
        nonzero_vel_ratio = float(np.count_nonzero(split_velocity) / split_velocity.size)

        split_stats[split] = {
            "samples": len(reader),
            "class_distribution": split_class_counts,
            "total_bytes": split_bytes,
            "spatial_min": round(float(np.min(split_spatial)), 4),
            "spatial_max": round(float(np.max(split_spatial)), 4),
            "spatial_mean": round(float(np.mean(split_spatial)), 4),
            "spatial_std": round(float(np.std(split_spatial)), 4),
            "velocity_min": round(float(np.min(split_velocity)), 4),
            "velocity_max": round(float(np.max(split_velocity)), 4),
            "velocity_mean": round(float(np.mean(split_velocity)), 4),
            "velocity_std": round(float(np.std(split_velocity)), 4),
            "velocity_nonzero_ratio": round(nonzero_vel_ratio, 4),
        }
        print(f"  Processed {len(reader)} samples ({split_bytes / 1024:.1f} KB). Vel non-zero: {nonzero_vel_ratio * 100:.2f}%")

    # 4. Save feature index CSV
    index_csv_path = FEATURES_DIR / "feature_index.csv"
    with open(index_csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_id",
            "split",
            "class_label",
            "signer_id",
            "source_dataset",
            "original_frames",
            "resampled_frames",
            "feature_dim",
            "file_size_bytes",
            "npz_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"\nSaved feature index to: {index_csv_path} ({len(index_rows)} rows)")

    # 5. Save summary JSON
    elapsed_time = round(time.perf_counter() - t_start, 2)
    summary_data = {
        "status": "success",
        "total_samples": total_samples,
        "total_file_size_bytes": total_bytes,
        "total_file_size_mb": round(total_bytes / (1024 * 1024), 2),
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "feature_dimensions": {
            "total": 600,
            "spatial": 300,
            "velocity": 300,
            "landmarks_count": 75,
            "channels": 4,
        },
        "target_classes": ISOLATED_CLASSES,
        "splits": split_stats,
        "elapsed_seconds": elapsed_time,
    }

    summary_json_path = FEATURES_DIR / "feature_summary.json"
    with open(summary_json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved feature summary to: {summary_json_path}")

    # 6. Generate Figures
    generate_figures(sample_for_temporal_plot, sample_for_velocity_plot, class_velocity_sums, class_sample_counts)

    print("\n" + "=" * 70)
    print(f"F-021 FEATURE DATASET GENERATION COMPLETE in {elapsed_time}s")
    print("=" * 70)


def generate_figures(sample_temporal, sample_vel, class_vel_sums, class_counts):
    print("\nGenerating visual verification figures...")

    # Figure 1: Temporal Resampling Comparison
    if sample_temporal is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        orig = sample_temporal["orig_landmarks"]
        resamp = sample_temporal["resampled_landmarks"]
        # Plot Right Wrist (index 16 in pose, index 54 in right hand)
        rw_orig_x = orig[:, 16, 0]
        rw_orig_y = orig[:, 16, 1]
        rw_resamp_x = resamp[:, 16, 0]
        rw_resamp_y = resamp[:, 16, 1]

        t_orig = np.arange(orig.shape[0])
        t_resamp = np.arange(resamp.shape[0])

        axes[0].plot(t_orig, rw_orig_x, label="Right Wrist X", color="#1f77b4", lw=2)
        axes[0].plot(t_orig, rw_orig_y, label="Right Wrist Y", color="#ff7f0e", lw=2)
        axes[0].set_title(f"Original Landmark Trajectory: {sample_temporal['class_label']} (T={orig.shape[0]} frames)")
        axes[0].set_ylabel("Normalized Coord")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(t_resamp, rw_resamp_x, label="Resampled Wrist X", color="#1f77b4", lw=2, linestyle="--")
        axes[1].plot(t_resamp, rw_resamp_y, label="Resampled Wrist Y", color="#ff7f0e", lw=2, linestyle="--")
        axes[1].set_title(f"Uniform Resampled Trajectory (T={resamp.shape[0]} frames, strict zero-masked)")
        axes[1].set_xlabel("Normalized Frame Index (0..63)")
        axes[1].set_ylabel("Normalized Coord")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        fig_path = FIGURES_DIR / "feature_temporal_resampling.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {fig_path.name}")

    # Figure 2: Spatial vs Velocity Heatmap / Comparison
    if sample_vel is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        spatial = sample_vel["spatial"]  # (64, 300)
        velocity = sample_vel["velocity"]  # (64, 300)

        im1 = axes[0].imshow(spatial.T, aspect="auto", cmap="viridis", interpolation="none")
        axes[0].set_title(f"Spatial Features (300 dims) across 64 frames - Class: '{sample_vel['class_label']}'")
        axes[0].set_ylabel("Spatial Channels")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(velocity.T, aspect="auto", cmap="coolwarm", interpolation="none")
        axes[1].set_title("Masked Velocity Features (300 dims) across 64 frames (Strict Zero When Absent)")
        axes[1].set_xlabel("Frame Index (0..63)")
        axes[1].set_ylabel("Velocity Channels")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "feature_spatial_vs_velocity.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved figure: {fig_path.name}")

    # Figure 3: Mean Velocity Magnitude Profiles by Class
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in ISOLATED_CLASSES:
        if class_counts[cls] > 0:
            mean_profile = class_vel_sums[cls] / class_counts[cls]
            ax.plot(mean_profile, label=f"{cls} (n={class_counts[cls]})", lw=1.8)

    ax.set_title("Mean Hand Velocity Profile Across 64 Frames by Sign Class")
    ax.set_xlabel("Normalized Frame Index (0..63)")
    ax.set_ylabel("Mean Hand Velocity Magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "feature_class_velocity_profile.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig_path.name}")


if __name__ == "__main__":
    process_features()
