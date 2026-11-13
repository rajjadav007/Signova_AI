"""
F-020: MediaPipe Landmark Extraction Pipeline for 10 ISL Classes.
Processes datasets/splits/{train,validation,test}.csv, extracts 75 landmarks
using ModernMediaPipeExtractor, and serializes compressed .npz archives
along with metadata indices and summary reports.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signova.f020_extraction")

LANDMARKS_DIR = PROJECT_ROOT / "datasets" / "landmarks"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"
FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"

def extract_all_splits():
    extractor = ModernMediaPipeExtractor()
    
    splits = ["train", "validation", "test"]
    all_records = []
    summary_stats = {
        "version": "1.0.0",
        "pipeline": "ModernMediaPipeExtractor (Tasks API: PoseLandmarker + HandLandmarker)",
        "landmarks_spec": "75 landmarks (33 pose, 21 left hand, 21 right hand x 4 channels [x,y,z,conf] = 300 features)",
        "total_attempted": 0,
        "total_successful": 0,
        "total_failed": 0,
        "splits": {},
        "failures": []
    }

    t0_global = time.perf_counter()

    for split_name in splits:
        csv_path = SPLITS_DIR / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Processing split '{split_name}' ({len(df)} clips)...")

        out_split_dir = LANDMARKS_DIR / split_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        split_success = 0
        split_fail = 0
        split_frames = 0
        pose_rates = []
        lh_rates = []
        rh_rates = []

        for idx, row in df.iterrows():
            summary_stats["total_attempted"] += 1
            vid = row["video_id"]
            class_label = row["class_label"]
            vpath = Path(row["absolute_video_path"])
            out_npz = out_split_dir / f"{vid}.npz"

            if not vpath.exists():
                err_msg = f"Video file missing: {vpath}"
                logger.error(err_msg)
                summary_stats["total_failed"] += 1
                split_fail += 1
                summary_stats["failures"].append({"video_id": vid, "error": err_msg})
                continue

            try:
                raw_spatial, meta = extractor.extract_isign_video(vpath)
                n_frames = meta["read_frames"]
                split_frames += n_frames
                
                # Reshape to (T, 75, 4)
                landmarks = raw_spatial.reshape(-1, 75, 4).astype(np.float32)
                spatial_features = raw_spatial.astype(np.float32)
                
                # Monotonic timestamps
                fps = meta["fps"] if meta["fps"] > 0 else 30.0
                timestamps_ms = np.array([int(round(i * 1000.0 / fps)) for i in range(n_frames)], dtype=np.int64)
                frame_indices = np.arange(n_frames, dtype=np.int32)
                
                # Detection mask: (T, 3) -> [pose, left_hand, right_hand]
                det_pose = np.any(raw_spatial[:, 0:132] != 0, axis=1)
                det_lh = np.any(raw_spatial[:, 132:216] != 0, axis=1)
                det_rh = np.any(raw_spatial[:, 216:300] != 0, axis=1)
                detection_mask = np.column_stack([det_pose, det_lh, det_rh]).astype(bool)

                # Save compressed npz
                np.savez_compressed(
                    out_npz,
                    landmarks=landmarks,
                    spatial_features=spatial_features,
                    timestamps_ms=timestamps_ms,
                    frame_indices=frame_indices,
                    detection_mask=detection_mask
                )

                split_success += 1
                summary_stats["total_successful"] += 1
                pose_rates.append(meta["pose_detection_rate"])
                lh_rates.append(meta["left_hand_detection_rate"])
                rh_rates.append(meta["right_hand_detection_rate"])

                all_records.append({
                    "video_id": vid,
                    "split": split_name,
                    "class_label": class_label,
                    "source_dataset": row["source_dataset"],
                    "signer_id": row["signer_id"],
                    "relative_video_path": row["relative_video_path"],
                    "npz_path": str(out_npz.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "frame_count": n_frames,
                    "spatial_dim": 300,
                    "landmark_shape": f"({n_frames}, 75, 4)",
                    "fps": meta["fps"],
                    "resolution": meta["resolution"],
                    "pose_detection_rate": meta["pose_detection_rate"],
                    "left_hand_detection_rate": meta["left_hand_detection_rate"],
                    "right_hand_detection_rate": meta["right_hand_detection_rate"],
                    "extraction_fps": meta["extraction_fps"],
                    "extraction_time_sec": meta["extraction_time_seconds"],
                    "extraction_status": "SUCCESS"
                })

                if (idx + 1) % 15 == 0 or (idx + 1) == len(df):
                    logger.info(f"[{split_name}] Processed {idx + 1}/{len(df)} clips (success={split_success}, fail={split_fail})")

            except Exception as e:
                err_msg = str(e)
                logger.error(f"Error extracting {vid} ({vpath}): {err_msg}")
                summary_stats["total_failed"] += 1
                split_fail += 1
                summary_stats["failures"].append({"video_id": vid, "error": err_msg})
                all_records.append({
                    "video_id": vid,
                    "split": split_name,
                    "class_label": class_label,
                    "source_dataset": row["source_dataset"],
                    "signer_id": row["signer_id"],
                    "relative_video_path": row["relative_video_path"],
                    "npz_path": "",
                    "frame_count": 0,
                    "spatial_dim": 0,
                    "landmark_shape": "",
                    "fps": 0,
                    "resolution": "",
                    "pose_detection_rate": 0.0,
                    "left_hand_detection_rate": 0.0,
                    "right_hand_detection_rate": 0.0,
                    "extraction_fps": 0.0,
                    "extraction_time_sec": 0.0,
                    "extraction_status": f"FAILED: {err_msg}"
                })

        summary_stats["splits"][split_name] = {
            "total_clips": len(df),
            "successful_clips": split_success,
            "failed_clips": split_fail,
            "total_frames_extracted": split_frames,
            "mean_pose_detection_rate": round(float(np.mean(pose_rates)), 4) if pose_rates else 0.0,
            "mean_left_hand_detection_rate": round(float(np.mean(lh_rates)), 4) if lh_rates else 0.0,
            "mean_right_hand_detection_rate": round(float(np.mean(rh_rates)), 4) if rh_rates else 0.0
        }

    total_time = time.perf_counter() - t0_global
    summary_stats["total_extraction_time_seconds"] = round(total_time, 2)

    # Save extraction index
    index_df = pd.DataFrame(all_records)
    out_csv = LANDMARKS_DIR / "extraction_index.csv"
    index_df.to_csv(out_csv, index=False)
    logger.info(f"Saved extraction index to {out_csv} ({len(index_df)} rows)")

    # Save extraction summary
    out_json = LANDMARKS_DIR / "extraction_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Saved extraction summary to {out_json}")

    return index_df, summary_stats

def generate_visualizations(index_df):
    logger.info("Generating sample landmark visualization plots...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pick representative samples
    samples = [
        {"class": "hello", "title": "Hello (Greeting gesture - Single/Dual hand wave)"},
        {"class": "thank you", "title": "Thank You (Gratitude gesture - Chin/Chest to Forward)"},
        {"class": "sorry", "title": "Sorry (Apology gesture - Circular chest movement)"},
        {"class": "stop", "title": "Stop (Halting gesture - Forward facing palm)"}
    ]

    for sample in samples:
        cls = sample["class"]
        matches = index_df[(index_df["class_label"] == cls) & (index_df["extraction_status"] == "SUCCESS")]
        if matches.empty:
            continue

        row = matches.iloc[0]
        npz_file = PROJECT_ROOT / row["npz_path"]
        data = np.load(npz_file)
        landmarks = data["landmarks"] # (T, 75, 4)
        T = landmarks.shape[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: 2D Spatial Skeleton Projection of 3 keyframes (Start, Mid, End)
        ax1.set_title(f"Keyframe Skeletons: {sample['title']}\n(Clip: {row['video_id']}, Frames: {T})", fontsize=11)
        ax1.set_xlim(-0.1, 1.1)
        ax1.set_ylim(1.1, -0.1) # Invert Y for image coordinate convention
        ax1.set_aspect("equal")
        ax1.grid(True, linestyle="--", alpha=0.5)

        colors = ["#2b5c8f", "#e67e22", "#27ae60"]
        keyframe_indices = [0, T // 2, T - 1]

        for k_idx, color in zip(keyframe_indices, colors):
            frame_lms = landmarks[k_idx] # (75, 4)
            # Body Pose (0..32)
            pose_pts = frame_lms[0:33, :2]
            valid_pose = frame_lms[0:33, 3] > 0.2
            if np.any(valid_pose):
                ax1.scatter(pose_pts[valid_pose, 0], pose_pts[valid_pose, 1], c=color, s=25, alpha=0.7, label=f"Frame {k_idx}")

            # Left Hand (33..53)
            lh_pts = frame_lms[33:54, :2]
            valid_lh = frame_lms[33:54, 3] > 0.1
            if np.any(valid_lh):
                ax1.scatter(lh_pts[valid_lh, 0], lh_pts[valid_lh, 1], c="purple", s=15, alpha=0.8)

            # Right Hand (54..74)
            rh_pts = frame_lms[54:75, :2]
            valid_rh = frame_lms[54:75, 3] > 0.1
            if np.any(valid_rh):
                ax1.scatter(rh_pts[valid_rh, 0], rh_pts[valid_rh, 1], c="crimson", s=15, alpha=0.8)

        ax1.legend(loc="lower right")

        # Subplot 2: Hand Trajectories over time (X-Y paths of wrists and index tips)
        ax2.set_title(f"Hand Motion Trajectories (T={T} frames)\nWrist (#0) and Index Tip (#8)", fontsize=11)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(1.1, -0.1)
        ax2.set_aspect("equal")
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Left wrist #33, Left index #41
        lw = landmarks[:, 33, :2]
        li = landmarks[:, 41, :2]
        # Right wrist #54, Right index #62
        rw = landmarks[:, 54, :2]
        ri = landmarks[:, 62, :2]

        # Valid masks
        mask_lh = (landmarks[:, 33, 3] > 0.1) & (landmarks[:, 41, 3] > 0.1)
        mask_rh = (landmarks[:, 54, 3] > 0.1) & (landmarks[:, 62, 3] > 0.1)

        if np.any(mask_lh):
            ax2.plot(lw[mask_lh, 0], lw[mask_lh, 1], "b--", alpha=0.6, label="Left Wrist Path")
            ax2.plot(li[mask_lh, 0], li[mask_lh, 1], "b-", alpha=0.8, label="Left Index Path")
        if np.any(mask_rh):
            ax2.plot(rw[mask_rh, 0], rw[mask_rh, 1], "r--", alpha=0.6, label="Right Wrist Path")
            ax2.plot(ri[mask_rh, 0], ri[mask_rh, 1], "r-", alpha=0.8, label="Right Index Path")

        ax2.legend(loc="lower right")

        out_fig = FIGURES_DIR / f"landmark_vis_{cls.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(out_fig, dpi=150)
        plt.close()
        logger.info(f"Saved visualization to {out_fig}")

if __name__ == "__main__":
    idx_df, stats = extract_all_splits()
    generate_visualizations(idx_df)
