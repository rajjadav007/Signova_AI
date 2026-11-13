"""
Extracts landmarks for the 20-class dataset splits.
Reuses existing 10-class landmarks and extracts only missing/new clips.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signova.extract_20class")

LANDMARKS_DIR = PROJECT_ROOT / "datasets" / "landmarks"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"

def extract_landmarks():
    extractor = ModernMediaPipeExtractor()
    splits = ["train", "validation", "test"]

    total_attempted = 0
    total_skipped = 0
    total_extracted = 0
    total_failed = 0

    t0 = time.perf_counter()

    for split_name in splits:
        csv_path = SPLITS_DIR / f"{split_name}_20class.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        out_split_dir = LANDMARKS_DIR / split_name
        out_split_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Split '{split_name}': {len(df)} total clips.")

        for _, row in df.iterrows():
            total_attempted += 1
            vid = row["video_id"]
            out_npz = out_split_dir / f"{vid}.npz"

            if out_npz.exists():
                total_skipped += 1
                continue

            vpath = Path(row["absolute_video_path"])
            if not vpath.exists():
                logger.error(f"Video missing: {vpath}")
                total_failed += 1
                continue

            try:
                raw_spatial, meta = extractor.extract_isign_video(vpath)
                n_frames = meta["read_frames"]

                landmarks = raw_spatial.reshape(-1, 75, 4).astype(np.float32)
                spatial_features = raw_spatial.astype(np.float32)

                fps = meta["fps"] if meta["fps"] > 0 else 30.0
                timestamps_ms = np.array([int(round(i * 1000.0 / fps)) for i in range(n_frames)], dtype=np.int64)
                frame_indices = np.arange(n_frames, dtype=np.int32)

                det_pose = np.any(raw_spatial[:, 0:132] != 0, axis=1)
                det_lh = np.any(raw_spatial[:, 132:216] != 0, axis=1)
                det_rh = np.any(raw_spatial[:, 216:300] != 0, axis=1)
                detection_mask = np.column_stack([det_pose, det_lh, det_rh]).astype(bool)

                np.savez_compressed(
                    out_npz,
                    landmarks=landmarks,
                    spatial_features=spatial_features,
                    timestamps_ms=timestamps_ms,
                    frame_indices=frame_indices,
                    detection_mask=detection_mask,
                )
                total_extracted += 1
                if total_extracted % 25 == 0:
                    logger.info(f"Extracted {total_extracted} new clips... (elapsed: {time.perf_counter() - t0:.1f}s)")
            except Exception as e:
                logger.error(f"Failed extracting {vid} ({vpath}): {e}")
                total_failed += 1

    elapsed = time.perf_counter() - t0
    logger.info(f"Extraction complete in {elapsed:.1f}s. Total: {total_attempted}, Reused: {total_skipped}, Extracted: {total_extracted}, Failed: {total_failed}")

if __name__ == "__main__":
    extract_landmarks()
