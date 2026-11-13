"""
Threaded MediaPipe Landmark Extraction for 20-class dataset splits.
Uses ThreadPoolExecutor(max_workers=2) to maximize CPU without MemoryError.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent MediaPipe C++ LiteRT double-free on Windows
try:
    from mediapipe.tasks.python import vision
    vision.HandLandmarker.__del__ = lambda self: None
    vision.PoseLandmarker.__del__ = lambda self: None
except Exception:
    pass

from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

LANDMARKS_DIR = PROJECT_ROOT / "datasets" / "landmarks"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"

def process_clip(extractor: ModernMediaPipeExtractor, task: dict) -> dict:
    vid = task["video_id"]
    split_name = task["split"]
    vpath = Path(task["absolute_video_path"])
    out_npz = LANDMARKS_DIR / split_name / f"{vid}.npz"

    if out_npz.exists():
        return {"video_id": vid, "status": "skipped"}

    if not vpath.exists():
        return {"video_id": vid, "status": "missing_video"}

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
        return {"video_id": vid, "status": "extracted", "frames": n_frames}
    except Exception as e:
        return {"video_id": vid, "status": "failed", "error": str(e)}

def main():
    splits = ["train", "validation", "test"]
    tasks = []

    for s in splits:
        csv_file = SPLITS_DIR / f"{s}_20class.csv"
        df = pd.read_csv(csv_file)
        (LANDMARKS_DIR / s).mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            vid = row["video_id"]
            npz_p = LANDMARKS_DIR / s / f"{vid}.npz"
            if not npz_p.exists():
                t = dict(row)
                t["split"] = s
                tasks.append(t)

    print(f"Total remaining clips to extract: {len(tasks)}")
    if not tasks:
        print("All clips already extracted!")
        return

    # Two independent extractor instances for 2 worker threads
    ext1 = ModernMediaPipeExtractor()
    ext2 = ModernMediaPipeExtractor()
    extractors = [ext1, ext2]

    t0 = time.perf_counter()
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_clip, extractors[i % 2], t): t["video_id"] for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if res["status"] == "failed":
                failed += 1
                print(f"Failed {res['video_id']}: {res.get('error')}")

            if completed % 20 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - t0
                rate = completed / max(elapsed, 1e-4)
                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%) in {elapsed:.1f}s [{rate:.2f} clips/s]")

    print(f"\nAll done! Processed {completed} clips ({failed} failed) in {time.perf_counter() - t0:.1f}s.")

if __name__ == "__main__":
    main()
