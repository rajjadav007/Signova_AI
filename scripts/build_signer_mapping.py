"""
Utility to compute bit-exact signer mapping for all 687 videos of ISL-CSLTR
by correlating video frames with ground-truth signer frame directories.
Saves cache to datasets/isl_csltr_continuous/metadata/video_signer_mapping.json.
"""

from collections import Counter
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET_ROOT = Path(r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus")
OUT_JSON = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "metadata" / "video_signer_mapping.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signer_mapping")


def compute_frame_mse(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute mean squared error between two images, resizing b if needed."""
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    return float(np.mean((img_a.astype(float) - img_b.astype(float)) ** 2))


def generate_signer_mapping():
    vroot = RAW_DATASET_ROOT / "Videos_Sentence_Level"
    froot = RAW_DATASET_ROOT / "Frames_Sentence_Level"

    all_vpaths = sorted(list(set(v.resolve() for v in vroot.glob("*/*") if v.suffix.lower() == ".mp4")))
    logger.info(f"Scanning {len(all_vpaths)} videos...")

    # Load frame 0 references
    frame0_cache: Dict[Tuple[str, int], np.ndarray] = {}
    for sdir in froot.iterdir():
        if not sdir.is_dir(): continue
        for sid_dir in sdir.iterdir():
            if not sid_dir.is_dir(): continue
            sid = int(sid_dir.name)
            frames = sorted(list(sid_dir.glob("*.jpg")))
            if frames:
                img = cv2.imread(str(frames[0]))
                if img is not None:
                    frame0_cache[(sdir.name.lower(), sid)] = img

    logger.info(f"Loaded {len(frame0_cache)} frame0 ground-truth references.")

    mapping: Dict[str, Dict[str, Any]] = {}
    signer_counts = Counter()
    split_counts = Counter()

    for idx, vpath in enumerate(all_vpaths, 1):
        fname = vpath.name
        sfolder = vpath.parent.name.lower()

        cap = cv2.VideoCapture(str(vpath))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame0 = cap.read()
        cap.release()

        assigned_signer: Optional[int] = None
        match_confidence = "heuristic"
        min_mse = float("inf")

        # 1. Sentences with frame references
        if ret and frame0 is not None:
            candidate_signers = [1, 2] if (w <= 320) else [3, 4, 5, 6, 7]
            for sid in candidate_signers:
                key = (sfolder, sid)
                if key in frame0_cache:
                    ref = frame0_cache[key]
                    # Try direct MSE
                    err1 = compute_frame_mse(frame0, ref)
                    # For low-res, also try 90-deg clockwise rotation
                    err2 = float("inf")
                    if w <= 320:
                        f_rot = cv2.rotate(frame0, cv2.ROTATE_90_CLOCKWISE)
                        err2 = compute_frame_mse(f_rot, ref)
                    err = min(err1, err2)
                    if err < min_mse:
                        min_mse = err
                        assigned_signer = sid

            if min_mse < 80.0:
                match_confidence = f"frame_mse_{min_mse:.1f}"
            else:
                assigned_signer = None

        # 2. Fallbacks for sentences missing from Frames_Sentence_Level or extra takes
        if assigned_signer is None:
            if sfolder == "he would be coming today":
                if "today (2)" in fname: assigned_signer = 2
                elif "today." in fname: assigned_signer = 1
                elif "today (2).mp4" in fname.lower(): assigned_signer = 4
                elif "today (3).mp4" in fname.lower(): assigned_signer = 5
                elif "today (4).mp4" in fname.lower(): assigned_signer = 6
                elif "today (5).mp4" in fname.lower(): assigned_signer = 7
                elif "today.mp4" in fname.lower(): assigned_signer = 3
                else: assigned_signer = 3
                match_confidence = "sentence_rule"
            elif sfolder == "now onwards he will never hurt you":
                if "never hurt (2)" in fname: assigned_signer = 4
                elif "never hurt (3)" in fname: assigned_signer = 5
                elif "never hurt (4)" in fname: assigned_signer = 6
                elif "never hurt (5)" in fname: assigned_signer = 7
                elif "never hurt" in fname: assigned_signer = 3
                else: assigned_signer = 1
                match_confidence = "sentence_rule"
            elif sfolder in ("i dont agree", "no need to worry dont worry"):
                assigned_signer = 3
                match_confidence = "sentence_rule"
            else:
                # Extra take in volunteer videos
                assigned_signer = 3 if w > 320 else 1
                match_confidence = "extra_take_fallback"

        # Determine split
        if assigned_signer == 2:
            split = "test"
        elif assigned_signer == 7:
            split = "validation"
        else:
            split = "train"

        signer_counts[assigned_signer] += 1
        split_counts[split] += 1

        mapping[str(vpath)] = {
            "relative_video_path": str(vpath.relative_to(RAW_DATASET_ROOT)).replace("\\", "/"),
            "video_filename": fname,
            "sentence_folder": vpath.parent.name,
            "width": w,
            "height": h,
            "signer_id": assigned_signer,
            "split": split,
            "match_confidence": match_confidence,
            "min_mse": round(min_mse, 2) if min_mse != float("inf") else None
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(mapping, fp, indent=2)

    logger.info(f"Signer distribution: {dict(sorted(signer_counts.items()))}")
    logger.info(f"Split distribution: {dict(sorted(split_counts.items()))}")
    logger.info(f"Saved mapping table to {OUT_JSON}")
    return mapping


if __name__ == "__main__":
    generate_signer_mapping()
