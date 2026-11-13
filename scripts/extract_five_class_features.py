"""
Signova AI - Multi-Frame MediaPipe Feature Extraction for Five-Class ISL Sentences.
Phases 3, 4, and 5:
- Extracts 75 MediaPipe landmarks (33 pose, 21 LH, 21 RH) per frame.
- Applies torso-centered coordinate normalization.
- Preserves chronological frame order.
- Saves processed sequences and metadata to data/processed/five_class_sequence/.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_five_class")

DATASET_ROOT = Path(r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "five_class_sequence"

CLASSES = [
    {"class_id": 0, "name": "comb your hair", "folder": "comb your hair"},
    {"class_id": 1, "name": "help me", "folder": "help me"},
    {"class_id": 2, "name": "i am fine. thank you sir.", "folder": "i am fine. thank you sir"},
    {"class_id": 3, "name": "i need water", "folder": "i need water"},
    {"class_id": 4, "name": "who are you", "folder": "who are you"},
]

SPLITS = {
    "train": {"1", "3", "4", "5", "6"},
    "validation": {"7"},
    "test": {"2"}
}


def natural_sort_key(p: Path):
    """Sort filenames numerically, e.g. (1).jpg, (2).jpg, (10).jpg."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', p.name)]


def normalize_torso_landmarks(keypoints_75x4: np.ndarray) -> np.ndarray:
    """
    Torso-centered normalization for 75 landmarks:
    - Origin at shoulder midpoint (landmarks 11 and 12).
    - Scale normalized by shoulder distance.
    - Missing components (zero visibility/confidence) remain 0.0.
    """
    norm_kps = keypoints_75x4.copy()

    # Pose landmarks 11: left_shoulder, 12: right_shoulder
    ls = norm_kps[11, :3]
    rs = norm_kps[12, :3]
    ls_vis = norm_kps[11, 3]
    rs_vis = norm_kps[12, 3]

    if ls_vis > 0.3 and rs_vis > 0.3:
        midpoint = (ls + rs) / 2.0
        shoulder_dist = float(np.linalg.norm(ls - rs))
        scale = shoulder_dist if shoulder_dist > 1e-3 else 1.0
    else:
        # Fallback to nose (idx 0) or default center
        nose = norm_kps[0, :3]
        nose_vis = norm_kps[0, 3]
        midpoint = nose if nose_vis > 0.3 else np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 1.0

    # Normalize present landmarks
    for i in range(75):
        if norm_kps[i, 3] > 0.0:  # Present landmark
            norm_kps[i, :3] = (norm_kps[i, :3] - midpoint) / scale
        else:
            norm_kps[i, :3] = 0.0
            norm_kps[i, 3] = 0.0

    return norm_kps


def run_extraction():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Also mirror to root data directory if different
    root_out_dir = Path(r"D:\Projects\data\processed\five_class_sequence")
    root_out_dir.mkdir(parents=True, exist_ok=True)

    extractor = ModernMediaPipeExtractor()

    manifest_records = []
    split_dataset = {"train": [], "validation": [], "test": []}

    total_frames_processed = 0
    t0 = time.time()

    for c in CLASSES:
        cdir = DATASET_ROOT / c["folder"]
        for signer_str in ["1", "2", "3", "4", "5", "6", "7"]:
            sdir = cdir / signer_str
            if not sdir.exists():
                logger.warning(f"Missing signer folder: {sdir}")
                continue

            all_files = sorted(list(sdir.glob("*")), key=natural_sort_key)
            frame_paths = [f for f in all_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

            if not frame_paths:
                logger.warning(f"No valid frames in {sdir}")
                continue

            split_name = "train" if signer_str in SPLITS["train"] else ("validation" if signer_str in SPLITS["validation"] else "test")
            seq_id = f"s{c['class_id']:03d}_signer{signer_str}_{re.sub(r'[^a-zA-Z0-9]', '_', c['name'])}"

            sequence_features = []
            frame_infos = []

            for fp in frame_paths:
                img_bgr = cv2.imread(str(fp))
                if img_bgr is None:
                    logger.error(f"Failed to read image: {fp}")
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                raw_75x4, info = extractor.extract_isign_frame(img_rgb)
                norm_75x4 = normalize_torso_landmarks(raw_75x4)

                sequence_features.append(norm_75x4.reshape(-1))  # 300 floats
                frame_infos.append(info)
                total_frames_processed += 1

            seq_array = np.array(sequence_features, dtype=np.float32)  # (T, 300)

            # Save individual .npz
            npz_filename = f"{seq_id}.npz"
            save_path = OUTPUT_DIR / npz_filename
            np.savez_compressed(
                save_path,
                features=seq_array,
                class_id=c["class_id"],
                sentence=c["name"],
                signer_id=int(signer_str),
                split=split_name,
                frame_count=len(frame_paths),
                raw_folder=str(sdir)
            )

            # Also mirror
            np.savez_compressed(
                root_out_dir / npz_filename,
                features=seq_array,
                class_id=c["class_id"],
                sentence=c["name"],
                signer_id=int(signer_str),
                split=split_name,
                frame_count=len(frame_paths),
                raw_folder=str(sdir)
            )

            record = {
                "sequence_id": seq_id,
                "class_id": c["class_id"],
                "sentence": c["name"],
                "signer_id": int(signer_str),
                "split": split_name,
                "frame_count": len(frame_paths),
                "feature_shape": list(seq_array.shape),
                "npz_file": npz_filename,
                "raw_folder": str(sdir)
            }
            manifest_records.append(record)
            split_dataset[split_name].append({
                "features": seq_array,
                "class_id": c["class_id"],
                "sentence": c["name"],
                "signer_id": int(signer_str),
                "sequence_id": seq_id
            })

            logger.info(f"Processed {seq_id}: {seq_array.shape} [{split_name}]")

    elapsed = time.time() - t0
    logger.info(f"Completed feature extraction for {len(manifest_records)} sequences ({total_frames_processed} frames) in {elapsed:.1f}s ({total_frames_processed/elapsed:.1f} fps).")

    # Save preprocessing config
    prep_config = {
        "num_classes": 5,
        "classes": CLASSES,
        "landmark_count": 75,
        "landmark_channels": 4,
        "feature_dim": 300,
        "normalization": "torso_centered",
        "origin_landmarks": [11, 12],
        "scale_reference": "shoulder_distance",
        "missing_hand_strategy": "zero_fill",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(OUTPUT_DIR / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(prep_config, f, indent=2)
    with open(root_out_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(prep_config, f, indent=2)

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    with open(root_out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)

    # Save aggregated dataset .npz for fast training loading
    dataset_agg_file = OUTPUT_DIR / "five_class_dataset.npz"
    np.savez_compressed(
        dataset_agg_file,
        train_features=np.array([d["features"] for d in split_dataset["train"]], dtype=object),
        train_labels=np.array([d["class_id"] for d in split_dataset["train"]], dtype=np.int64),
        train_signers=np.array([d["signer_id"] for d in split_dataset["train"]], dtype=np.int64),
        train_seq_ids=np.array([d["sequence_id"] for d in split_dataset["train"]]),
        val_features=np.array([d["features"] for d in split_dataset["validation"]], dtype=object),
        val_labels=np.array([d["class_id"] for d in split_dataset["validation"]], dtype=np.int64),
        val_signers=np.array([d["signer_id"] for d in split_dataset["validation"]], dtype=np.int64),
        val_seq_ids=np.array([d["sequence_id"] for d in split_dataset["validation"]]),
        test_features=np.array([d["features"] for d in split_dataset["test"]], dtype=object),
        test_labels=np.array([d["class_id"] for d in split_dataset["test"]], dtype=np.int64),
        test_signers=np.array([d["signer_id"] for d in split_dataset["test"]], dtype=np.int64),
        test_seq_ids=np.array([d["sequence_id"] for d in split_dataset["test"]]),
    )
    # Mirror aggregated dataset
    np.savez_compressed(root_out_dir / "five_class_dataset.npz", **np.load(dataset_agg_file, allow_pickle=True))

    logger.info(f"Aggregated dataset saved to {dataset_agg_file}")
    logger.info(f"Train sequences: {len(split_dataset['train'])}, Val sequences: {len(split_dataset['validation'])}, Test sequences: {len(split_dataset['test'])}")


if __name__ == "__main__":
    run_extraction()
