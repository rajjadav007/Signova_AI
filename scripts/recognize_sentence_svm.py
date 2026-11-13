"""
Support Vector Machine (SVM RBF) ISL Sentence Inference CLI.

Script: scripts/recognize_sentence_svm.py
Task: 2.2B Deliverable - Standalone inference script for SVM sentence classification.

Usage:
  python scripts/recognize_sentence_svm.py --sample-index 0
  python scripts/recognize_sentence_svm.py --landmarks path/to/sample.npz
  python scripts/recognize_sentence_svm.py --video path/to/video.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from scipy.special import softmax

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.temporal_features import extract_temporal_features
from scripts.extract_ml_features import extract_temporal_pyramid

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("signova.recognize_sentence_svm")


def load_sentence_classes(
    classes_json_path: Optional[Path] = None,
) -> Dict[int, Dict[str, Any]]:
    """Load 101 sentence classes mapping."""
    if classes_json_path is None or not classes_json_path.exists():
        classes_json_path = (
            PROJECT_ROOT / "models" / "isl_csltr_continuous" / "sentence_classes.json"
        )

    mapping: Dict[int, Dict[str, Any]] = {}
    if classes_json_path.exists():
        with open(classes_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for c in data.get("classes", []):
            sid = int(c["sentence_id"])
            mapping[sid] = {
                "sentence_id": sid,
                "sentence": c.get("sentence", f"Sentence {sid}"),
                "isl_gloss": c.get("isl_gloss", ""),
                "tokens": c.get("tokens", []),
            }
    return mapping


def extract_features_from_npz(npz_path: Union[str, Path], fps: float = 25.0) -> np.ndarray:
    """Load landmarks or features from .npz file."""
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ file not found: {p}")

    data = np.load(p)
    if "features" in data:
        return data["features"].astype(np.float32)

    landmarks = data["landmarks"].astype(np.float32)
    detection_mask = data["detection_mask"] if "detection_mask" in data else None
    timestamps_ms = data["timestamps_ms"] if "timestamps_ms" in data else None

    features = extract_temporal_features(
        landmarks=landmarks,
        detection_mask=detection_mask,
        timestamps_ms=timestamps_ms,
        fps=fps,
    )
    return features


def extract_features_from_video(
    video_path: Union[str, Path],
    max_frames: Optional[int] = 300,
) -> np.ndarray:
    """Extract MediaPipe landmarks and compute 678 temporal features."""
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

    extractor = ModernMediaPipeExtractor()
    raw_spatial, meta = extractor.extract_isign_video(str(vpath), max_frames=max_frames)
    detection_mask = meta.get("detection_mask", None)
    timestamps_ms = meta.get("timestamps_ms", None)

    features = extract_temporal_features(
        landmarks=raw_spatial,
        detection_mask=detection_mask,
        timestamps_ms=timestamps_ms,
        fps=25.0,
    )
    return features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SVM (RBF Kernel) Inference for ISL Continuous Sentences."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to input sign language video (.mp4/.avi)")
    group.add_argument("--landmarks", type=str, help="Path to pre-extracted .npz landmarks or features")
    group.add_argument("--sample-index", type=int, help="Index of sample in validation cache (0..84)")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "isl_csltr_continuous" / "best_svm_checkpoint.joblib"),
        help="Path to trained SVM model joblib checkpoint",
    )
    parser.add_argument(
        "--preprocessors",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "preprocessors.pkl"),
        help="Path to preprocessors.pkl",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=str(PROJECT_ROOT / "models" / "isl_csltr_continuous" / "sentence_classes.json"),
        help="Path to sentence_classes.json",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t_total_0 = time.time()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    # 1. Load model checkpoint and configuration
    logger.info(f"Loading SVM checkpoint from {ckpt_path}...")
    ckpt = joblib.load(ckpt_path)
    model = ckpt["model"]
    feat_set = ckpt.get("feature_set", "PCA_128")
    cfg_name = ckpt.get("best_config", "PCA_128_C10.0_gscale_cwNone")

    classes_map = load_sentence_classes(Path(args.classes))

    # 2. Load preprocessors
    prep_path = Path(args.preprocessors)
    with open(prep_path, "rb") as f:
        preprocessors = pickle.load(f)
    scaler = preprocessors["pyramid_scaler"]
    pca128 = preprocessors.get("pyramid_pca128")
    pca64 = preprocessors.get("pyramid_pca64")

    # 3. Obtain query representation
    t_feat_0 = time.time()
    ground_truth_id = None
    query_video_id = "custom_input"

    if args.sample_index is not None:
        cache_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache"
        val_npz = np.load(cache_dir / "pyramid_validation.npz")
        idx = args.sample_index
        if idx < 0 or idx >= len(val_npz["y"]):
            logger.error(f"Sample index {idx} out of range [0, {len(val_npz['y'])-1}]")
            sys.exit(1)

        ground_truth_id = int(val_npz["y"][idx])
        query_video_id = str(val_npz["video_ids"][idx])

        if feat_set == "PCA_128":
            query_vec = val_npz["X_pca128"][idx:idx+1]
        elif feat_set == "PCA_64":
            query_vec = val_npz["X_pca64"][idx:idx+1]
        else:
            query_vec = val_npz["X_scaled"][idx:idx+1]
        feat_time = time.time() - t_feat_0
    else:
        if args.landmarks:
            raw_seq = extract_features_from_npz(args.landmarks)
        else:
            raw_seq = extract_features_from_video(args.video)

        # Extract 7-segment temporal pyramid
        pyr_raw = extract_temporal_pyramid(raw_seq).reshape(1, -1)
        pyr_scaled = scaler.transform(pyr_raw)

        if feat_set == "PCA_128":
            query_vec = pca128.transform(pyr_scaled)
        elif feat_set == "PCA_64":
            query_vec = pca64.transform(pyr_scaled)
        else:
            query_vec = pyr_scaled
        feat_time = time.time() - t_feat_0

    logger.info(f"Query representation shape: {query_vec.shape} (Feature pipeline: {feat_time*1000:.1f}ms)")

    # 4. Perform SVM inference
    t_infer_0 = time.time()
    dec_scores = model.decision_function(query_vec)  # (1, C_train)
    probs_seen = softmax(dec_scores, axis=1)

    probs = np.zeros(101, dtype=np.float32)
    for col_idx, cls_id in enumerate(model.classes_):
        if cls_id < 101:
            probs[cls_id] = probs_seen[0, col_idx]

    infer_time = time.time() - t_infer_0
    total_time = time.time() - t_total_0

    # 5. Format Top-K predictions
    top_indices = np.argsort(probs)[::-1][:args.top_k]

    print("\n" + "=" * 70)
    print(f"{'SVM (RBF KERNEL) ISL SENTENCE RECOGNITION RESULT':^70}")
    print("=" * 70)
    print(f"Input:        {query_video_id}")
    print(f"Feature Set:  7-segment Temporal Pyramid -> {feat_set} ({query_vec.shape[1]} dims)")
    print(f"Model:        SVC(kernel=rbf, C={model.C}, gamma={model.gamma})")
    print(f"Classes:      101 known ISL sentence classes")
    if ground_truth_id is not None:
        gt_info = classes_map.get(ground_truth_id, {})
        print(f"Ground Truth: Class {ground_truth_id:3d} -> \"{gt_info.get('sentence', 'Unknown')}\"")
        print(f"              ISL Gloss: {gt_info.get('isl_gloss', '')}")
    print("-" * 70)
    print(f"{'Rank':<5} | {'Class':<6} | {'Confidence':<12} | {'Predicted Sentence':<42}")
    print("-" * 70)

    for rank, cid in enumerate(top_indices, 1):
        c_info = classes_map.get(cid, {})
        sentence_text = c_info.get("sentence", f"Sentence {cid}")
        conf = probs[cid]
        is_hit = " [CORRECT]" if (ground_truth_id is not None and cid == ground_truth_id) else ""
        print(f"#{rank:<4} | {cid:<6d} | {conf*100:6.2f}%      | {sentence_text:<42}{is_hit}")

    print("-" * 70)
    print(f"Timing:       Feature Pipeline: {feat_time*1000:.1f}ms | SVM Forward: {infer_time*1000:.2f}ms | Total: {total_time*1000:.1f}ms")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
