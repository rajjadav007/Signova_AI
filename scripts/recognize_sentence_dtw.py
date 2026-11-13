"""
Dynamic Time Warping + K-Nearest Neighbors (DTW + KNN) Inference CLI.

Script: scripts/recognize_sentence_dtw.py
Task: 2.2A Deliverable - Standalone inference script for DTW+KNN sentence classification.

Usage:
  python scripts/recognize_sentence_dtw.py --sample-index 0
  python scripts/recognize_sentence_dtw.py --landmarks path/to/sample.npz
  python scripts/recognize_sentence_dtw.py --video path/to/video.mp4
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

import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier
from ai.continuous.temporal_features import extract_temporal_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("signova.recognize_sentence_dtw")


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


def preprocess_sequence(
    raw_features: np.ndarray,
    preprocessors_path: Optional[Path] = None,
    stride: int = 2,
    preprocessors: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Project raw 678-dimensional temporal features to 48-dim Frame PCA with temporal stride.
    Supports both legacy global PCA and improved group-partitioned PCA preprocessors.
    """
    if preprocessors is None:
        if preprocessors_path is None or not preprocessors_path.exists():
            # Check improved first, fallback to legacy
            improved_path = (
                PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "preprocessors_improved.pkl"
            )
            if improved_path.exists():
                preprocessors_path = improved_path
            else:
                preprocessors_path = (
                    PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "preprocessors.pkl"
                )

        with open(preprocessors_path, "rb") as f:
            preprocessors = pickle.load(f)

    # Temporal stride downsampling
    sub_seq = raw_features[::stride]  # (T//stride, 678)

    # Check for group-partitioned improved preprocessors
    if "pca_spatial" in preprocessors:
        spatial_std = preprocessors["spatial_std"]
        vel_std = preprocessors["vel_std"]
        acc_std = preprocessors["acc_std"]
        acc_threshold_3sigma = preprocessors["acc_threshold_3sigma"]
        pca_spatial = preprocessors["pca_spatial"]
        pca_vel = preprocessors["pca_vel"]
        pca_acc = preprocessors["pca_acc"]

        s = sub_seq[:, 0:225] / spatial_std
        v = sub_seq[:, 225:450] / vel_std
        a_clip = np.clip(sub_seq[:, 450:675], -acc_threshold_3sigma, acc_threshold_3sigma)
        a = a_clip / acc_std

        s_pca = pca_spatial.transform(s)
        v_pca = pca_vel.transform(v)
        a_pca = pca_acc.transform(a)
        return np.concatenate([s_pca, v_pca, a_pca], axis=-1).astype(np.float32)

    # Legacy fallback
    frame_scaler = preprocessors["frame_scaler"]
    frame_pca_48 = preprocessors["frame_pca_48"]
    sub_seq_scaled = frame_scaler.transform(sub_seq)
    sub_seq_pca = frame_pca_48.transform(sub_seq_scaled).astype(np.float32)
    return sub_seq_pca


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
        description="Dynamic Time Warping + KNN Inference for ISL Continuous Sentences."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to input sign language video (.mp4/.avi)")
    group.add_argument("--landmarks", type=str, help="Path to pre-extracted .npz landmarks or features")
    group.add_argument("--sample-index", type=int, help="Index of sample in validation cache (0..84)")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "isl_csltr_continuous" / "best_dtw_knn_model.pkl"),
        help="Path to trained DTWKNNClassifier pickle",
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
    parser.add_argument("--k", type=int, default=None, help="Override K nearest neighbors")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes for parallel DTW")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t_total_0 = time.time()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    # 1. Load model and vocabulary
    logger.info(f"Loading DTW+KNN model from {ckpt_path}...")
    clf = DTWKNNClassifier.load(ckpt_path)
    if args.k is not None:
        clf.n_neighbors = args.k

    classes_map = load_sentence_classes(Path(args.classes))

    # 2. Obtain query sequence (48-dim Frame PCA)
    t_feat_0 = time.time()
    ground_truth_id = None
    query_video_id = "custom_input"

    if args.sample_index is not None:
        cache_path = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "dtw_dataset.pkl"
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        val_data = data["validation"]
        idx = args.sample_index
        if idx < 0 or idx >= len(val_data["seqs"]):
            logger.error(f"Sample index {idx} out of range [0, {len(val_data['seqs'])-1}]")
            sys.exit(1)
        query_pca = val_data["seqs"][idx]["features_pca"]
        ground_truth_id = int(val_data["labels"][idx])
        query_video_id = val_data["video_ids"][idx]
        feat_time = time.time() - t_feat_0
    elif args.landmarks:
        raw_feat = extract_features_from_npz(args.landmarks)
        query_pca = preprocess_sequence(raw_feat, Path(args.preprocessors))
        feat_time = time.time() - t_feat_0
    elif args.video:
        raw_feat = extract_features_from_video(args.video)
        query_pca = preprocess_sequence(raw_feat, Path(args.preprocessors))
        feat_time = time.time() - t_feat_0

    logger.info(f"Query sequence: {query_video_id} | Shape: {query_pca.shape} | Feature extraction: {feat_time*1000:.1f}ms")

    # 3. Perform DTW+KNN inference
    t_dtw_0 = time.time()
    dist_matrix = clf.compute_distance_matrix([query_pca], max_workers=args.workers)
    preds, probs = clf.predict_from_distances(dist_matrix, k=args.k)
    dtw_time = time.time() - t_dtw_0
    total_time = time.time() - t_total_0

    # 4. Format Top-K predictions
    top_indices = np.argsort(probs[0])[::-1][:args.top_k]

    print("\n" + "=" * 70)
    print(f"{'DTW + KNN ISL SENTENCE RECOGNITION RESULT':^70}")
    print("=" * 70)
    print(f"Input:        {query_video_id}")
    print(f"Sequence:     {query_pca.shape[0]} frames (48-dim Frame PCA)")
    print(f"Algorithm:    DTW ({clf.metric}, Sakoe-Chiba window={clf.window}) + KNN (k={clf.n_neighbors})")
    print(f"Exemplars:    {len(clf.train_seqs)} training sequences across 101 classes")
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
        conf = probs[0, cid]
        is_hit = " [CORRECT]" if (ground_truth_id is not None and cid == ground_truth_id) else ""
        print(f"#{rank:<4} | {cid:<6d} | {conf*100:6.2f}%      | {sentence_text:<42}{is_hit}")

    print("-" * 70)
    print(f"Timing:       Feature Ext: {feat_time*1000:.1f}ms | DTW Match: {dtw_time*1000:.1f}ms | Total: {total_time*1000:.1f}ms")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
