"""
Signova AI: Hackathon Demo Recognition CLI.

5-Class Continuous ISL Sentence Recognition Prototype using MediaPipe + DTW (1-NN).
Script: scripts/demo_recognize_sentence.py

Usage:
  python scripts/demo_recognize_sentence.py --video path/to/video.mp4
  python scripts/demo_recognize_sentence.py --landmarks path/to/sample.npz
"""

from __future__ import annotations

import argparse
import json
import logging
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

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier, compute_dtw_distance
from scripts.recognize_sentence_dtw import (
    extract_features_from_npz,
    extract_features_from_video,
    preprocess_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signova.demo_recognize")

DEFAULT_DEMO_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "hackathon_demo"


def load_demo_artifacts(demo_dir: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[np.ndarray], np.ndarray]:
    """Load 5-class demo metadata, references and sequences."""
    if demo_dir is None:
        demo_dir = DEFAULT_DEMO_DIR

    classes_path = demo_dir / "demo_classes.json"
    dataset_path = demo_dir / "reference_dataset.pkl"

    if not classes_path.exists() or not dataset_path.exists():
        raise FileNotFoundError(
            f"Demo artifacts missing in {demo_dir}. Run scripts/build_hackathon_demo_references.py first."
        )

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)

    return classes, ds["references"], ds["sequences"], ds["labels"]


def recognize_sequence(
    proc_seq: np.ndarray,
    ref_seqs: List[np.ndarray],
    ref_labels: np.ndarray,
    classes: List[Dict[str, Any]],
    distance_threshold: float = 0.40,
    min_margin: float = 0.015,
    metric: str = "cosine",
    window: int = 10,
) -> Dict[str, Any]:
    """
    Match query sequence against the 10 reference exemplars using 1-NN DTW.
    Ranks the 5 classes and evaluates rejection threshold for unknown input.
    """
    t_start = time.perf_counter()

    # Pairwise DTW to 10 references
    ref_distances = [
        compute_dtw_distance(proc_seq, ref, metric=metric, window=window)
        for ref in ref_seqs
    ]

    # Aggregate minimum distance per class (1-NN nearest exemplar)
    num_classes = len(classes)
    class_rankings = []
    for c_id in range(num_classes):
        c_ref_dists = [ref_distances[i] for i in range(len(ref_distances)) if ref_labels[i] == c_id]
        min_c_dist = float(min(c_ref_dists))
        sim = float(max(0.0, 1.0 - min_c_dist))
        c_info = classes[c_id]
        class_rankings.append({
            "class_id": c_id,
            "original_sentence_id": c_info["original_sentence_id"],
            "sentence": c_info["sentence"],
            "isl_gloss": c_info["isl_gloss"],
            "dtw_distance": round(min_c_dist, 4),
            "similarity": round(sim, 4),
        })

    # Sort classes by distance ascending (similarity descending)
    class_rankings.sort(key=lambda x: x["dtw_distance"])

    top1 = class_rankings[0]
    top2 = class_rankings[1]
    margin = round(top2["dtw_distance"] - top1["dtw_distance"], 4)

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Unknown input rejection check
    # Rejected if minimum DTW distance exceeds threshold OR margin between top 2 is negligible
    is_rejected = (top1["dtw_distance"] > distance_threshold) or (margin < min_margin)

    if is_rejected:
        prediction_status = "Sentence not recognized."
        predicted_sentence = None
    else:
        prediction_status = "Recognized"
        predicted_sentence = top1["sentence"]

    return {
        "status": prediction_status,
        "is_recognized": not is_rejected,
        "predicted_sentence": predicted_sentence,
        "predicted_class_id": None if is_rejected else top1["class_id"],
        "top_similarity": top1["similarity"],
        "top_distance": top1["dtw_distance"],
        "margin_to_second": margin,
        "rankings": class_rankings,
        "latency_ms": round(latency_ms, 2),
        "note": "Similarity score is uncalibrated DTW alignment similarity [0.0..1.0], not a class probability.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Signova AI: 5-Class ISL Continuous Sentence Recognition (Demo Mode)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to input ISL video file (.mp4/.avi/.mov)")
    group.add_argument("--landmarks", type=str, help="Path to pre-extracted landmark/feature .npz file")

    parser.add_argument(
        "--demo-dir",
        type=str,
        default=str(DEFAULT_DEMO_DIR),
        help="Path to demo artifacts folder",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Rejection distance threshold (default: 0.40; similarity < 0.60)",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.015,
        help="Minimum margin between top-1 and top-2 class distances (default: 0.015)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_dir = Path(args.demo_dir)

    classes, ref_meta, ref_seqs, ref_labels = load_demo_artifacts(demo_dir)

    t0 = time.perf_counter()
    if args.video:
        vpath = Path(args.video)
        if not vpath.exists():
            print(f"Error: Video file not found: {vpath}", file=sys.stderr)
            sys.exit(1)
        raw_feat = extract_features_from_video(vpath)
    else:
        lpath = Path(args.landmarks)
        if not lpath.exists():
            print(f"Error: Landmarks file not found: {lpath}", file=sys.stderr)
            sys.exit(1)
        raw_feat = extract_features_from_npz(lpath)

    proc_seq = preprocess_sequence(raw_feat)
    total_prep_ms = (time.perf_counter() - t0) * 1000.0

    result = recognize_sequence(
        proc_seq=proc_seq,
        ref_seqs=ref_seqs,
        ref_labels=ref_labels,
        classes=classes,
        distance_threshold=args.threshold,
        min_margin=args.min_margin,
    )
    result["feature_frames"] = int(proc_seq.shape[0])
    result["total_pipeline_ms"] = round(total_prep_ms + result["latency_ms"], 2)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 65)
    print(" SIGNOVA AI: CONTINUOUS ISL SENTENCE RECOGNITION (DEMO MODE)")
    print("=" * 65)
    print(f"Input: {args.video or args.landmarks}")
    print(f"Frames: {proc_seq.shape[0]} downsampled frames (48 dims)")
    print(f"Matching Latency: {result['latency_ms']} ms | Total Pipeline: {result['total_pipeline_ms']} ms")
    print("-" * 65)

    if result["is_recognized"]:
        print(f"STATUS: SUCCESS")
        print(f"PREDICTED SENTENCE: \"{result['predicted_sentence']}\"")
        print(f"SIMILARITY SCORE:   {result['top_similarity']:.4f} (DTW Distance: {result['top_distance']:.4f})")
        print(f"DECISION MARGIN:    +{result['margin_to_second']:.4f} over 2nd rank")
    else:
        print(f"STATUS: REJECTED")
        print(f"RESULT: \"{result['status']}\"")
        print(f"CLOSEST MATCH:      \"{result['rankings'][0]['sentence']}\" (Dist: {result['top_distance']:.4f}, Sim: {result['top_similarity']:.4f})")
        print(f"REASON:             DTW distance exceeded {args.threshold:.2f} or margin < {args.min_margin:.3f}")

    print("-" * 65)
    print("5-CLASS RANKINGS (Ranked by DTW Cosine Distance):")
    for r_idx, r in enumerate(result["rankings"], 1):
        mark = "->" if r_idx == 1 and result["is_recognized"] else "  "
        print(f" {mark} Rank {r_idx}: \"{r['sentence']}\" (Gloss: {r['isl_gloss']})")
        print(f"          DTW Dist: {r['dtw_distance']:.4f} | Similarity: {r['similarity']:.4f}")
    print("-" * 65)
    print(f"Note: {result['note']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
