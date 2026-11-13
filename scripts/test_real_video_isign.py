"""
Single Real ISL Video Test: Grandfather Is Sick
Evaluates iSign-T5 (t5-small-motion) on '10 granfather sick  (2).mov'
Reuses F-013 model loader, F-014 preprocessor, and F-015 PoseTranslator.
"""

import collections
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid matplotlib cache warning
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "models" / "assets")

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ai.inference.model_loader import get_model, is_model_loaded, load_model, resolve_device
from ai.inference.pose_translator import PoseTranslator, predict_with_metadata
from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from ai.preprocessing.pose_preprocessor import (
    BASE_SPATIAL_DIM,
    HAND_LANDMARK_COUNT,
    POSE_LANDMARK_COUNT,
    TOTAL_LANDMARKS,
    PreprocessedPose,
    assemble_frame_landmarks,
    preprocess_pose,
)

# Constants for this test
TEST_VIDEO_PATH = PROJECT_ROOT / "datasets" / "videos" / "10 granfather sick  (2).mov"
GROUND_TRUTH_TEXT = "Grandfather is sick"
MODEL_TASK_PATH = PROJECT_ROOT / "models" / "assets" / "holistic_landmarker.task"
REPORT_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "videos" / "10_granfather_sick_experiment_report.json"


# ---------------------------------------------------------------------------
# Metric Implementations (BLEU-4, ROUGE-L, chrF) - Pure Python
# ---------------------------------------------------------------------------

def compute_lcs(a: List[str], b: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 score."""
    c_tokens = candidate.lower().split()
    r_tokens = reference.lower().split()
    if not c_tokens or not r_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = compute_lcs(c_tokens, r_tokens)
    p = lcs / len(c_tokens)
    r = lcs / len(r_tokens)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Compute sentence BLEU-4 with brevity penalty and smoothing."""
    c_tokens = candidate.lower().split()
    r_tokens = reference.lower().split()
    c_len, r_len = len(c_tokens), len(r_tokens)
    if c_len == 0 or r_len == 0:
        return 0.0

    bp = 1.0 if c_len >= r_len else math.exp(1.0 - r_len / c_len)

    actual_max_n = min(max_n, c_len, r_len)
    if actual_max_n == 0:
        return 0.0

    precisions = []
    for n in range(1, actual_max_n + 1):
        c_ngrams = collections.Counter(tuple(c_tokens[i : i + n]) for i in range(len(c_tokens) - n + 1))
        r_ngrams = collections.Counter(tuple(r_tokens[i : i + n]) for i in range(len(r_tokens) - n + 1))
        possible = len(c_tokens) - n + 1
        clipped = sum(min(cnt, r_ngrams.get(ng, 0)) for ng, cnt in c_ngrams.items())
        p_n = (clipped + 0.01) / (possible + 0.01) if clipped == 0 else clipped / possible
        precisions.append(p_n)

    log_sum = sum((1.0 / len(precisions)) * math.log(p) for p in precisions)
    bleu = bp * math.exp(log_sum)
    return round(bleu, 4)


def compute_chrf(candidate: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Compute character n-gram F-score (chrF)."""
    c_chars = candidate.lower().replace(" ", "")
    r_chars = reference.lower().replace(" ", "")
    if not c_chars or not r_chars:
        return 0.0

    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        c_ngrams = collections.Counter(c_chars[i : i + n] for i in range(len(c_chars) - n + 1))
        r_ngrams = collections.Counter(r_chars[i : i + n] for i in range(len(r_chars) - n + 1))
        total_c = max(1, len(c_chars) - n + 1)
        total_r = max(1, len(r_chars) - n + 1)
        matched = sum(min(cnt, r_ngrams.get(ng, 0)) for ng, cnt in c_ngrams.items())
        precisions.append(matched / total_c)
        recalls.append(matched / total_r)

    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    denom = (beta**2 * avg_p) + avg_r
    if denom == 0:
        return 0.0
    score = (1 + beta**2) * (avg_p * avg_r) / denom
    return round(score, 4)


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------

def run_experiment(
    video_path: Path = TEST_VIDEO_PATH,
    ground_truth: str = GROUND_TRUTH_TEXT,
    num_beams: int = 4,
    device: str = "cpu",
    engine: str = "modern",
    output_report: Optional[Path] = None,
) -> Dict[str, Any]:
    print("=" * 70)
    print("Signova AI - Single Real ISL Video Test: Grandfather Is Sick")
    print("=" * 70)

    total_start_time = time.perf_counter()
    report_target = output_report or REPORT_OUTPUT_PATH

    # -----------------------------------------------------------------------
    # Step 1 & 2: Video Reading with OpenCV
    # -----------------------------------------------------------------------
    print(f"\n[Step 2] Reading Video: {video_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Test video not found: {video_path}")

    t_video_start = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Container Metadata -> Frame Count: {expected_frames}, FPS: {fps}, Resolution: {width}x{height}")

    frames = []
    corrupted_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            corrupted_count += 1
            continue
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        frame_idx += 1

    cap.release()
    t_video_end = time.perf_counter()
    video_read_duration = t_video_end - t_video_start

    print(f"  Successfully read {len(frames)} frames ({corrupted_count} corrupted)")
    print(f"  Video reading duration: {video_read_duration:.4f}s")

    if len(frames) == 0:
        raise RuntimeError("No valid frames could be read from video.")

    # -----------------------------------------------------------------------
    # Step 3: Extract Pose Landmarks (MediaPipe Tasks API)
    # -----------------------------------------------------------------------
    print(f"\n[Step 3] Extracting Landmarks using MediaPipe Tasks API ({engine})...")
    t_extract_start = time.perf_counter()
    extractor = MediaPipeLandmarkExtractor(engine=engine)
    raw_spatial_array, video_meta = extractor.extract_isign_video(video_path)
    extractor.close()
    t_extract_end = time.perf_counter()
    pose_extract_duration = t_extract_end - t_extract_start

    pose_detected_count = int(round(video_meta["pose_detection_rate"] * len(frames)))
    left_hand_detected_count = int(round(video_meta["left_hand_detection_rate"] * len(frames)))
    right_hand_detected_count = int(round(video_meta["right_hand_detection_rate"] * len(frames)))

    print(f"  Landmark Extraction Complete:")
    print(f"    Raw spatial sequence shape: {raw_spatial_array.shape}")
    print(f"    Pose detected: {pose_detected_count}/{len(frames)} frames ({video_meta['pose_detection_rate']*100:.1f}%)")
    print(f"    Left hand detected: {left_hand_detected_count}/{len(frames)} frames ({video_meta['left_hand_detection_rate']*100:.1f}%)")
    print(f"    Right hand detected: {right_hand_detected_count}/{len(frames)} frames ({video_meta['right_hand_detection_rate']*100:.1f}%)")
    avg_vis = video_meta.get("average_pose_visibility", 0.0)
    print(f"    Average Pose Landmark Visibility: {avg_vis:.4f}")
    print(f"    Extraction duration: {pose_extract_duration:.4f}s ({len(frames)/pose_extract_duration:.1f} fps)")

    # -----------------------------------------------------------------------
    # Step 4: Prepare Model Input (F-014 Preprocessing)
    # -----------------------------------------------------------------------
    print(f"\n[Step 4] Preprocessing with F-014 (Z-score normalization, motion features, padding)...")
    t_prep_start = time.perf_counter()

    preprocessed = preprocess_pose(
        pose_input=raw_spatial_array,
        max_length=500,
        normalization="zscore",
        use_motion=True,
        device=device,
    )

    t_prep_end = time.perf_counter()
    prep_duration = t_prep_end - t_prep_start

    print(f"  F-014 Preprocessing Results:")
    print(f"    Pose Tensor Shape: {tuple(preprocessed.pose_tensor.shape)} (batch=1, max_len=500, dim=600)")
    print(f"    Pose Length: {preprocessed.actual_length} frames (tensor value: {preprocessed.pose_length.item()})")
    print(f"    Tensor Dtype: {preprocessed.pose_tensor.dtype}")
    print(f"    Tensor Device: {preprocessed.device}")
    print(f"    Preprocessing duration: {prep_duration*1000:.2f} ms")

    # -----------------------------------------------------------------------
    # Step 5: Model Inference (F-013 & F-015)
    # -----------------------------------------------------------------------
    print(f"\n[Step 5] Running Standalone Inference with F-013/F-015 (t5-small-motion)...")
    translator = PoseTranslator(auto_load=True, device=device)

    t_infer_start = time.perf_counter()
    inference_result = translator.predict_with_metadata(
        pose_data=preprocessed,
        num_beams=num_beams,
        length_penalty=2.0,
    )
    t_infer_end = time.perf_counter()
    actual_infer_time_ms = (t_infer_end - t_infer_start) * 1000.0

    print(f"  Inference Complete:")
    print(f"    Exact Generated Sentence: \"{inference_result.text}\"")
    print(f"    Generated Token IDs: {inference_result.token_ids}")
    print(f"    Inference Latency: {inference_result.latency_ms:.2f} ms")
    print(f"    Device Used: {inference_result.device}")

    # -----------------------------------------------------------------------
    # Step 6: Evaluation Against Ground Truth
    # -----------------------------------------------------------------------
    print(f"\n[Step 6] Comparing against Ground Truth: \"{ground_truth}\"")
    exact_match = (inference_result.text.strip().lower() == ground_truth.strip().lower())
    bleu_score = compute_bleu(inference_result.text, ground_truth)
    rouge_l_scores = compute_rouge_l(inference_result.text, ground_truth)
    chrf_score = compute_chrf(inference_result.text, ground_truth)

    print(f"  Evaluation Metrics:")
    print(f"    Exact Match: {exact_match}")
    print(f"    BLEU-4: {bleu_score}")
    print(f"    ROUGE-L: {rouge_l_scores}")
    print(f"    chrF: {chrf_score}")

    total_end_time = time.perf_counter()
    total_execution_time = total_end_time - total_start_time

    # -----------------------------------------------------------------------
    # Step 7 & 9: Compile Comprehensive Report
    # -----------------------------------------------------------------------
    report_data: Dict[str, Any] = {
        "experiment": "Single Real ISL Video Test: Grandfather Is Sick",
        "video_path": str(video_path),
        "ground_truth": ground_truth,
        "selected_model": "manavdhamecha77/iSign-t5-pose-to-text",
        "checkpoint": "t5-small-motion",
        "video_reading": {
            "is_opened": True,
            "container_frame_count": expected_frames,
            "successfully_read_frames": len(frames),
            "corrupted_frames": corrupted_count,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "frame_order_preserved": True,
            "reading_time_seconds": round(video_read_duration, 4),
        },
        "pose_extraction": {
            "extractor": f"MediaPipe Tasks API ({engine})",
            "engine": engine,
            "pose_landmarks_per_frame": POSE_LANDMARK_COUNT,
            "left_hand_landmarks_per_frame": HAND_LANDMARK_COUNT,
            "right_hand_landmarks_per_frame": HAND_LANDMARK_COUNT,
            "total_landmarks_per_frame": TOTAL_LANDMARKS,
            "spatial_dim_per_frame": BASE_SPATIAL_DIM,
            "raw_spatial_shape": list(raw_spatial_array.shape),
            "pose_detected_frames": pose_detected_count,
            "pose_detection_rate": round(pose_detected_count / len(frames), 4),
            "left_hand_detected_frames": left_hand_detected_count,
            "left_hand_detection_rate": round(left_hand_detected_count / len(frames), 4),
            "right_hand_detected_frames": right_hand_detected_count,
            "right_hand_detection_rate": round(right_hand_detected_count / len(frames), 4),
            "average_pose_visibility": round(avg_vis, 4),
            "extraction_time_seconds": round(pose_extract_duration, 4),
            "extraction_fps": round(len(frames) / pose_extract_duration, 2),
        },
        "preprocessing": {
            "module": "F-014 Pose Input Preparation",
            "method": "zscore",
            "use_motion": True,
            "max_length": 500,
            "output_tensor_shape": list(preprocessed.pose_tensor.shape),
            "pose_length": preprocessed.actual_length,
            "dtype": str(preprocessed.pose_tensor.dtype),
            "device": str(preprocessed.device),
            "preprocessing_time_ms": round(prep_duration * 1000.0, 2),
        },
        "inference": {
            "module": "F-013 Model Loader & F-015 PoseTranslator",
            "num_beams": num_beams,
            "length_penalty": 2.0,
            "exact_generated_text": inference_result.text,
            "token_ids": inference_result.token_ids,
            "inference_latency_ms": inference_result.latency_ms,
            "device": inference_result.device,
        },
        "evaluation": {
            "exact_match": exact_match,
            "bleu_4": bleu_score,
            "rouge_l_precision": rouge_l_scores["precision"],
            "rouge_l_recall": rouge_l_scores["recall"],
            "rouge_l_f1": rouge_l_scores["f1"],
            "chrf": chrf_score,
        },
        "performance_timings": {
            "video_reading_seconds": round(video_read_duration, 4),
            "pose_extraction_seconds": round(pose_extract_duration, 4),
            "preprocessing_ms": round(prep_duration * 1000.0, 2),
            "inference_ms": inference_result.latency_ms,
            "total_execution_seconds": round(total_execution_time, 4),
        },
        "blockers_and_errors": [],
        "status": "SUCCESS",
    }

    # Persist JSON report
    with open(report_target, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[Artifact] Report saved to: {report_target}")
    print("=" * 70)
    return report_data


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run single ISL real video translation experiment using iSign-T5.")
    parser.add_argument("--video", type=str, default=str(TEST_VIDEO_PATH), help="Path to input video file.")
    parser.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_TEXT, help="Ground truth English text.")
    parser.add_argument("--beams", type=int, default=4, help="Beam search width (default: 4).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--engine", type=str, default="modern", choices=["modern", "legacy"], help="MediaPipe engine.")
    parser.add_argument("--output", type=str, default=str(REPORT_OUTPUT_PATH), help="Path to save output JSON report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_experiment(
        video_path=Path(args.video),
        ground_truth=args.ground_truth,
        num_beams=args.beams,
        device=args.device,
        engine=args.engine,
        output_report=Path(args.output),
    )
