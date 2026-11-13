"""
Direct Video Testing Entry Point for Signova AI with Normal iSign T5-Base.
Executes the complete pipeline:
Video -> Frame Extraction -> MediaPipe Landmarks -> Pose Features -> Preprocessing -> T5-Base Inference -> English Prediction

Usage:
    python tests/test_video.py --video "datasets/videos/10 granfather sick  (2).mov" --reference "Grandfather is sick"
    python tests/test_video.py --video "video1.mp4" "video2.mp4" "video3.mp4"
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.inference.model_loader import load_model, unload_model
from ai.inference.pose_translator import PoseTranslator
from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from ai.preprocessing.pose_preprocessor import preprocess_pose, to_spatial_matrix

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("signova.test_video")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def format_report(
    video_name: str,
    device: str,
    duration: float,
    fps: float,
    extracted_frames: int,
    raw_shape: tuple,
    missing_count: int,
    missing_pct: float,
    feature_shape: tuple,
    tensor_shape: tuple,
    pose_length: int,
    inference_time: float,
    predicted_text: str,
    reference_text: Optional[str] = None,
    hand_strategy: str = "zero",
    max_fill_frames: int = 15,
) -> str:
    """Format the report exactly as requested in F-024/F-027 specification."""
    ref_display = f'"{reference_text}"' if reference_text else "Not provided"
    if reference_text:
        is_match = predicted_text.strip().lower() == reference_text.strip().lower()
        match_display = "YES" if is_match else f"NO (differs from reference)"
    else:
        match_display = "N/A"

    report = f"""========================================
SIGNOVA AI - VIDEO PREDICTION TEST
========================================

Video: {video_name}

MODEL
Checkpoint: iSign T5-Base
Device: {device.upper()}
Motion: Disabled

VIDEO
Duration: {duration:.2f}s
Original FPS: {fps:.2f}
Extracted frames: {extracted_frames}

LANDMARK EXTRACTION
Raw shape: {raw_shape}
Landmarks per frame: 75
Missing landmarks: {missing_count}
Missing percentage: {missing_pct:.2f}%
Hand strategy: {hand_strategy} (max_fill={max_fill_frames} frames)

POSE FEATURES
Feature shape: {feature_shape}
Features per frame: 300

PREPROCESSING
Tensor shape: {tensor_shape}
Pose length: {pose_length}
Normalization: zscore

INFERENCE
Model loaded: SUCCESS
Inference time: {inference_time:.3f}s

PREDICTION
Generated English text:
"{predicted_text}"

REFERENCE
{ref_display}

VERIFICATION
Exact match: {match_display}
Human verification: Pending

========================================"""
    return report


def _run_single_pipeline(
    video_path: Path,
    hand_strategy: str,
    max_fill_frames: int,
    ref_text: Optional[str],
    extractor: MediaPipeLandmarkExtractor,
    translator: PoseTranslator,
    resolved_device: str,
    num_beams: int,
) -> dict:
    """Execute video -> extraction -> features -> preprocess -> inference."""
    raw_spatial, meta = extractor.extract_isign_video(
        video_path,
        hand_missing_strategy=hand_strategy,
        max_forward_fill_frames=max_fill_frames,
    )

    read_frames = meta.get("read_frames", len(raw_spatial))
    fps = float(meta.get("fps", 30.0))
    duration = read_frames / fps if fps > 0 else 0.0

    reshaped = raw_spatial.reshape(read_frames, 75, 4)
    visibilities = reshaped[:, :, 3]
    missing_mask = (visibilities == 0.0)
    missing_count = int(np.sum(missing_mask))
    total_landmarks = read_frames * 75
    missing_pct = (missing_count / total_landmarks * 100.0) if total_landmarks > 0 else 0.0

    spatial_mat = to_spatial_matrix(raw_spatial)

    preprocessed = preprocess_pose(
        spatial_mat,
        max_length=500,
        use_motion=False,
        device=resolved_device,
    )

    t_infer_start = time.perf_counter()
    result = translator.predict_with_metadata(
        pose_data=preprocessed,
        max_length=128,
        num_beams=num_beams,
    )
    t_infer = time.perf_counter() - t_infer_start

    report_str = format_report(
        video_name=video_path.name,
        device=resolved_device,
        duration=duration,
        fps=fps,
        extracted_frames=read_frames,
        raw_shape=tuple(reshaped.shape),
        missing_count=missing_count,
        missing_pct=missing_pct,
        feature_shape=tuple(spatial_mat.shape),
        tensor_shape=tuple(preprocessed.pose_tensor.shape),
        pose_length=preprocessed.actual_length,
        inference_time=t_infer,
        predicted_text=result.text,
        reference_text=ref_text,
        hand_strategy=hand_strategy,
        max_fill_frames=max_fill_frames,
    )

    return {
        "report_str": report_str,
        "predicted_text": result.text,
        "inference_time": t_infer,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
    }


def run_video_tests(
    video_paths: List[str],
    references: Optional[List[str]] = None,
    device: str = "cpu",
    decoding: str = "beam",
    hand_strategy: str = "zero",
    max_fill_frames: int = 15,
) -> int:
    """Run full pipeline on provided video paths reusing single loaded model."""
    resolved_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    num_beams = 4 if decoding == "beam" else 1

    print(f"\nInitializing Signova AI Video Testing Suite on {resolved_device.upper()}...")
    print(f"Hand Missing Strategy: {hand_strategy.upper()} (max_fill={max_fill_frames} frames)")

    # 1. Initialize MediaPipe Extractor
    extractor = MediaPipeLandmarkExtractor()

    # 2. Pre-load T5-Base Simple Model once
    print("Loading iSign T5-Base simple model checkpoint (300 in -> 256 -> 768 out)...")
    translator = PoseTranslator(auto_load=True, device=resolved_device)

    total_videos = len(video_paths)
    successful_runs = 0

    try:
        for idx, raw_path_str in enumerate(video_paths):
            video_path = Path(raw_path_str).resolve()
            ref_text = None
            if references:
                if len(references) == 1 and total_videos > 1:
                    ref_text = references[0]
                elif idx < len(references):
                    ref_text = references[idx]

            print(f"\n[{idx + 1}/{total_videos}] Processing: {video_path.name}")

            # Validation
            if not video_path.exists():
                print(f"ERROR: Video file not found at '{video_path}'", file=sys.stderr)
                continue

            if video_path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
                print(
                    f"ERROR: Unsupported format '{video_path.suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
                    file=sys.stderr,
                )
                continue

            try:
                if hand_strategy == "compare":
                    # Run A: Zero-fill baseline
                    res_zero = _run_single_pipeline(
                        video_path=video_path,
                        hand_strategy="zero",
                        max_fill_frames=max_fill_frames,
                        ref_text=ref_text,
                        extractor=extractor,
                        translator=translator,
                        resolved_device=resolved_device,
                        num_beams=num_beams,
                    )
                    print(res_zero["report_str"])

                    # Run B: Forward-fill
                    res_fill = _run_single_pipeline(
                        video_path=video_path,
                        hand_strategy="forward_fill",
                        max_fill_frames=max_fill_frames,
                        ref_text=ref_text,
                        extractor=extractor,
                        translator=translator,
                        resolved_device=resolved_device,
                        num_beams=num_beams,
                    )
                    print(res_fill["report_str"])

                    pred_differs = res_zero["predicted_text"] != res_fill["predicted_text"]
                    status_change = "YES (Prediction Changed)" if pred_differs else "NO (Identical Output)"

                    print(f"""============================================================
A/B HAND-STRATEGY COMPARISON: {video_path.name}
------------------------------------------------------------
Reference Sentence:       {f'"{ref_text}"' if ref_text else "Not provided"}
Strategy 'zero':         "{res_zero['predicted_text']}"
Strategy 'forward_fill': "{res_fill['predicted_text']}"
Prediction Changed:      {status_change}
============================================================\n""")
                else:
                    res = _run_single_pipeline(
                        video_path=video_path,
                        hand_strategy=hand_strategy,
                        max_fill_frames=max_fill_frames,
                        ref_text=ref_text,
                        extractor=extractor,
                        translator=translator,
                        resolved_device=resolved_device,
                        num_beams=num_beams,
                    )
                    print(res["report_str"])

                successful_runs += 1

            except Exception as exc:
                print(f"FAILED processing {video_path.name}: {exc}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    finally:
        extractor.close()
        unload_model()

    print(f"\nVideo testing completed. Successfully processed {successful_runs}/{total_videos} videos.")
    return 0 if successful_runs == total_videos else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Signova AI End-to-End Video Testing with Normal iSign T5-Base")
    parser.add_argument(
        "--video",
        nargs="+",
        required=True,
        help="One or more video file paths (.mp4, .mov, .webm, .avi, .mkv)",
    )
    parser.add_argument(
        "--reference",
        nargs="*",
        default=None,
        help="Optional English reference sentence(s) for verification comparison",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Hardware execution device (default: cpu)",
    )
    parser.add_argument(
        "--decoding",
        choices=["beam", "greedy"],
        default="beam",
        help="T5 text generation decoding strategy (default: beam)",
    )
    parser.add_argument(
        "--hand-strategy",
        choices=["zero", "forward_fill", "compare"],
        default="zero",
        help="Missing hand imputation strategy (default: zero)",
    )
    parser.add_argument(
        "--max-fill-frames",
        type=int,
        default=15,
        help="Maximum consecutive missing hand frames to forward-fill (default: 15)",
    )

    args = parser.parse_args()
    ret_code = run_video_tests(
        video_paths=args.video,
        references=args.reference,
        device=args.device,
        decoding=args.decoding,
        hand_strategy=args.hand_strategy,
        max_fill_frames=args.max_fill_frames,
    )
    sys.exit(ret_code)


if __name__ == "__main__":
    main()
