"""
Runner script for MediaPipe Modern Tasks API Real Video Benchmarks.
Evaluates:
1. 10 granfather sick  (2).mov
2. Indian Sign Language Video Dataset.mp4
3. Learn-Indian-Sign-SnapYT.App (1) (online-video-cutter.com).mp4
"""

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "models" / "assets")

from ai.inference.model_loader import load_model
from ai.inference.pose_translator import PoseTranslator
from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from ai.preprocessing.pose_preprocessor import preprocess_pose


def run_benchmark():
    videos = [
        {
            "id": "video_1",
            "name": "10 granfather sick  (2).mov",
            "path": PROJECT_ROOT / "datasets" / "videos" / "10 granfather sick  (2).mov",
            "ground_truth": "Grandfather is sick",
        },
        {
            "id": "video_2",
            "name": "Indian Sign Language Video Dataset.mp4",
            "path": PROJECT_ROOT / "datasets" / "videos" / "Indian Sign Language Video Dataset.mp4",
            "ground_truth": "ISL Video Dataset Sample",
        },
        {
            "id": "video_3",
            "name": "Learn-Indian-Sign-SnapYT.App (1) (online-video-cutter.com).mp4",
            "path": PROJECT_ROOT / "datasets" / "videos" / "Learn-Indian-Sign-SnapYT.App (1) (online-video-cutter.com).mp4",
            "ground_truth": "hello everybody i hope you are all doing good",
        },
    ]

    extractor = MediaPipeLandmarkExtractor(engine="modern")
    load_model(device="cpu")
    translator = PoseTranslator(auto_load=True, device="cpu")

    results = []

    for v in videos:
        print("=" * 70)
        print(f"Processing {v['name']} via Modern Tasks API...")
        t0 = time.perf_counter()
        raw_kps, meta = extractor.extract_isign_video(v["path"])
        t_extract = time.perf_counter() - t0

        # Preprocess
        prep = preprocess_pose(
            raw_kps, max_length=500, use_motion=True, device="cpu", metadata=meta
        )

        # Motion feature stats
        motion_part = prep.features[: prep.actual_length, 300:]
        motion_stats = {
            "min": round(float(np.min(motion_part)), 4),
            "max": round(float(np.max(motion_part)), 4),
            "mean": round(float(np.mean(motion_part)), 4),
            "std": round(float(np.std(motion_part)), 4),
            "zero_ratio": round(float(np.mean(motion_part == 0.0)), 4),
        }

        # Inference
        t_inf_start = time.perf_counter()
        res = translator.predict_with_metadata(prep, num_beams=4, max_length=128)
        t_inf = (time.perf_counter() - t_inf_start) * 1000.0

        entry = {
            "id": v["id"],
            "video_name": v["name"],
            "ground_truth": v["ground_truth"],
            "container_frame_count": meta["container_frame_count"],
            "read_frames": meta["read_frames"],
            "fps": round(meta["fps"], 2),
            "resolution": meta["resolution"],
            "engine": meta.get("engine", "modern_tasks"),
            "extraction_fps": meta.get("extraction_fps", round(meta["read_frames"] / max(t_extract, 1e-4), 2)),
            "pose_detection_rate": meta["pose_detection_rate"],
            "left_hand_detection_rate": meta["left_hand_detection_rate"],
            "right_hand_detection_rate": meta["right_hand_detection_rate"],
            "missing_hand_percentage": meta.get("missing_hand_percentage", 0.0),
            "missing_landmark_percentage": meta["missing_landmark_percentage"],
            "average_pose_visibility": meta["average_pose_visibility"],
            "occlusion_warning": meta["occlusion_warning"],
            "motion_stats": motion_stats,
            "tensor_shape": list(prep.pose_tensor.shape),
            "tensor_dtype": str(prep.pose_tensor.dtype),
            "pose_length": prep.actual_length,
            "inference_latency_ms": round(t_inf, 2),
            "generated_text": res.text,
            "token_ids": res.token_ids,
        }
        results.append(entry)
        print(f"Extraction: {t_extract:.2f}s ({entry['extraction_fps']} FPS) | Latency: {t_inf:.1f}ms")
        print(f"Pose Rate: {meta['pose_detection_rate'] * 100:.1f}% | LH: {meta['left_hand_detection_rate'] * 100:.1f}% | RH: {meta['right_hand_detection_rate'] * 100:.1f}%")
        print(f"Generated text: \"{res.text}\"")

    extractor.close()

    out_file = PROJECT_ROOT / "datasets" / "videos" / "modern_tasks_real_video_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark results successfully saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
