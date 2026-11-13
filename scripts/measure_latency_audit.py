"""
Benchmark script to measure latency across all individual stages of the isolated ISL recognition pipeline.
"""
import time
import json
from pathlib import Path
import cv2
import numpy as np

import sys
PROJECT_ROOT = Path("D:/Projects/Signova AI")
sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.inference.classifier import IsolatedSignRecognizer
from ai.isolated.preprocessing.normalizer import spatial_normalize, detect_active_motion_window, temporal_resample
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor, _assign_hands_anatomically

def benchmark_pipeline():
    demo_dir = PROJECT_ROOT / "datasets" / "demo_samples"
    demo_manifest = demo_dir / "demo_manifest.json"
    with open(demo_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Measure Model Load Time
    t0 = time.perf_counter()
    recognizer = IsolatedSignRecognizer()
    recognizer.load_model()
    t_model_load = (time.perf_counter() - t0) * 1000.0
    print(f"Model Load Time: {t_model_load:.2f} ms")

    # 2. Measure MediaPipe Extractor Init Time
    t0 = time.perf_counter()
    extractor = ModernMediaPipeExtractor()
    t_mp_init = (time.perf_counter() - t0) * 1000.0
    print(f"MediaPipe Extractor Init Time: {t_mp_init:.2f} ms")

    results = []

    for sign_name, info in manifest.items():
        video_path = demo_dir / info["video_file"]
        if not video_path.exists():
            continue

        # Open video and get stats
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = frame_count / fps
        cap.release()

        # Step A: Fresh extractor init as currently done in predict_video
        t0 = time.perf_counter()
        fresh_extractor = ModernMediaPipeExtractor()
        t_fresh_init = (time.perf_counter() - t0) * 1000.0

        # Step B: Video extraction (OpenCV read + MediaPipe Video mode per frame)
        t0 = time.perf_counter()
        raw_spatial, meta = fresh_extractor.extract_isign_video(video_path)
        t_video_extract = (time.perf_counter() - t0) * 1000.0

        landmarks_75x4 = raw_spatial.reshape(-1, 75, 4)
        T = landmarks_75x4.shape[0]

        # Step C: Spatial normalization
        t0 = time.perf_counter()
        normed = spatial_normalize(landmarks_75x4)
        t_spatial_norm = (time.perf_counter() - t0) * 1000.0

        # Step D: Motion window detection
        t0 = time.perf_counter()
        s_idx, e_idx = detect_active_motion_window(normed)
        active_normed = normed[s_idx:e_idx]
        t_motion_window = (time.perf_counter() - t0) * 1000.0

        # Step E: Temporal resampling to 64
        t0 = time.perf_counter()
        resampled = temporal_resample(active_normed, target_length=64)
        t_resample = (time.perf_counter() - t0) * 1000.0

        # Step F: Feature extraction (spatial 300 + velocity 300)
        t0 = time.perf_counter()
        features = extract_isolated_features(resampled)
        t_feat = (time.perf_counter() - t0) * 1000.0

        # Step G: Bi-GRU Model Inference
        t0 = time.perf_counter()
        seq_input = features[np.newaxis, ...]
        probs = recognizer.model.predict_proba(seq_input)[0]
        t_infer = (time.perf_counter() - t0) * 1000.0

        # Step H: Ambiguity rejection & formatting
        t0 = time.perf_counter()
        top_idx = int(np.argmax(probs))
        pred_sign = recognizer.idx_to_class.get(top_idx, "unknown")
        conf = float(probs[top_idx])
        t_post = (time.perf_counter() - t0) * 1000.0

        total_post_capture = t_fresh_init + t_video_extract + t_spatial_norm + t_motion_window + t_resample + t_feat + t_infer + t_post

        results.append({
            "sign": sign_name,
            "pred_sign": pred_sign,
            "conf": conf,
            "frames": frame_count,
            "duration_s": duration_s,
            "t_fresh_init": t_fresh_init,
            "t_video_extract": t_video_extract,
            "ms_per_frame": t_video_extract / max(frame_count, 1),
            "t_spatial_norm": t_spatial_norm,
            "t_motion_window": t_motion_window,
            "t_resample": t_resample,
            "t_feat": t_feat,
            "t_infer": t_infer,
            "t_post": t_post,
            "total_post_capture": total_post_capture,
        })
        print(f"[{sign_name}] ({frame_count} frames, {duration_s:.1f}s): "
              f"MP Init={t_fresh_init:.1f}ms, Extract={t_video_extract:.1f}ms ({t_video_extract/frame_count:.1f}ms/frame), "
              f"Norm={t_spatial_norm:.2f}ms, MotionWin={t_motion_window:.2f}ms, Resample={t_resample:.2f}ms, "
              f"Feat={t_feat:.2f}ms, Infer={t_infer:.1f}ms. "
              f"Total Post-Capture={total_post_capture:.1f}ms. Pred={pred_sign} ({conf:.2%})")

    # Aggregate stats
    extract_times = [r["t_video_extract"] for r in results]
    infer_times = [r["t_infer"] for r in results]
    post_capture_times = [r["total_post_capture"] for r in results]
    init_times = [r["t_fresh_init"] for r in results]

    print("\n" + "="*60)
    print("AGGREGATE LATENCY AUDIT (10 Demo Samples)")
    print("="*60)
    print(f"Extractor Fresh Init: Avg={np.mean(init_times):.1f}ms, Median={np.median(init_times):.1f}ms")
    print(f"MediaPipe Extraction: Avg={np.mean(extract_times):.1f}ms, Median={np.median(extract_times):.1f}ms, P95={np.percentile(extract_times, 95):.1f}ms")
    print(f"Spatial Normalization: Avg={np.mean([r['t_spatial_norm'] for r in results]):.2f}ms")
    print(f"Motion Window Detection: Avg={np.mean([r['t_motion_window'] for r in results]):.2f}ms")
    print(f"Temporal Resampling: Avg={np.mean([r['t_resample'] for r in results]):.2f}ms")
    print(f"Feature Extraction: Avg={np.mean([r['t_feat'] for r in results]):.2f}ms")
    print(f"Bi-GRU Model Inference: Avg={np.mean(infer_times):.1f}ms, Median={np.median(infer_times):.1f}ms, P95={np.percentile(infer_times, 95):.1f}ms")
    print(f"Total Post-Capture Backend Processing: Avg={np.mean(post_capture_times):.1f}ms, Median={np.median(post_capture_times):.1f}ms, P95={np.percentile(post_capture_times, 95):.1f}ms")

if __name__ == "__main__":
    benchmark_pipeline()
