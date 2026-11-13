"""
F-031: Comprehensive Evaluation Runner for Signova AI Isolated ISL Recognition.
Evaluates the Bi-GRU champion model on:
1. Held-out test set (30 genuine samples across 10 classes).
2. Cross-tabulation across recording conditions (signer, resolution, duration).
3. Out-of-vocabulary (OOV) rejection testing (non-10-class signs, continuous videos, noise).
4. Threshold sweep and coverage analysis.
5. Latency profiling on CPU.
"""

import csv
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    LANDMARKS_DIR,
    SPLITS_DIR,
)
from ai.isolated.evaluation.metrics import compute_isolated_metrics
from ai.isolated.inference.classifier import IsolatedSignRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("signova.eval_runner")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_CSV_PATH = SPLITS_DIR / "test.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "datasets" / "isolated_test_evaluation_report.json"


def evaluate_test_split(recognizer: IsolatedSignRecognizer) -> Dict[str, Any]:
    """Evaluates all 30 held-out test samples."""
    with open(TEST_CSV_PATH, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info("Evaluating %d test samples from %s", len(rows), TEST_CSV_PATH)
    
    y_true_indices = []
    y_pred_indices = []
    y_probabilities = []
    sample_records = []
    latencies = []

    for idx, row in enumerate(rows, 1):
        vid = row["video_id"]
        gt_label = row["class_label"]
        npz_file = LANDMARKS_DIR / "test" / f"{vid}.npz"
        
        if not npz_file.exists():
            logger.error("Missing landmark file: %s", npz_file)
            continue

        data = np.load(npz_file)
        kps = data["landmarks"]  # (T, 75, 4)

        t0 = time.perf_counter()
        pred_res = recognizer.predict_landmarks(kps, top_k=3)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        pred_label = pred_res["predicted_sign"]
        conf = pred_res["confidence"]
        status = pred_res["status"]
        top_preds = pred_res["top_predictions"]
        top3_signs = [p["sign"] for p in top_preds]

        gt_idx = CLASS_TO_IDX[gt_label]
        pred_idx = CLASS_TO_IDX.get(pred_label, -1)
        
        # Get raw probability vector
        features = recognizer.predict_landmarks.__wrapped__ if hasattr(recognizer.predict_landmarks, "__wrapped__") else None
        # To get probability array for metrics:
        # Re-run proba directly via model
        from ai.isolated.features.feature_extractor import extract_isolated_features
        from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample
        normed = spatial_normalize(kps)
        resampled = temporal_resample(normed, target_length=recognizer.sequence_length)
        feats = extract_isolated_features(resampled)
        probs = recognizer.model.predict_proba(feats[np.newaxis, ...])[0]

        y_true_indices.append(gt_idx)
        y_pred_indices.append(pred_idx)
        y_probabilities.append(probs)

        is_correct = (pred_label == gt_label)
        is_top3_correct = (gt_label in top3_signs)

        sample_records.append({
            "sample_index": idx,
            "video_id": vid,
            "ground_truth": gt_label,
            "predicted_sign": pred_label,
            "confidence": round(float(conf), 4),
            "status": status,
            "is_correct": is_correct,
            "is_top3_correct": is_top3_correct,
            "top_3": top_preds,
            "latency_ms": round(lat_ms, 2),
            "signer_id": row.get("signer_id", "unknown"),
            "source_dataset": row.get("source_dataset", "unknown"),
            "resolution": row.get("resolution", "unknown"),
            "fps": row.get("fps", "unknown"),
            "duration_sec": row.get("duration_sec", "unknown"),
            "frames": int(kps.shape[0]),
            "video_path": row.get("relative_video_path", ""),
        })

    y_true = np.array(y_true_indices, dtype=np.int64)
    y_pred = np.array(y_pred_indices, dtype=np.int64)
    y_prob_arr = np.array(y_probabilities, dtype=np.float32)

    metrics = compute_isolated_metrics(y_true, y_pred, y_prob_arr, target_names=ISOLATED_CLASSES)

    return {
        "metrics": metrics,
        "sample_records": sample_records,
        "latency_stats": {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "min_ms": round(float(np.min(latencies)), 2),
            "max_ms": round(float(np.max(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        },
    }


def evaluate_recording_conditions(sample_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Slices test performance by signer, resolution, duration, and dataset source."""
    def compute_slice_acc(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(records)
        if total == 0:
            return {"total": 0, "accuracy": 0.0, "top3_accuracy": 0.0}
        corr = sum(1 for r in records if r["is_correct"])
        top3 = sum(1 for r in records if r["is_top3_correct"])
        return {
            "total": total,
            "correct": corr,
            "top3_correct": top3,
            "accuracy": round(corr / total, 4),
            "top3_accuracy": round(top3 / total, 4),
        }

    # Group by Signer
    signers: Dict[str, List[Dict[str, Any]]] = {}
    for r in sample_records:
        signers.setdefault(r["signer_id"], []).append(r)
    signer_breakdown = {s: compute_slice_acc(recs) for s, recs in signers.items()}

    # Group by Resolution
    resolutions: Dict[str, List[Dict[str, Any]]] = {}
    for r in sample_records:
        resolutions.setdefault(r["resolution"], []).append(r)
    resolution_breakdown = {res: compute_slice_acc(recs) for res, recs in resolutions.items()}

    # Group by Source Dataset
    sources: Dict[str, List[Dict[str, Any]]] = {}
    for r in sample_records:
        sources.setdefault(r["source_dataset"], []).append(r)
    source_breakdown = {src: compute_slice_acc(recs) for src, recs in sources.items()}

    # Group by Duration (< 2.5s, 2.5 - 4.0s, > 4.0s)
    duration_groups = {"short (<2.5s)": [], "medium (2.5-4s)": [], "long (>4s)": []}
    for r in sample_records:
        try:
            d = float(r["duration_sec"])
            if d < 2.5:
                duration_groups["short (<2.5s)"].append(r)
            elif d <= 4.0:
                duration_groups["medium (2.5-4s)"].append(r)
            else:
                duration_groups["long (>4s)"].append(r)
        except Exception:
            pass
    duration_breakdown = {grp: compute_slice_acc(recs) for grp, recs in duration_groups.items()}

    return {
        "by_signer": signer_breakdown,
        "by_resolution": resolution_breakdown,
        "by_source_dataset": source_breakdown,
        "by_duration": duration_breakdown,
    }


def evaluate_threshold_sweep(sample_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluates accuracy vs coverage across varying rejection thresholds."""
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    sweep_results = []
    total = len(sample_records)

    for tau in thresholds:
        accepted = [r for r in sample_records if r["confidence"] >= tau]
        rejected = [r for r in sample_records if r["confidence"] < tau]
        
        n_acc = len(accepted)
        n_rej = len(rejected)
        coverage_pct = round((n_acc / total) * 100.0, 1) if total > 0 else 0.0

        if n_acc > 0:
            corr_acc = sum(1 for r in accepted if r["is_correct"])
            acc_on_accepted = round(corr_acc / n_acc, 4)
        else:
            acc_on_accepted = 0.0

        sweep_results.append({
            "threshold": tau,
            "coverage_pct": coverage_pct,
            "accepted_count": n_acc,
            "rejected_count": n_rej,
            "accuracy_on_accepted": acc_on_accepted,
        })

    return sweep_results


def evaluate_out_of_vocabulary(recognizer: IsolatedSignRecognizer) -> Dict[str, Any]:
    """Tests non-10-class gestures and noise to verify rejection mechanism."""
    oov_results = []

    # 1. Test unselected isolated sign categories from dataset
    oov_folders = ["brother", "eat", "father", "goodbye", "hospital", "mother", "school", "tea"]
    isl_40_base = PROJECT_ROOT / "dataset" / "isl-isolated-40words"
    
    from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor
    extractor = ModernMediaPipeExtractor()

    cache_dir = PROJECT_ROOT / "datasets" / ".oov_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for folder_name in oov_folders:
        folder_path = isl_40_base / folder_name
        if not folder_path.exists():
            continue
        vids = list(folder_path.glob("*.mp4"))
        if not vids:
            continue
        test_vid = vids[0]  # Pick 1 sample per OOV class
        cache_file = cache_dir / f"{folder_name}.npz"
        try:
            if cache_file.exists():
                kps = np.load(cache_file)["kps"]
            else:
                raw_spatial, meta = extractor.extract_isign_video(test_vid)
                kps = raw_spatial.reshape(-1, 75, 4)
                np.savez_compressed(cache_file, kps=kps)

            pred = recognizer.predict_landmarks(kps, category="oov_word")
            oov_results.append({
                "type": "unselected_isolated_sign",
                "source": f"{folder_name}/{test_vid.name}",
                "true_concept": folder_name,
                "predicted_sign": pred["predicted_sign"],
                "confidence": pred["confidence"],
                "status": pred["status"],
                "is_rejected_properly": (pred["status"] == "uncertain"),
            })
        except Exception as exc:
            logger.warning("Failed OOV video test for %s: %s", test_vid, exc)

    # 2. Test user continuous videos
    continuous_videos = [
        ("10 granfather sick  (2).mov", "continuous_phrase: grandfather sick"),
        ("WIN_20261112_12_11_45_Pro.mp4", "webcam_ambient_continuous"),
    ]
    for filename, desc in continuous_videos:
        vpath = PROJECT_ROOT / "datasets" / "videos" / filename
        if vpath.exists():
            cache_file = cache_dir / f"{Path(filename).stem}.npz"
            try:
                if cache_file.exists():
                    kps = np.load(cache_file)["kps"]
                else:
                    raw_spatial, meta = extractor.extract_isign_video(vpath)
                    kps = raw_spatial.reshape(-1, 75, 4)
                    np.savez_compressed(cache_file, kps=kps)

                pred = recognizer.predict_landmarks(kps, category="continuous_video")
                oov_results.append({
                    "type": "continuous_isl_video",
                    "source": filename,
                    "true_concept": desc,
                    "predicted_sign": pred["predicted_sign"],
                    "confidence": pred["confidence"],
                    "status": pred["status"],
                    "is_rejected_properly": (pred["status"] == "uncertain"),
                })
            except Exception as exc:
                logger.warning("Failed continuous video test for %s: %s", filename, exc)

    # 3. Test Gaussian noise and zero input
    noise_seq = np.random.randn(64, 75, 4).astype(np.float32)
    noise_pred = recognizer.predict_landmarks(noise_seq, category="synthetic_noise")
    oov_results.append({
        "type": "synthetic_noise",
        "source": "gaussian_noise_64x75x4",
        "true_concept": "random_gaussian_noise",
        "predicted_sign": noise_pred["predicted_sign"],
        "confidence": noise_pred["confidence"],
        "status": noise_pred["status"],
        "is_rejected_properly": (noise_pred["status"] == "uncertain"),
    })

    zero_seq = np.zeros((64, 75, 4), dtype=np.float32)
    zero_pred = recognizer.predict_landmarks(zero_seq, category="synthetic_zeros")
    oov_results.append({
        "type": "synthetic_zeros",
        "source": "all_zeros_64x75x4",
        "true_concept": "zero_landmark_input",
        "predicted_sign": zero_pred["predicted_sign"],
        "confidence": zero_pred["confidence"],
        "status": zero_pred["status"],
        "is_rejected_properly": (zero_pred["status"] == "uncertain"),
    })

    total_oov = len(oov_results)
    rejected_cnt = sum(1 for r in oov_results if r["is_rejected_properly"])
    rejection_rate = round(rejected_cnt / total_oov, 4) if total_oov > 0 else 0.0

    return {
        "total_oov_samples": total_oov,
        "correctly_rejected_count": rejected_cnt,
        "false_acceptance_count": total_oov - rejected_cnt,
        "oov_rejection_rate": rejection_rate,
        "samples": oov_results,
    }


def main():
    logger.info("Initializing IsolatedSignRecognizer...")
    recognizer = IsolatedSignRecognizer(checkpoint_path=ISOLATED_CHECKPOINT_PATH)
    recognizer.load_model()
    assert recognizer.is_loaded, "Model failed to load!"

    test_eval = evaluate_test_split(recognizer)
    sample_records = test_eval["sample_records"]

    cond_eval = evaluate_recording_conditions(sample_records)
    sweep_eval = evaluate_threshold_sweep(sample_records)
    oov_eval = evaluate_out_of_vocabulary(recognizer)

    full_report = {
        "task": "F-031: Validate Signova AI Isolated ISL Recognition",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": {
            "type": "IsolatedBiGRU",
            "wrapper": "GRUClassifierWrapper",
            "checkpoint": str(ISOLATED_CHECKPOINT_PATH),
            "feature_dim": 600,
            "sequence_length": 64,
            "classes": ISOLATED_CLASSES,
            "rejection_threshold": recognizer.rejection_threshold,
        },
        "test_split_evaluation": {
            "total_samples": len(sample_records),
            "metrics": test_eval["metrics"],
            "latency_stats": test_eval["latency_stats"],
            "sample_records": sample_records,
        },
        "recording_conditions_analysis": cond_eval,
        "rejection_threshold_sweep": sweep_eval,
        "out_of_vocabulary_evaluation": oov_eval,
    }

    with open(OUTPUT_REPORT_PATH, mode="w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    logger.info("Comprehensive evaluation report saved to: %s", OUTPUT_REPORT_PATH)
    print("\n" + "=" * 60)
    print("F-031 EVALUATION COMPLETE")
    print(f"Top-1 Test Accuracy: {test_eval['metrics']['accuracy'] * 100:.2f}%")
    print(f"Top-3 Test Accuracy: {test_eval['metrics']['top3_accuracy'] * 100:.2f}%")
    print(f"Average CPU Latency: {test_eval['latency_stats']['mean_ms']:.2f} ms")
    print(f"OOV Rejection Rate: {oov_eval['oov_rejection_rate'] * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
