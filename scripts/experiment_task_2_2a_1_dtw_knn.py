"""
Task 2.2A.1 Experiment: Signer Generalization and Error Correction.

Champions:
- Original Baseline: Cosine DTW k=5 (Top-1: 4.71%, Top-3: 9.41%, Macro F1: 0.0368)
- Sprint 2.2A Champion: Cosine DTW win=8, equal weights, k=7 (Top-1: 7.06%, Top-3: 12.94%, Macro F1: 0.0632)
- Sprint 2.2A.1 Champion: Cosine DTW win=10, velocity-dominant (0.05, 1.0, 0.05), k=9
  Achieves: Top-1: 17.65% (+150% relative over 2.2A), Top-3: 34.12%, Macro F1: 0.1375, 43 unique classes.

Artifacts saved:
- Checkpoint: models/isl_csltr_continuous/best_dtw_knn_task22a1_model.pkl
- Evaluation: models/isl_csltr_continuous/eval_task_2_2a_1_dtw_knn.json
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict

import numpy as np
import psutil

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier
from scripts.eval_utils import compute_benchmark_metrics

ALIAS_MAP: Dict[int, int] = {
    49: 46,
    63: 14,
}


def main() -> None:
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)

    cache_path = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "dtw_dataset_improved.pkl"
    models_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Task 2.2A improved dataset cache...")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    train_feats = data["train"]["seqs"]
    y_train = data["train"]["labels"]
    val_feats = data["validation"]["seqs"]
    y_val = data["validation"]["labels"]
    y_val_m = np.array([ALIAS_MAP.get(int(y), int(y)) for y in y_val])

    print(f"Loaded {len(train_feats)} train exemplars and {len(val_feats)} val sequences.")
    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)

    # Champion 2.2A.1 configuration: Velocity-dominant kinematics
    # Window: 10, Weights: (0.05 spatial, 1.0 velocity, 0.05 acceleration), k: 9
    champion_metric = "cosine"
    champion_window = 10
    champion_weights = (0.05, 1.0, 0.05)
    champion_k = 9

    print(f"\nComputing DTW distance matrix for Task 2.2A.1 Champion...")
    print(f"Config: metric={champion_metric}, window={champion_window}, weights={champion_weights}, k={champion_k}")

    clf = DTWKNNClassifier(
        n_neighbors=champion_k,
        metric=champion_metric,
        window=champion_window,
        weighting="distance",
        num_classes=101,
        group_weights=champion_weights,
    )
    clf.fit(train_feats, y_train)

    t0 = time.time()
    dist_matrix = clf.compute_distance_matrix(val_feats, max_workers=8)
    dtw_time = time.time() - t0
    mem_after_dtw_mb = process.memory_info().rss / (1024 * 1024)

    t_infer_0 = time.time()
    y_pred, y_probs = clf.predict_from_distances(dist_matrix, k=champion_k)
    infer_time = (time.time() - t_infer_0) + dtw_time
    latency_per_sample_ms = (infer_time / len(val_feats)) * 1000.0

    y_pred_m = np.array([ALIAS_MAP.get(int(y), int(y)) for y in y_pred])

    # Re-map posterior probabilities for aliases
    remapped_probs = y_probs.copy()
    for alias_cls, canon_cls in ALIAS_MAP.items():
        if alias_cls < 101 and canon_cls < 101:
            remapped_probs[:, canon_cls] += remapped_probs[:, alias_cls]
            remapped_probs[:, alias_cls] = 0.0

    metrics = compute_benchmark_metrics(
        y_true=y_val_m,
        y_pred=y_pred_m,
        y_probs=remapped_probs,
        num_classes=101,
        train_time_sec=0.0,
        inference_time_sec=infer_time,
    )

    pred_dist = metrics["prediction_distribution"]
    top_pred_class = max(pred_dist.keys(), key=lambda c: pred_dist[c]) if pred_dist else "none"
    top_pred_count = pred_dist.get(top_pred_class, 0)

    # Save dedicated champion model checkpoint
    model_save_path = models_dir / "best_dtw_knn_task22a1_model.pkl"
    clf.save(model_save_path)
    print(f"\nSaved Task 2.2A.1 Champion model checkpoint to: {model_save_path}")

    # Build evaluation results artifact
    results: Dict[str, Any] = {
        "experiment": "Task 2.2A.1: Signer Generalization and Error Correction",
        "description": "Velocity-Dominant Kinematic DTW with Constrained Warping Window (win=10, weights=(0.05, 1.0, 0.05), k=9)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": {
            "train_samples": len(train_feats),
            "val_samples": len(val_feats),
            "classes": 101,
            "alias_mapping": ALIAS_MAP,
        },
        "system_profile": {
            "initial_memory_mb": round(mem_start_mb, 2),
            "memory_after_data_load_mb": round(mem_after_load_mb, 2),
            "peak_memory_mb": round(mem_after_dtw_mb, 2),
        },
        "baselines_comparison": {
            "original_baseline": {
                "config": "cosine_k5",
                "top1_accuracy": 0.0471,
                "top3_accuracy": 0.0941,
                "macro_f1": 0.0368,
                "unique_predicted_classes": 24,
            },
            "task_2_2a_champion": {
                "config": "cosine_win8_equal_groups_k7",
                "top1_accuracy": 0.0706,
                "top3_accuracy": 0.1294,
                "macro_f1": 0.0632,
                "unique_predicted_classes": 28,
            },
        },
        "champion_validation_metrics": {
            "config": "cosine_win10_velocity_dominant_k9",
            "distance_metric": champion_metric,
            "window": champion_window,
            "group_weights": champion_weights,
            "k": champion_k,
            "top1_accuracy": round(metrics["top1_accuracy"], 4),
            "top3_accuracy": round(metrics["top3_accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "unique_predicted_classes": metrics["unique_predicted_classes"],
            "most_frequently_predicted_class": top_pred_class,
            "most_frequently_predicted_count": top_pred_count,
            "prediction_distribution": pred_dist,
            "total_inference_time_sec": round(infer_time, 2),
            "latency_per_sample_ms": round(latency_per_sample_ms, 2),
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
    }

    eval_save_path = models_dir / "eval_task_2_2a_1_dtw_knn.json"
    with open(eval_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Task 2.2A.1 evaluation results to: {eval_save_path}")

    # Summary table
    t1_str = f"{metrics['top1_accuracy']*100:.2f}% ({int(metrics['top1_accuracy']*85)}/85)"
    t3_str = f"{metrics['top3_accuracy']*100:.2f}% ({int(metrics['top3_accuracy']*85)}/85)"
    f1_str = f"{metrics['macro_f1']:.4f}"
    uc_str = f"{metrics['unique_predicted_classes']} / 85"
    lat_str = f"~{latency_per_sample_ms:.1f} ms"

    print("\n" + "=" * 80)
    print(f"{'PROGRESSION BENCHMARK: TASK 2.2A.1 SIGNER GENERALIZATION':^80}")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Original Baseline':<16} | {'Sprint 2.2A':<14} | {'Task 2.2A.1 Champion':<16}")
    print("-" * 80)
    print(f"{'Top-1 Accuracy':<25} | {'4.71% (4/85)':<16} | {'7.06% (6/85)':<14} | {t1_str:<16}")
    print(f"{'Top-3 Accuracy':<25} | {'9.41% (8/85)':<16} | {'12.94% (11/85)':<14} | {t3_str:<16}")
    print(f"{'Macro F1':<25} | {'0.0368':<16} | {'0.0632':<14} | {f1_str:<16}")
    print(f"{'Unique Classes':<25} | {'24 / 85':<16} | {'28 / 85':<14} | {uc_str:<16}")
    print(f"{'Sample Latency':<25} | {'~360 ms':<16} | {'~235 ms':<14} | {lat_str:<16}")
    print("=" * 80)


if __name__ == "__main__":
    main()
