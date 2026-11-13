"""
Experiment A: Dynamic Time Warping + K-Nearest Neighbors (DTW + KNN).

Evaluates:
- Distance metrics: Euclidean vs Cosine distance.
- Neighbor counts: k in {1, 3, 5, 7}.
- Distance weighting: inverse distance weighted voting for k > 1.
- Missing landmark mask handling: zero-imputation in feature space with mask flags.
- Dimensionality reduction: Frame-level 48-dim PCA with 2x temporal stride.
- 499 canonical train vs 85 validation samples across 101 classes.
- Tracks execution runtime and RSS memory usage via psutil.
- Saves best champion model to models/isl_csltr_continuous/best_dtw_knn_model.pkl.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier, compute_dtw_distance
from scripts.eval_utils import compute_benchmark_metrics


def main() -> None:
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)

    cache_path = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "dtw_dataset.pkl"
    models_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DTW dataset cache...")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    train_data = data["train"]
    val_data = data["validation"]

    train_feats = [s["features_pca"] for s in train_data["seqs"]]
    val_feats = [s["features_pca"] for s in val_data["seqs"]]
    y_train = train_data["labels"]
    y_val = val_data["labels"]

    print(f"Loaded {len(train_feats)} train sequences and {len(val_feats)} validation sequences.")
    mem_after_load_mb = process.memory_info().rss / (1024 * 1024)

    results: Dict[str, Any] = {
        "experiment": "Experiment A: DTW + KNN",
        "description": "Dynamic Time Warping + K-Nearest Neighbors on 48-dim Frame PCA with 2x temporal stride",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": {
            "train_samples": len(train_feats),
            "val_samples": len(val_feats),
            "classes": 101,
        },
        "system_profile": {
            "initial_memory_mb": round(mem_start_mb, 2),
            "memory_after_data_load_mb": round(mem_after_load_mb, 2),
        },
        "configurations": {},
    }

    metrics_list = ["euclidean", "cosine"]
    k_list = [1, 3, 5, 7]

    best_config = ""
    best_top1 = -1.0
    best_f1 = -1.0
    best_metrics = None
    best_dist_matrix = None
    best_metric_name = "cosine"
    best_k = 5

    distance_matrices: Dict[str, np.ndarray] = {}

    for metric in metrics_list:
        print(f"\n--- Computing pairwise DTW distance matrix (metric={metric}, window=10) ---")
        clf = DTWKNNClassifier(n_neighbors=5, metric=metric, window=10, weighting="distance", num_classes=101)
        clf.fit(train_feats, y_train)

        t0 = time.time()
        dist_matrix = clf.compute_distance_matrix(val_feats, max_workers=8)
        dtw_time = time.time() - t0
        distance_matrices[metric] = dist_matrix

        mem_after_dtw_mb = process.memory_info().rss / (1024 * 1024)
        print(f"Pairwise DTW ({len(val_feats)} x {len(train_feats)} = {len(val_feats)*len(train_feats)} pairs) computed in {dtw_time:.2f}s | RSS: {mem_after_dtw_mb:.2f} MB")

        for k in k_list:
            t_infer_0 = time.time()
            y_pred, y_probs = clf.predict_from_distances(dist_matrix, k=k)
            infer_time = (time.time() - t_infer_0) + dtw_time
            latency_per_sample_ms = (infer_time / len(val_feats)) * 1000.0

            metrics = compute_benchmark_metrics(
                y_true=y_val,
                y_pred=y_pred,
                y_probs=y_probs,
                num_classes=101,
                train_time_sec=0.0,  # Non-parametric lazy model
                inference_time_sec=infer_time,
            )

            config_key = f"{metric}_k{k}"
            results["configurations"][config_key] = {
                "distance_metric": metric,
                "k": k,
                "weighting": "distance" if k > 1 else "uniform",
                "top1_accuracy": round(metrics["top1_accuracy"], 4),
                "top3_accuracy": round(metrics["top3_accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "unique_predicted_classes": metrics["unique_predicted_classes"],
                "prediction_distribution": metrics["prediction_distribution"],
                "total_inference_time_sec": round(infer_time, 2),
                "latency_per_sample_ms": round(latency_per_sample_ms, 2),
                "peak_memory_mb": round(mem_after_dtw_mb, 2),
                "per_class": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
            }

            print(
                f"Config {config_key:15s} | Top-1: {metrics['top1_accuracy']*100:5.2f}% | "
                f"Top-3: {metrics['top3_accuracy']*100:5.2f}% | "
                f"Macro F1: {metrics['macro_f1']:6.4f} | "
                f"Unique Preds: {metrics['unique_predicted_classes']:2d}/85 | "
                f"Latency: {latency_per_sample_ms:5.1f}ms/sample"
            )

            # Select best by Top-1, then Macro F1
            if (metrics["top1_accuracy"] > best_top1) or (
                metrics["top1_accuracy"] == best_top1 and metrics["macro_f1"] > best_f1
            ):
                best_top1 = metrics["top1_accuracy"]
                best_f1 = metrics["macro_f1"]
                best_config = config_key
                best_metrics = metrics
                best_dist_matrix = dist_matrix
                best_metric_name = metric
                best_k = k

    # Save champion model
    champion_clf = DTWKNNClassifier(
        n_neighbors=best_k,
        metric=best_metric_name,
        window=10,
        weighting="distance",
        num_classes=101,
    )
    champion_clf.fit(train_feats, y_train)
    champion_model_path = models_dir / "best_dtw_knn_model.pkl"
    champion_clf.save(champion_model_path)
    print(f"\nSaved Champion DTW+KNN model checkpoint ({best_config}) to {champion_model_path}")

    # Add best validation summary
    results["best_configuration"] = best_config
    results["best_validation_metrics"] = {
        "config": best_config,
        "distance_metric": best_metric_name,
        "k": best_k,
        "top1_accuracy": round(best_metrics["top1_accuracy"], 4),
        "top3_accuracy": round(best_metrics["top3_accuracy"], 4),
        "macro_f1": round(best_metrics["macro_f1"], 4),
        "unique_predicted_classes": best_metrics["unique_predicted_classes"],
        "prediction_distribution": best_metrics["prediction_distribution"],
    }

    # Save full results json
    output_path = models_dir / "eval_exp_a_dtw_knn.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Print Comparison against V2 BiGRU
    v2_eval_path = models_dir / "best_temporal_sentence_classifier_v2_evaluation.json"
    if v2_eval_path.exists():
        with open(v2_eval_path, "r", encoding="utf-8") as f:
            v2_eval = json.load(f)
        v2_top1 = v2_eval.get("best_val_accuracy", 0.0471)
        v2_top3 = v2_eval.get("validation_metrics", {}).get("top3_accuracy", 0.0588)
        v2_f1 = v2_eval.get("best_val_macro_f1", 0.0087)
        v2_unique = v2_eval.get("validation_metrics", {}).get("unique_predicted_classes", 12)

        print("\n" + "=" * 75)
        print(f"{'MODEL COMPARISON: DTW+KNN vs V2 BiGRU':^75}")
        print("=" * 75)
        print(f"{'Metric':<30} | {'V2 BiGRU Checkpoint':<20} | {'DTW + KNN (Champion)':<20}")
        print("-" * 75)
        print(f"{'Architecture':<30} | {'2-layer BiGRU+Attn':<20} | {f'DTW+KNN ({best_config})':<20}")
        print(f"{'Validation Top-1 Acc':<30} | {v2_top1*100:6.2f}%              | {best_metrics['top1_accuracy']*100:6.2f}%")
        print(f"{'Validation Top-3 Acc':<30} | {v2_top3*100:6.2f}%              | {best_metrics['top3_accuracy']*100:6.2f}%")
        print(f"{'Validation Macro F1':<30} | {v2_f1:6.4f}              | {best_metrics['macro_f1']:6.4f}")
        print(f"{'Unique Classes Predicted':<30} | {v2_unique:2d} / 85               | {best_metrics['unique_predicted_classes']:2d} / 85")
        print(f"{'Training Time':<30} | {'397.84s':<20} | {'0.00s (Lazy)':<20}")
        latency_str = f"{results['configurations'][best_config]['latency_per_sample_ms']:.1f}ms/sample"
        print(f"{'Inference Latency':<30} | {'~12.4ms/sample':<20} | {latency_str:<20}")
        print("=" * 75)


if __name__ == "__main__":
    main()
