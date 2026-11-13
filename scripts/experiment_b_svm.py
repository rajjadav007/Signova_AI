"""
Experiment B: Support Vector Machine (SVM with RBF Kernel) for ISL Sentence Recognition.

Task: 2.2B — Evaluate RBF SVM on 7-segment Temporal Pyramid features (9,492 dims).
Compares against:
- V2 BiGRU + Attention
- DTW + KNN cosine K=5

Evaluates:
- Feature sets: PCA_128, PCA_64, StandardScaled_9492.
- Hyperparameter grid:
  * C in {0.1, 1.0, 10.0, 100.0}
  * gamma in {'scale', 'auto', 0.001}
  * class_weight in {None, 'balanced'}
- Probability estimation: Softmax over continuous margin scores from decision_function.
- Data hygiene: Train 499 canonical, Validation 85 (Signer 7). Test strictly untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import psutil
from scipy.special import softmax
from sklearn.svm import SVC

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import compute_benchmark_metrics


def evaluate_svm_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    gamma: Any,
    class_weight: Any,
    feat_name: str,
    num_classes: int = 101,
) -> Dict[str, Any]:
    """Train and evaluate a single SVM configuration."""
    clf = SVC(
        C=C,
        gamma=gamma,
        kernel="rbf",
        class_weight=class_weight,
        probability=False,
        random_state=42,
    )

    t0_train = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0_train

    t0_infer = time.time()
    y_pred = clf.predict(X_val)
    dec_scores = clf.decision_function(X_val)  # (N, C_train)
    infer_time = time.time() - t0_infer

    # Softmax-normalized margin scores for exact Top-3 ranking
    probs_seen = softmax(dec_scores, axis=1)

    # Map seen class probabilities to full num_classes space
    N = len(y_val)
    y_probs = np.zeros((N, num_classes), dtype=np.float32)
    for col_idx, cls_id in enumerate(clf.classes_):
        if cls_id < num_classes:
            y_probs[:, cls_id] = probs_seen[:, col_idx]

    metrics = compute_benchmark_metrics(
        y_true=y_val,
        y_pred=y_pred,
        y_probs=y_probs,
        num_classes=num_classes,
        train_time_sec=train_time,
        inference_time_sec=infer_time,
    )

    return {
        "clf": clf,
        "metrics": metrics,
        "params": {
            "feature_set": feat_name,
            "C": C,
            "gamma": gamma,
            "class_weight": class_weight,
        },
    }


def main() -> None:
    process = psutil.Process()
    mem_start_mb = process.memory_info().rss / (1024 * 1024)

    cache_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache"
    models_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prepared 7-segment temporal pyramid features (9,492 dims)...")
    train_npz = np.load(cache_dir / "pyramid_train.npz")
    val_npz = np.load(cache_dir / "pyramid_validation.npz")

    y_train = train_npz["y"]
    y_val = val_npz["y"]

    feature_sets = {
        "PCA_128": (train_npz["X_pca128"], val_npz["X_pca128"]),
        "PCA_64": (train_npz["X_pca64"], val_npz["X_pca64"]),
        "StandardScaled_9492": (train_npz["X_scaled"], val_npz["X_scaled"]),
    }

    print(f"Loaded {len(y_train)} train samples, {len(y_val)} validation samples.")
    print(f"Feature set shapes: PCA_128={feature_sets['PCA_128'][0].shape}, StandardScaled={feature_sets['StandardScaled_9492'][0].shape}")

    # Targeted hyperparameter grid
    param_grid = [
        # (feat_name, C, gamma, class_weight)
        ("PCA_128", 0.1, "scale", None),
        ("PCA_128", 1.0, "scale", None),
        ("PCA_128", 10.0, "scale", None),
        ("PCA_128", 10.0, "scale", "balanced"),
        ("PCA_128", 100.0, "scale", None),
        ("PCA_128", 10.0, "auto", None),
        ("PCA_128", 10.0, 0.001, None),

        ("PCA_64", 1.0, "scale", None),
        ("PCA_64", 10.0, "scale", None),
        ("PCA_64", 10.0, "scale", "balanced"),
        ("PCA_64", 100.0, "scale", None),

        ("StandardScaled_9492", 0.1, "scale", None),
        ("StandardScaled_9492", 1.0, "scale", None),
        ("StandardScaled_9492", 10.0, "scale", None),
        ("StandardScaled_9492", 10.0, "scale", "balanced"),
        ("StandardScaled_9492", 100.0, "scale", None),
        ("StandardScaled_9492", 10.0, 0.001, None),
    ]

    results: Dict[str, Any] = {
        "experiment": "Experiment B: SVM (RBF Kernel)",
        "description": "Support Vector Machine with RBF kernel on 7-segment Temporal Pyramid features (9,492 dims)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_split": {
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "classes": 101,
        },
        "configurations": {},
    }

    best_top1 = -1.0
    best_f1 = -1.0
    best_config_name = ""
    best_model = None
    best_eval = None

    print(f"\nEvaluating {len(param_grid)} SVM configurations across validation split...")
    for feat_name, C, gamma, cw in param_grid:
        X_tr, X_v = feature_sets[feat_name]
        cfg_name = f"{feat_name}_C{C}_g{gamma}_cw{cw}"

        out = evaluate_svm_config(
            X_train=X_tr,
            y_train=y_train,
            X_val=X_v,
            y_val=y_val,
            C=C,
            gamma=gamma,
            class_weight=cw,
            feat_name=feat_name,
            num_classes=101,
        )

        m = out["metrics"]
        latency_ms = (m["inference_time_sec"] / len(y_val)) * 1000.0
        peak_rss_mb = process.memory_info().rss / (1024 * 1024)

        results["configurations"][cfg_name] = {
            "params": out["params"],
            "top1_accuracy": round(m["top1_accuracy"], 4),
            "top3_accuracy": round(m["top3_accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "unique_predicted_classes": m["unique_predicted_classes"],
            "prediction_distribution": m["prediction_distribution"],
            "train_time_sec": round(m["train_time_sec"], 3),
            "inference_time_sec": round(m["inference_time_sec"], 3),
            "latency_per_sample_ms": round(latency_ms, 2),
            "peak_memory_mb": round(peak_rss_mb, 2),
            "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"],
        }

        print(
            f"Config {cfg_name:44s} | Top-1: {m['top1_accuracy']*100:5.2f}% | "
            f"Top-3: {m['top3_accuracy']*100:5.2f}% | "
            f"Macro F1: {m['macro_f1']:6.4f} | "
            f"Unique: {m['unique_predicted_classes']:2d}/85 | "
            f"Train: {m['train_time_sec']:5.2f}s"
        )

        # Prioritize Top-1, then Top-3, then Macro F1
        if (m["top1_accuracy"] > best_top1) or (
            m["top1_accuracy"] == best_top1 and m["top3_accuracy"] > (best_eval["top3_accuracy"] if best_eval else -1.0)
        ) or (
            m["top1_accuracy"] == best_top1 and m["top3_accuracy"] == (best_eval["top3_accuracy"] if best_eval else -1.0) and m["macro_f1"] > best_f1
        ):
            best_top1 = m["top1_accuracy"]
            best_f1 = m["macro_f1"]
            best_config_name = cfg_name
            best_model = out["clf"]
            best_eval = m

    results["best_configuration"] = best_config_name
    results["best_validation_metrics"] = {
        "config": best_config_name,
        "params": results["configurations"][best_config_name]["params"],
        "top1_accuracy": round(best_eval["top1_accuracy"], 4),
        "top3_accuracy": round(best_eval["top3_accuracy"], 4),
        "macro_f1": round(best_eval["macro_f1"], 4),
        "unique_predicted_classes": best_eval["unique_predicted_classes"],
        "prediction_distribution": best_eval["prediction_distribution"],
        "train_time_sec": results["configurations"][best_config_name]["train_time_sec"],
        "latency_per_sample_ms": results["configurations"][best_config_name]["latency_per_sample_ms"],
    }

    # Save checkpoint and results
    eval_file = models_dir / "eval_exp_b_svm.json"
    ckpt_file = models_dir / "best_svm_checkpoint.joblib"

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump({
        "model": best_model,
        "best_config": best_config_name,
        "params": results["configurations"][best_config_name]["params"],
        "feature_set": results["configurations"][best_config_name]["params"]["feature_set"],
        "metrics": results["best_validation_metrics"],
    }, ckpt_file)

    print(f"\nSaved Best SVM checkpoint ({best_config_name}) to {ckpt_file}")
    print(f"Results saved to {eval_file}")

    # Load baseline metrics for direct comparison
    v2_eval_path = models_dir / "best_temporal_sentence_classifier_v2_evaluation.json"
    dtw_eval_path = models_dir / "eval_exp_a_dtw_knn.json"

    v2_top1, v2_top3, v2_f1, v2_uniq, v2_time, v2_lat = 0.0471, 0.0588, 0.0087, 12, "397.84s", "12.4ms"
    if v2_eval_path.exists():
        with open(v2_eval_path, "r", encoding="utf-8") as f:
            v2_d = json.load(f)
            v2_top1 = v2_d.get("best_val_accuracy", 0.0471)
            v2_top3 = v2_d.get("validation_metrics", {}).get("top3_accuracy", 0.0588)
            v2_f1 = v2_d.get("best_val_macro_f1", 0.0087)
            v2_uniq = v2_d.get("validation_metrics", {}).get("unique_predicted_classes", 12)

    dtw_top1, dtw_top3, dtw_f1, dtw_uniq, dtw_time, dtw_lat = 0.0471, 0.0941, 0.0368, 24, "0.00s", "356.7ms"
    if dtw_eval_path.exists():
        with open(dtw_eval_path, "r", encoding="utf-8") as f:
            dtw_d = json.load(f)
            best_dtw = dtw_d.get("best_validation_metrics", {})
            dtw_top1 = best_dtw.get("top1_accuracy", 0.0471)
            dtw_top3 = best_dtw.get("top3_accuracy", 0.0941)
            dtw_f1 = best_dtw.get("macro_f1", 0.0368)
            dtw_uniq = best_dtw.get("unique_predicted_classes", 24)
            best_dtw_cfg = dtw_d.get("best_configuration", "cosine_k5")
            dtw_lat_val = dtw_d.get("configurations", {}).get(best_dtw_cfg, {}).get("latency_per_sample_ms", 356.7)
            dtw_lat = f"{dtw_lat_val:.1f}ms"

    best_svm_lat = f"{results['best_validation_metrics']['latency_per_sample_ms']:.1f}ms"
    best_svm_train = f"{results['best_validation_metrics']['train_time_sec']:.2f}s"

    print("\n" + "=" * 95)
    print(f"{'THREE-WAY MODEL BENCHMARK: V2 BiGRU vs DTW+KNN vs SVM (RBF)':^95}")
    print("=" * 95)
    print(f"{'Metric':<25} | {'V2 BiGRU+Attn':<18} | {'DTW+KNN (cos_k5)':<18} | {'SVM RBF (Champion)':<20}")
    print("-" * 95)
    print(f"{'Validation Top-1 Acc':<25} | {v2_top1*100:6.2f}%            | {dtw_top1*100:6.2f}%            | {best_eval['top1_accuracy']*100:6.2f}%")
    print(f"{'Validation Top-3 Acc':<25} | {v2_top3*100:6.2f}%            | {dtw_top3*100:6.2f}%            | {best_eval['top3_accuracy']*100:6.2f}%")
    print(f"{'Validation Macro F1':<25} | {v2_f1:6.4f}            | {dtw_f1:6.4f}            | {best_eval['macro_f1']:6.4f}")
    print(f"{'Unique Classes Pred':<25} | {v2_uniq:2d} / 85             | {dtw_uniq:2d} / 85             | {best_eval['unique_predicted_classes']:2d} / 85")
    print(f"{'Training Time':<25} | {v2_time:<18} | {dtw_time:<18} | {best_svm_train:<20}")
    print(f"{'Inference Latency':<25} | {v2_lat:<18} | {dtw_lat:<18} | {best_svm_lat:<20}")
    print(f"{'Best Configuration':<25} | {'2-layer BiGRU':<18} | {'cosine_k5':<18} | {best_config_name:<20}")
    print("=" * 95)


if __name__ == "__main__":
    main()
