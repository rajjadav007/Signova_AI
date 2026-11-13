"""
Experiment C: XGBoost (Extreme Gradient Boosting on Temporal Pyramid).

Evaluates:
- Temporal Pyramid features (5,424 dims) preserving early, middle, and late movement phases.
- Dimensionality options: Raw scaled (5,424 dims) and PCA-128.
- Hyperparameter tuning: max_depth in {3, 4}, learning_rate in {0.05, 0.1}, n_estimators in {100, 150},
  colsample_bytree in {0.3, 0.5}, sample weights for class balance.
- Prevents excessive complexity to resist overfitting on 499 training samples.
- Validation split (Signer 7, 85 samples) across 101 classes.
- Saves best XGBoost model checkpoint and validation report.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import xgboost as xgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_utils import compute_benchmark_metrics


def compute_sample_weights(y: np.ndarray, num_classes: int = 101, smoothing: float = 0.5) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights_per_class = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights_per_class[c] = (total / (counts[c] ** smoothing))
    # Normalize weights so mean is 1.0
    weights_per_class = weights_per_class / np.mean(weights_per_class[weights_per_class > 0])
    return np.array([weights_per_class[label] for label in y], dtype=np.float32)


def evaluate_xgb_config(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_classes: np.ndarray,
    sample_weights: np.ndarray,
    max_depth: int,
    learning_rate: float,
    n_estimators: int,
    colsample_bytree: float,
    subsample: float,
    feat_name: str,
) -> Dict[str, Any]:
    n_unique_classes = len(train_classes)
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_unique_classes,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    t0_train = time.time()
    clf.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    train_time = time.time() - t0_train

    t0_infer = time.time()
    # probs_encoded: (N, n_unique_classes)
    probs_encoded = clf.predict_proba(X_val)

    # Map back to full 101-class space
    N = len(y_val)
    y_probs = np.zeros((N, 101), dtype=np.float32)
    for local_idx, orig_cid in enumerate(train_classes):
        if orig_cid < 101:
            y_probs[:, orig_cid] = probs_encoded[:, local_idx]

    y_pred = np.argmax(y_probs, axis=1)
    infer_time = time.time() - t0_infer

    metrics = compute_benchmark_metrics(
        y_true=y_val,
        y_pred=y_pred,
        y_probs=y_probs,
        num_classes=101,
        train_time_sec=train_time,
        inference_time_sec=infer_time,
    )

    return {
        "clf": clf,
        "metrics": metrics,
        "params": {
            "feature_set": feat_name,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
        },
    }


def main() -> None:
    cache_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache"
    models_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prepared pyramid features for XGBoost...")
    train_npz = np.load(cache_dir / "pyramid_train.npz")
    val_npz = np.load(cache_dir / "pyramid_validation.npz")

    y_train = train_npz["y"]
    y_val = val_npz["y"]

    train_classes = np.sort(np.unique(y_train))
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    y_train_encoded = np.array([class_to_idx[c] for c in y_train], dtype=np.int64)

    feature_sets = {
        "PCA_128": (train_npz["X_pca128"], val_npz["X_pca128"]),
        "StandardScaled_5424": (train_npz["X_scaled"], val_npz["X_scaled"]),
    }

    sample_weights = compute_sample_weights(y_train, num_classes=101, smoothing=0.5)

    param_grid = [
        # (feat_name, max_depth, lr, n_estimators, colsample_bytree, subsample)
        ("PCA_128", 3, 0.1, 100, 0.5, 0.8),
        ("PCA_128", 4, 0.05, 150, 0.5, 0.8),
        ("PCA_128", 3, 0.05, 100, 0.3, 0.8),
        ("StandardScaled_5424", 3, 0.1, 100, 0.3, 0.8),
        ("StandardScaled_5424", 3, 0.05, 100, 0.2, 0.8),
        ("StandardScaled_5424", 4, 0.05, 150, 0.3, 0.8),
    ]

    results: Dict[str, Any] = {
        "experiment": "Experiment C: XGBoost",
        "description": "Extreme Gradient Boosting on Temporal Pyramid features",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": {},
    }

    best_top1 = -1.0
    best_config_name = ""
    best_model = None
    best_eval = None

    print(f"\nEvaluating {len(param_grid)} XGBoost configurations...")
    for feat_name, md, lr, n_est, col, sub in param_grid:
        X_tr, X_v = feature_sets[feat_name]
        cfg_name = f"{feat_name}_d{md}_lr{lr}_n{n_est}_col{col}"

        out = evaluate_xgb_config(
            X_train=X_tr,
            y_train_encoded=y_train_encoded,
            X_val=X_v,
            y_val=y_val,
            train_classes=train_classes,
            sample_weights=sample_weights,
            max_depth=md,
            learning_rate=lr,
            n_estimators=n_est,
            colsample_bytree=col,
            subsample=sub,
            feat_name=feat_name,
        )

        m = out["metrics"]
        results["configurations"][cfg_name] = {
            "params": out["params"],
            "top1_accuracy": m["top1_accuracy"],
            "top3_accuracy": m["top3_accuracy"],
            "macro_f1": m["macro_f1"],
            "unique_predicted_classes": m["unique_predicted_classes"],
            "prediction_distribution": m["prediction_distribution"],
            "train_time_sec": m["train_time_sec"],
            "inference_time_sec": m["inference_time_sec"],
            "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"],
        }

        print(
            f"Config {cfg_name:38s} | Top-1: {m['top1_accuracy']*100:5.2f}% | "
            f"Top-3: {m['top3_accuracy']*100:5.2f}% | "
            f"Macro F1: {m['macro_f1']:6.4f} | "
            f"Unique Preds: {m['unique_predicted_classes']:2d}/85 | "
            f"Train: {m['train_time_sec']:.1f}s"
        )

        if m["top1_accuracy"] > best_top1:
            best_top1 = m["top1_accuracy"]
            best_config_name = cfg_name
            best_model = out["clf"]
            best_eval = m

    results["best_configuration"] = best_config_name
    results["best_validation_metrics"] = {
        "config": best_config_name,
        "top1_accuracy": best_eval["top1_accuracy"],
        "top3_accuracy": best_eval["top3_accuracy"],
        "macro_f1": best_eval["macro_f1"],
        "unique_predicted_classes": best_eval["unique_predicted_classes"],
        "prediction_distribution": best_eval["prediction_distribution"],
    }

    eval_file = models_dir / "eval_exp_c_xgboost.json"
    ckpt_file = models_dir / "best_xgboost_checkpoint.json"

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_model.save_model(str(ckpt_file))

    print(f"\nExperiment C completed! Best: {best_config_name} (Top-1: {best_top1*100:.2f}%)")
    print(f"Results saved to {eval_file}")
    print(f"Checkpoint saved to {ckpt_file}")


if __name__ == "__main__":
    main()
