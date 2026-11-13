"""
Evaluation Utilities for ISL Sentence Recognition ML Benchmark.

Provides:
- compute_metrics: Top-1 acc, Top-3 acc, Macro F1, prediction distribution, per-class PRF, confusion matrix.
- JSON-serializable output formatting.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score


def compute_benchmark_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 101,
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    N = len(y_true)

    # Top-1 Accuracy
    top1 = float(accuracy_score(y_true, y_pred))

    # Top-3 Accuracy
    if y_probs is not None and y_probs.shape[1] >= 3:
        # y_probs: (N, C)
        top3_indices = np.argsort(y_probs, axis=1)[:, -3:]  # top 3 indices
        correct_top3 = sum(y_true[i] in top3_indices[i] for i in range(N))
        top3 = float(correct_top3 / N)
    else:
        # Fallback if no probabilities (e.g. strict 1-NN): top3 equals top1
        top3 = top1

    # Macro F1 (zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Prediction Distribution
    pred_counts = Counter(int(p) for p in y_pred)
    unique_pred = len(pred_counts)

    # Per-Class Precision, Recall, F1, Support
    p_arr, r_arr, f_arr, s_arr = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    per_class = {}
    for cid in range(num_classes):
        per_class[str(cid)] = {
            "class_id": cid,
            "precision": float(p_arr[cid]),
            "recall": float(r_arr[cid]),
            "f1": float(f_arr[cid]),
            "support": int(s_arr[cid]),
            "predicted_count": int(pred_counts.get(cid, 0)),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "macro_f1": macro_f1,
        "unique_predicted_classes": unique_pred,
        "prediction_distribution": {str(k): int(v) for k, v in pred_counts.items()},
        "total_samples": N,
        "train_time_sec": float(train_time_sec),
        "inference_time_sec": float(inference_time_sec),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
