"""
Evaluation and benchmark metrics for Individual ISL Sign Recognition.
Computes top-k accuracy, macro F1, per-class recall/precision, and confusion matrix.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ai.isolated.config.isolated_settings import ISOLATED_CLASSES


def compute_isolated_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    target_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard multi-class evaluation metrics for isolated sign recognition.
    """
    labels = target_names or ISOLATED_CLASSES
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    results: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": report_dict,
    }

    if y_probs is not None:
        # Top-3 accuracy
        top3_preds = np.argsort(y_probs, axis=1)[:, -3:]
        top3_correct = sum(y_true[i] in top3_preds[i] for i in range(len(y_true)))
        results["top3_accuracy"] = round(float(top3_correct) / len(y_true), 4)

    return results
