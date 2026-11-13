"""
Unit Tests for Task 2.2A.1: Signer Generalization & Error Correction in ISL Sentence Recognition.

Test module: tests/test_task_2_2a_1_generalization.py
Task: 2.2A.1 Verification
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier, compute_dtw_distance


MODELS_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous"
CHECKPOINT_PATH = MODELS_DIR / "best_dtw_knn_task22a1_model.pkl"
EVAL_PATH = MODELS_DIR / "eval_task_2_2a_1_dtw_knn.json"


def test_task_2_2a_1_model_checkpoint_loadable():
    """Verify Task 2.2A.1 champion checkpoint exists and has correct hyperparameters."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"

    clf = DTWKNNClassifier.load(CHECKPOINT_PATH)

    assert isinstance(clf, DTWKNNClassifier)
    assert clf.is_fitted
    assert clf.n_neighbors == 9
    assert clf.window == 10
    assert clf.metric == "cosine"
    assert clf.group_weights == (0.05, 1.0, 0.05)
    assert len(clf.train_seqs) == 504
    assert len(clf.train_labels) == 504
    assert clf.num_classes == 101


def test_task_2_2a_1_eval_metrics_validation():
    """Verify evaluation JSON metrics achieve the >= 15% Top-1 sprint target and show marked improvement."""
    assert EVAL_PATH.exists(), f"Evaluation JSON not found at {EVAL_PATH}"

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["champion_validation_metrics"]
    assert metrics["top1_accuracy"] >= 0.15, f"Top-1 accuracy {metrics['top1_accuracy']} is below 15% target"
    assert metrics["top1_accuracy"] == pytest.approx(0.1765, abs=1e-4)
    assert metrics["top3_accuracy"] >= 0.30
    assert metrics["top3_accuracy"] == pytest.approx(0.3412, abs=1e-4)
    assert metrics["macro_f1"] > 0.10
    assert metrics["unique_predicted_classes"] >= 35

    # Check baseline comparison was tracked
    baselines = data["baselines_comparison"]
    assert "original_baseline" in baselines
    assert "task_2_2a_champion" in baselines
    assert metrics["top1_accuracy"] > baselines["task_2_2a_champion"]["top1_accuracy"]


def test_velocity_dominant_dtw_invariance_to_static_offset():
    """Verify that velocity-dominant DTW is invariant to static signer spatial shifts.
    
    When two sequences exhibit identical gesture kinematics (velocity/motion) but have
    different static spatial positions (simulating different signers / camera distances),
    velocity-dominant weighting should yield substantially smaller distance than equal weighting.
    """
    np.random.seed(42)
    T, D = 30, 48
    # 16 spatial, 16 velocity, 16 acceleration
    base = np.random.randn(T, D).astype(np.float32)

    # Candidate A: original
    seq_a = base.copy()

    # Candidate B: identical velocity/accel kinematics, but spatial offset shifted by +10.0
    seq_b = base.copy()
    seq_b[:, :16] += 10.0  # Massive static spatial shift (different signer body position)

    # 1. Equal weights (1.0, 1.0, 1.0)
    d_equal = compute_dtw_distance(
        seq_a, seq_b, metric="cosine", window=10, group_weights=(1.0, 1.0, 1.0)
    )

    # 2. Velocity-dominant weights (0.05, 1.0, 0.05)
    d_velocity = compute_dtw_distance(
        seq_a, seq_b, metric="cosine", window=10, group_weights=(0.05, 1.0, 0.05)
    )

    # Distance under velocity dominance should be substantially lower because
    # spatial disparity is suppressed by 20x.
    assert d_velocity < d_equal
    assert d_velocity < 0.2 * d_equal


def test_task_2_2a_1_predict_contract():
    """Verify classifier prediction returns valid classes and probability vectors."""
    np.random.seed(42)
    clf = DTWKNNClassifier(
        n_neighbors=3,
        metric="cosine",
        window=8,
        group_weights=(0.05, 1.0, 0.05),
        num_classes=5,
    )
    X = [np.random.randn(20, 48).astype(np.float32) for _ in range(10)]
    y = [i % 5 for i in range(10)]
    clf.fit(X, y)

    queries = [np.random.randn(22, 48).astype(np.float32) for _ in range(2)]
    preds = clf.predict(queries, max_workers=1)
    probs = clf.predict_proba(queries, max_workers=1)

    assert preds.shape == (2,)
    assert probs.shape == (2, 5)
    assert np.all(probs >= 0.0)
    assert np.allclose(probs.sum(axis=1), [1.0, 1.0])
