"""
Unit Tests for Dynamic Time Warping + K-Nearest Neighbors ISL Sentence Recognition.

Test module: tests/test_dtw_knn_sentence_recognition.py
Task: 2.2A Verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier, compute_dtw_distance


def test_dtw_distance_zero_for_identical():
    """DTW distance between identical sequences should be approximately 0.0."""
    A = np.random.randn(30, 48).astype(np.float32)
    # Cosine distance
    d_cos = compute_dtw_distance(A, A, metric="cosine", window=10)
    assert d_cos == pytest.approx(0.0, abs=1e-5)

    # Euclidean distance
    d_euc = compute_dtw_distance(A, A, metric="euclidean", window=10)
    assert d_euc == pytest.approx(0.0, abs=1e-5)


def test_dtw_distance_variable_length():
    """DTW distance should successfully align sequences of different lengths."""
    A = np.random.randn(20, 48).astype(np.float32)
    B = np.random.randn(35, 48).astype(np.float32)

    d = compute_dtw_distance(A, B, metric="cosine", window=10)
    assert isinstance(d, float)
    assert d > 0.0
    assert np.isfinite(d)


def test_dtw_distance_sakoe_chiba_band():
    """Sakoe-Chiba constraint band should be applied without throwing index errors."""
    A = np.random.randn(15, 48).astype(np.float32)
    B = np.random.randn(40, 48).astype(np.float32)

    # window smaller than |T1 - T2| should auto-expand to min |T1 - T2|
    d = compute_dtw_distance(A, B, metric="cosine", window=5)
    assert np.isfinite(d)


def test_dtw_classifier_fit_and_predict():
    """DTWKNNClassifier should fit exemplars and return valid class predictions and probabilities."""
    np.random.seed(42)
    # 4 classes, 3 exemplars each
    X_train = []
    y_train = []
    for c in range(4):
        base = np.random.randn(25, 48).astype(np.float32)
        for _ in range(3):
            noisy = base + np.random.normal(0, 0.05, size=base.shape).astype(np.float32)
            X_train.append(noisy)
            y_train.append(c)

    clf = DTWKNNClassifier(n_neighbors=3, metric="cosine", window=5, num_classes=4)
    clf.fit(X_train, y_train)
    assert clf.is_fitted

    # Query matching class 2
    query = X_train[6] + np.random.normal(0, 0.02, size=X_train[6].shape).astype(np.float32)
    preds = clf.predict([query], max_workers=1)
    probs = clf.predict_proba([query], max_workers=1)

    assert preds.shape == (1,)
    assert preds[0] == 2
    assert probs.shape == (1, 4)
    assert probs[0, 2] > 0.5
    assert np.isclose(probs[0].sum(), 1.0)


def test_dtw_classifier_serialization():
    """DTWKNNClassifier should save to disk and reload identically."""
    np.random.seed(42)
    X_train = [np.random.randn(10, 48).astype(np.float32) for _ in range(3)]
    y_train = [0, 1, 0]

    clf = DTWKNNClassifier(n_neighbors=1, metric="euclidean", window=5, num_classes=2)
    clf.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_dtw_model.pkl"
        clf.save(save_path)
        assert save_path.exists()

        loaded_clf = DTWKNNClassifier.load(save_path)
        assert loaded_clf.is_fitted
        assert loaded_clf.n_neighbors == 1
        assert loaded_clf.metric == "euclidean"
        assert len(loaded_clf.train_seqs) == 3

        query = [np.random.randn(10, 48).astype(np.float32)]
        orig_preds = clf.predict(query, max_workers=1)
        loaded_preds = loaded_clf.predict(query, max_workers=1)
        np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_dtw_group_weighted_distance():
    """DTW distance with group weights should correctly scale spatial/velocity/acceleration groups."""
    A = np.random.randn(25, 48).astype(np.float32)
    B = np.random.randn(30, 48).astype(np.float32)

    # Identical sequences should yield ~0 distance
    d_ident = compute_dtw_distance(A, A, metric="cosine", group_weights=(1.0, 0.5, 0.2))
    assert d_ident == pytest.approx(0.0, abs=1e-5)

    # Different sequences should yield positive finite distance
    d_diff = compute_dtw_distance(A, B, metric="cosine", group_weights=(1.0, 0.5, 0.2))
    assert d_diff > 0.0
    assert np.isfinite(d_diff)


def test_dtw_group_weighted_classifier_serialization():
    """DTWKNNClassifier with group weights should serialize and restore group_weights correctly."""
    np.random.seed(42)
    X_train = [np.random.randn(15, 48).astype(np.float32) for _ in range(4)]
    y_train = [0, 1, 2, 3]

    clf = DTWKNNClassifier(
        n_neighbors=1,
        metric="cosine",
        window=8,
        num_classes=4,
        group_weights=(1.0, 0.5, 0.2),
    )
    clf.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_group_weighted_dtw.pkl"
        clf.save(save_path)
        assert save_path.exists()

        loaded_clf = DTWKNNClassifier.load(save_path)
        assert loaded_clf.is_fitted
        assert loaded_clf.group_weights == (1.0, 0.5, 0.2)

        query = [np.random.randn(15, 48).astype(np.float32)]
        orig_preds = clf.predict(query, max_workers=1)
        loaded_preds = loaded_clf.predict(query, max_workers=1)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

