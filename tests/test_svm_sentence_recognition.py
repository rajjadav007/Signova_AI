"""
Unit Tests for SVM ISL Sentence Recognition and 7-Segment Temporal Pyramid Pooling.

Test module: tests/test_svm_sentence_recognition.py
Task: 2.2B Verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pytest
from scipy.special import softmax
from sklearn.svm import SVC

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_ml_features import extract_temporal_pyramid


def test_extract_temporal_pyramid_shape():
    """7-segment temporal pyramid should yield exactly 9,492 dimensions."""
    T = 60
    D = 678
    seq = np.random.randn(T, D).astype(np.float32)

    pyr = extract_temporal_pyramid(seq)
    assert pyr.ndim == 1
    # 7 segments * 2 statistics (mean, std) * 678 features = 9,492
    assert pyr.shape[0] == 9492
    assert pyr.dtype == np.float32
    assert np.all(np.isfinite(pyr))


def test_extract_temporal_pyramid_short_sequence():
    """Short sequences (T < 4) should fall back safely without crashing."""
    T = 2
    D = 678
    seq = np.random.randn(T, D).astype(np.float32)

    pyr = extract_temporal_pyramid(seq)
    assert pyr.shape[0] == 9492
    assert np.all(np.isfinite(pyr))


def test_svm_decision_function_ranking():
    """Softmax over SVM decision_function should produce valid probability distribution for Top-3."""
    np.random.seed(42)
    # 5 classes, 10 samples each
    X = np.random.randn(50, 64).astype(np.float32)
    y = np.repeat(np.arange(5), 10)

    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=False, random_state=42)
    clf.fit(X, y)

    scores = clf.decision_function(X[:5])
    probs = softmax(scores, axis=1)

    assert probs.shape == (5, 5)
    assert np.allclose(probs.sum(axis=1), 1.0)
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    assert top3_idx.shape == (5, 3)


def test_svm_checkpoint_serialization():
    """SVM model and metadata should save to joblib and restore identically."""
    X = np.random.randn(20, 32).astype(np.float32)
    y = np.array([0, 1] * 10)

    clf = SVC(kernel="rbf", C=5.0, gamma="scale", probability=False, random_state=42)
    clf.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_svm.joblib"
        joblib.dump({
            "model": clf,
            "feature_set": "PCA_32",
            "best_config": "PCA_32_C5.0",
        }, ckpt_path)

        loaded = joblib.load(ckpt_path)
        loaded_model = loaded["model"]
        assert loaded["feature_set"] == "PCA_32"

        orig_preds = clf.predict(X[:5])
        loaded_preds = loaded_model.predict(X[:5])
        np.testing.assert_array_equal(orig_preds, loaded_preds)
