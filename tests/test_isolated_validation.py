"""
Unit and regression tests for F-031 Isolated ISL Recognition Validation.
Verifies:
- Checkpoint loading and class mapping consistency.
- Held-out test sample prediction format, validity, and top-k output.
- Rejection threshold behavior on uncertain/out-of-vocabulary inputs.
- Sub-150ms CPU inference latency target.
"""

from pathlib import Path
import numpy as np
import pytest

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    IDX_TO_CLASS,
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    NUM_CLASSES,
)
from ai.isolated.inference.classifier import IsolatedSignRecognizer


@pytest.fixture(scope="module")
def recognizer():
    """Initializes and loads the champion isolated recognizer once for tests."""
    rec = IsolatedSignRecognizer(checkpoint_path=ISOLATED_CHECKPOINT_PATH)
    rec.load_model()
    assert rec.is_loaded
    return rec


def test_isolated_champion_model_loading(recognizer):
    """Verify champion model loads correctly with expected attributes."""
    assert recognizer.is_loaded is True
    assert recognizer.model is not None
    assert recognizer.model_type == "GRUClassifierWrapper"
    assert recognizer.sequence_length == DEFAULT_SEQUENCE_LENGTH
    assert 0.0 < recognizer.rejection_threshold <= 1.0
    assert len(ISOLATED_CLASSES) == NUM_CLASSES == 10


def test_per_class_mapping_consistency():
    """Verify bidirectional index-to-class mappings."""
    for idx, name in IDX_TO_CLASS.items():
        assert CLASS_TO_IDX[name] == idx
    assert len(CLASS_TO_IDX) == NUM_CLASSES


def test_held_out_test_sample_inference(recognizer):
    """Verify inference on real held-out landmark files."""
    test_npz = Path("datasets/landmarks/test/ISL10_0012.npz")
    if not test_npz.exists():
        pytest.skip(f"Test sample {test_npz} not found.")

    data = np.load(test_npz)
    kps = data["landmarks"]  # (T, 75, 4)
    
    result = recognizer.predict_landmarks(kps, top_k=3)
    
    assert result["success"] is True
    assert result["predicted_sign"] in ISOLATED_CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["status"] in ("recognized", "uncertain")
    assert len(result["top_predictions"]) == 3
    assert result["processing_time_ms"] > 0.0

    # Top predictions must be sorted in descending confidence
    confs = [p["confidence"] for p in result["top_predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_out_of_vocabulary_rejection_noise(recognizer):
    """Verify low-confidence rejection triggers on synthetic random noise."""
    np.random.seed(42)
    noise_landmarks = np.random.randn(64, 75, 4).astype(np.float32)
    
    result = recognizer.predict_landmarks(noise_landmarks, top_k=3)
    
    assert result["success"] is True
    # Pure noise should fail the optimal rejection threshold (0.80)
    assert result["status"] == "uncertain"
    assert result["is_valid"] is False
    assert result["reason"] is not None
    assert "rejection threshold" in result["reason"]


def test_zero_landmark_input_handling(recognizer):
    """Verify zero landmark inputs are handled gracefully without exceptions."""
    zeros = np.zeros((64, 75, 4), dtype=np.float32)
    result = recognizer.predict_landmarks(zeros, top_k=3)
    
    assert result["status"] == "uncertain"
    assert result["is_valid"] is False


def test_cpu_inference_latency_benchmark(recognizer):
    """Verify average CPU inference latency is well below 150ms budget."""
    test_seq = np.random.randn(64, 75, 4).astype(np.float32)
    
    # Warmup
    _ = recognizer.predict_landmarks(test_seq)
    
    latencies = []
    for _ in range(10):
        res = recognizer.predict_landmarks(test_seq)
        latencies.append(res["processing_time_ms"])
        
    avg_latency = float(np.mean(latencies))
    assert avg_latency < 150.0, f"Latency {avg_latency:.2f}ms exceeds 150ms limit"
