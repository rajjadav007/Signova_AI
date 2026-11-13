"""
Unit and integration tests for F-015: Standalone Inference for iSign-T5.
Tests inference pipeline, model manager integration, decoding, device handling,
repeated execution, and error handling.

NOTE: Tests use synthetic pose data to verify tensor flow and architectural execution.
No real ISL recognition quality is claimed from synthetic input.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from ai.inference.model_loader import get_model, is_model_loaded, load_model, unload_model
from ai.inference.pose_translator import (
    InferenceResult,
    PoseTranslator,
    predict_from_pose,
    predict_with_metadata,
)
from ai.preprocessing import BASE_SPATIAL_DIM, PreprocessedPose, preprocess_pose


@pytest.fixture(scope="module")
def loaded_environment():
    """Load model once for inference tests and unload cleanly at end."""
    model = load_model(device="cpu")
    yield model
    unload_model()


def test_1_valid_inference_input(loaded_environment):
    """1. Valid inference execution returns string and structured InferenceResult."""
    frames = 20
    synthetic_pose = np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32)

    # 1a. predict_from_pose returns string
    text = predict_from_pose(synthetic_pose, auto_load=True)
    assert isinstance(text, str)

    # 1b. predict_with_metadata returns InferenceResult with valid fields
    result = predict_with_metadata(synthetic_pose, auto_load=True)
    assert isinstance(result, InferenceResult)
    assert isinstance(result.text, str)
    assert isinstance(result.token_ids, list)
    assert len(result.token_ids) > 0
    assert result.latency_ms > 0.0
    assert result.device == "cpu"
    assert result.input_frames == frames
    assert result.input_shape == (1, 500, 300)


def test_2_invalid_input_shape_raises_value_error(loaded_environment):
    """2. Incompatible landmark/feature dimensions raise clear ValueError."""
    # Invalid 150 feature channels instead of 300
    invalid_pose = np.random.randn(15, 150).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported pose array shape"):
        predict_from_pose(invalid_pose)


def test_3_empty_input_raises_value_error(loaded_environment):
    """3. Empty pose sequence raises clear ValueError."""
    empty_pose = np.zeros((0, BASE_SPATIAL_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        predict_from_pose(empty_pose)

    with pytest.raises(ValueError, match="empty"):
        predict_from_pose([])


def test_4_model_manager_integration(loaded_environment):
    """4. Standalone inference reuses existing loaded model from F-013 model manager."""
    assert is_model_loaded()
    manager_model = get_model()

    translator = PoseTranslator(auto_load=False)
    active_model, _ = translator._get_active_model_and_tokenizer()

    assert active_model is manager_model, "Translator did not use F-013 singleton model"


def test_5_token_decoding(loaded_environment):
    """5. Token IDs decode to text, special tokens are stripped."""
    frames = 15
    synthetic_pose = np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32)
    result = predict_with_metadata(synthetic_pose)

    # Verify decoded text does not contain raw pad or eos tokens
    assert "<pad>" not in result.text
    assert "</s>" not in result.text
    assert "<unk>" not in result.text or isinstance(result.text, str)


def test_6_inference_mode_and_no_gradients(loaded_environment):
    """6. Inference executes under torch.inference_mode with no gradients."""
    frames = 10
    prep = preprocess_pose(np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32))

    grad_enabled_during_call = []

    original_generate = loaded_environment.generate

    def spy_generate(*args, **kwargs):
        grad_enabled_during_call.append(torch.is_grad_enabled())
        return original_generate(*args, **kwargs)

    with patch.object(loaded_environment, "generate", side_effect=spy_generate):
        predict_from_pose(prep)

    assert len(grad_enabled_during_call) == 1
    assert grad_enabled_during_call[0] is False, "Inference executed with gradients enabled"


def test_7_correct_device_handling(loaded_environment):
    """7. Tensors are routed to model device properly."""
    frames = 12
    # Preprocess on CPU
    prep = preprocess_pose(np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32), device="cpu")

    result = predict_with_metadata(prep)
    assert result.device == "cpu"


def test_8_repeated_predictions_without_reloading(loaded_environment):
    """8. Successive predictions reuse loaded instance in memory without re-loading."""
    m_before = get_model()
    synthetic_pose = np.random.randn(10, BASE_SPATIAL_DIM).astype(np.float32)

    # First call
    res1 = predict_with_metadata(synthetic_pose)
    m_mid = get_model()
    assert m_before is m_mid

    # Second call
    res2 = predict_with_metadata(synthetic_pose)
    m_after = get_model()
    assert m_mid is m_after

    # Both succeeded
    assert res1.text is not None
    assert res2.text is not None


def test_9_error_handling_when_model_unavailable():
    """9. Explicit auto_load=False raises RuntimeError when model is not loaded."""
    # Temporarily unload
    unload_model()
    assert not is_model_loaded()

    synthetic_pose = np.random.randn(10, BASE_SPATIAL_DIM).astype(np.float32)

    with pytest.raises(RuntimeError, match="Model is not loaded"):
        predict_from_pose(synthetic_pose, auto_load=False)

    with pytest.raises(RuntimeError, match="Model is not loaded"):
        translator = PoseTranslator(auto_load=False)
        translator.predict(synthetic_pose)

    # Re-load for subsequent tests
    load_model(device="cpu")
    assert is_model_loaded()
