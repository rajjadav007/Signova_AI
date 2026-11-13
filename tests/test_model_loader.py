"""
Unit and integration tests for F-013: Model Loading Module.
Verifies loading of iSign T5 small motion checkpoint, offline tokenizer, device handling,
evaluation mode, gradients, singleton lifecycle, and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from ai.config.model_settings import ModelSettings, model_settings
from ai.inference.model import PoseEncoder, PoseT5
from ai.inference.model_loader import (
    ModelLoader,
    get_model,
    get_model_status,
    get_tokenizer,
    is_model_loaded,
    load_model,
    resolve_device,
    unload_model,
    validate_loaded_model,
)


@pytest.fixture(autouse=True)
def clean_model_state():
    """Ensure clean model loading state between tests."""
    unload_model()
    yield
    unload_model()


def test_1_checkpoint_file_validation():
    """1. Checkpoint file validation succeeds for real path and fails for invalid path."""
    model_settings.validate_checkpoint()
    model_settings.validate_tokenizer()
    model_settings.validate_all()

    invalid_settings = ModelSettings(
        model_dir=Path("D:/non_existent_model_dir"),
        checkpoint_name="t5-small-motion",
        tokenizer_dir=model_settings.tokenizer_dir,
        offline=True,
        device="cpu",
        config_data={},
    )
    with pytest.raises(FileNotFoundError, match="Model base directory not found"):
        invalid_settings.validate_checkpoint()


def test_2_tokenizer_loading():
    """2. Tokenizer loading from local directory with local_files_only."""
    tokenizer = model_settings.load_tokenizer(local_files_only=True)
    assert tokenizer is not None
    assert isinstance(tokenizer, AutoTokenizer.__base__ or object)
    assert tokenizer.vocab_size == 32100

    encoded = tokenizer("ISL pose token test", return_tensors="pt")
    decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True).strip()
    assert decoded == "ISL pose token test"


def test_3_pose_encoder_loading():
    """3. Pose encoder architecture and weights loading."""
    state_dict = torch.load(model_settings.pose_encoder_path, map_location="cpu")
    encoder = PoseEncoder(pose_dim=300, embedding_dim=768)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    # Verify layer dimensions for t5-base simple model (300 in, 768 out)
    assert encoder.encoder[0].in_features == 300
    assert encoder.encoder[0].out_features == 256
    assert encoder.encoder[3].in_features == 256
    assert encoder.encoder[3].out_features == 768

    # Forward pass test with dummy pose input: (batch_size=2, seq_length=15, pose_dim=300)
    dummy_pose = torch.randn(2, 15, 300)
    with torch.no_grad():
        out = encoder(dummy_pose)
    assert out.shape == (2, 15, 768)


def test_4_t5_model_loading_and_architecture():
    """4. T5 model loading and architecture verification."""
    # Test PoseT5 wrapper structure
    model = load_model(device="cpu")
    assert isinstance(model, PoseT5)
    assert isinstance(model.t5, T5ForConditionalGeneration)
    assert model.t5.config.d_model == 768
    assert model.t5.config.model_type == "t5"


def test_5_device_selection_and_fallback():
    """5. Device selection: CPU fallback when CUDA is not available."""
    # Forced CPU
    cpu_dev = resolve_device("cpu")
    assert cpu_dev.type == "cpu"

    # Requested CUDA when not available should gracefully fall back to CPU
    with patch("torch.cuda.is_available", return_value=False):
        dev = resolve_device("cuda")
        assert dev.type == "cpu"

    # Simulated CUDA error should also fall back to CPU
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.zeros", side_effect=RuntimeError("CUDA out of memory")):
        dev = resolve_device("cuda")
        assert dev.type == "cpu"


def test_6_model_evaluation_mode_and_gradients_disabled():
    """6. Model evaluation mode and frozen gradients."""
    model = load_model(device="cpu")
    assert model.training is False
    assert model.pose_encoder.training is False
    assert model.t5.training is False

    # Check that all parameters have requires_grad=False
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} still requires gradient"

    # validate_loaded_model should succeed
    tokenizer = get_tokenizer()
    validate_loaded_model(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        expected_pose_dim=300,
        expected_embedding_dim=768,
    )


def test_7_repeated_loading_behavior_and_singleton():
    """7. Repeated load_model calls reuse cached instance; unload releases it."""
    assert not is_model_loaded()

    m1 = load_model(device="cpu")
    assert is_model_loaded()
    assert get_model() is m1

    # Second call must return identical object without re-reading from disk
    m2 = load_model(device="cpu")
    assert m1 is m2

    status = get_model_status()
    assert status["is_loaded"] is True
    assert status["checkpoint_name"] == "t5-base"
    assert status["pose_dim"] == 300
    assert status["embedding_dim"] == 768
    assert status["device_type"] == "cpu"
    assert status["eval_mode"] is True
    assert status["requires_grad"] is False
    assert status["components"]["pose_encoder"] is True
    assert status["components"]["t5"] is True
    assert status["components"]["tokenizer"] is True

    # Unload
    unload_model()
    assert not is_model_loaded()
    assert get_model() is None
    assert get_tokenizer() is None
    assert get_model_status()["is_loaded"] is False


def test_8_missing_checkpoint_error_handling():
    """8. Missing checkpoint files produce clear FileNotFoundError without silent fallback."""
    fake_settings = ModelSettings(
        model_dir=Path("D:/non_existent_checkpoint_xyz"),
        checkpoint_name="t5-small-motion",
        tokenizer_dir=model_settings.tokenizer_dir,
        offline=True,
        device="cpu",
        config_data={},
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        load_model(settings=fake_settings)
    assert "Model base directory not found" in str(exc_info.value)


def test_validate_loaded_model_catches_invalid_dimensions():
    """Verification helper catches dimension or eval mode mismatches."""
    mock_model = MagicMock(spec=PoseT5)
    mock_model.training = False

    mock_pose_encoder = MagicMock(spec=PoseEncoder)
    mock_pose_encoder.training = False

    # Simulate wrong in_features (e.g. 200 instead of 300)
    mock_in_linear = MagicMock(in_features=200)
    mock_out_linear = MagicMock(out_features=768)
    mock_pose_encoder.encoder = [mock_in_linear, None, None, mock_out_linear]
    mock_model.pose_encoder = mock_pose_encoder

    mock_t5 = MagicMock()
    mock_t5.training = False
    mock_t5.config.d_model = 768
    mock_model.t5 = mock_t5

    with pytest.raises(ValueError, match="Pose encoder input dimension mismatch"):
        validate_loaded_model(
            model=mock_model,
            tokenizer=MagicMock(),
            device=torch.device("cpu"),
            expected_pose_dim=300,
            expected_embedding_dim=768,
        )
