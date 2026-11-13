"""
Unit tests for F-012: Model Checkpoint Setup and Offline Tokenizer Verification.
"""

import json
from pathlib import Path
import pytest
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from ai.config.model_settings import ModelSettings, model_settings
from backend.app.core.config import settings as backend_settings


def test_selected_checkpoint_directory_exists():
    """1. Selected checkpoint directory exists."""
    assert model_settings.model_dir.exists(), f"Model dir {model_settings.model_dir} does not exist"
    assert model_settings.checkpoint_dir.exists(), f"Checkpoint dir {model_settings.checkpoint_dir} does not exist"
    assert model_settings.checkpoint_name == "t5-base"


def test_required_checkpoint_files_exist():
    """2. Required checkpoint files exist."""
    assert model_settings.pose_encoder_path.exists(), "pose_encoder.pt missing"
    assert model_settings.t5_dir.exists(), "t5 directory missing"
    assert model_settings.t5_config_path.exists(), "t5/config.json missing"
    assert model_settings.t5_safetensors_path.exists(), "t5/model.safetensors missing"
    assert (model_settings.t5_dir / "generation_config.json").exists(), "t5/generation_config.json missing"


def test_pose_encoder_weights_valid():
    """3. Pose encoder weights are available and valid."""
    state_dict = torch.load(model_settings.pose_encoder_path, map_location="cpu")
    assert "encoder.0.weight" in state_dict
    assert "encoder.3.weight" in state_dict
    # For t5-base simple model: in_features = 300, hidden = 256, out = 768
    assert state_dict["encoder.0.weight"].shape == (256, 300)
    assert state_dict["encoder.3.weight"].shape == (768, 256)


def test_t5_model_configuration_valid():
    """4. T5 model configuration is valid."""
    with open(model_settings.t5_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config.get("model_type") == "t5"
    assert config.get("d_model") == 768
    assert config.get("vocab_size") == 32128

    # Verify safetensors can open
    with safe_open(model_settings.t5_safetensors_path, framework="pt") as st:
        keys = list(st.keys())
        assert len(keys) > 100, f"Expected >100 tensors, found {len(keys)}"


def test_required_tokenizer_files_exist():
    """5. Required tokenizer files exist."""
    assert model_settings.tokenizer_dir.exists(), f"Tokenizer dir {model_settings.tokenizer_dir} missing"
    assert (model_settings.tokenizer_dir / "tokenizer_config.json").exists()
    assert (model_settings.tokenizer_dir / "tokenizer.json").exists()
    assert (model_settings.tokenizer_dir / "spiece.model").exists()
    assert (model_settings.tokenizer_dir / "special_tokens_map.json").exists()


def test_tokenizer_loads_locally_and_encodes_decodes():
    """6, 7, 8, 9. Tokenizer loads with local_files_only, encodes, and decodes."""
    tokenizer = model_settings.load_tokenizer(local_files_only=True)
    assert tokenizer is not None
    assert tokenizer.vocab_size == 32100

    sample_sentence = "Indian Sign Language translation using iSign T5 small motion"
    encoded = tokenizer(sample_sentence, return_tensors="pt")
    input_ids = encoded["input_ids"]
    assert input_ids.shape[0] == 1
    assert input_ids.shape[1] > 0

    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True).strip()
    assert decoded == sample_sentence


def test_missing_model_or_tokenizer_paths_produce_clear_errors():
    """10. Missing model or tokenizer paths produce clear errors."""
    invalid_settings = ModelSettings(
        model_dir=Path("D:/non_existent_model_dir_xyz"),
        checkpoint_name="t5-small-motion",
        tokenizer_dir=model_settings.tokenizer_dir,
        offline=True,
        device="cpu",
        config_data={},
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        invalid_settings.validate_checkpoint()
    assert "Model base directory not found" in str(exc_info.value)

    invalid_tokenizer_settings = ModelSettings(
        model_dir=model_settings.model_dir,
        checkpoint_name="t5-small-motion",
        tokenizer_dir=Path("D:/non_existent_tokenizer_xyz"),
        offline=True,
        device="cpu",
        config_data={},
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        invalid_tokenizer_settings.validate_tokenizer()
    assert "Tokenizer directory not found" in str(exc_info.value)


def test_backend_settings_match():
    """Verify backend settings integration."""
    assert backend_settings.ISIGN_CHECKPOINT_NAME == "t5-base"
    assert backend_settings.ISIGN_OFFLINE is True
