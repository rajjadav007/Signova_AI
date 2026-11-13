"""
Signova AI - Model Checkpoint and Offline Settings Configuration (F-012).
Provides environment-driven, validated paths for iSign T5 checkpoints and offline tokenizer.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve project root relative to this file
AI_CONFIG_DIR = Path(__file__).resolve().parent
AI_DIR = AI_CONFIG_DIR.parent
PROJECT_ROOT = AI_DIR.parent

DEFAULT_CONFIG_JSON = PROJECT_ROOT / "models" / "config" / "model_config.json"


@dataclass
class ModelSettings:
    """Configuration settings for iSign T5 model checkpoints and tokenizer."""

    # Base directory for the 15.2 GB external model repository
    model_dir: Path
    # Specific checkpoint variant (e.g. 't5-small-motion')
    checkpoint_name: str
    # Directory containing offline tokenizer files
    tokenizer_dir: Path
    # Flag to enforce strictly local/offline loading (no remote HuggingFace calls)
    offline: bool
    # Device for inference (cpu / cuda)
    device: str
    # Loaded model config metadata
    config_data: Dict[str, Any]

    @classmethod
    def from_env(cls) -> "ModelSettings":
        """Instantiate settings from environment variables or defaults."""
        raw_model_dir = os.getenv("ISIGN_MODEL_DIR", "D:/iSign-t5-pose-to-text")
        raw_checkpoint = os.getenv("ISIGN_CHECKPOINT_NAME", "t5-base")
        raw_tokenizer_dir = os.getenv(
            "ISIGN_TOKENIZER_DIR",
            str(PROJECT_ROOT / "models" / "tokenizer" / "t5-small"),
        )
        raw_offline = os.getenv("ISIGN_OFFLINE", "true").lower() in ("true", "1", "yes")
        raw_device = os.getenv("MODEL_DEVICE", "cpu")

        # Resolve paths (support both absolute and project-relative paths)
        model_path = Path(raw_model_dir)
        if not model_path.is_absolute():
            model_path = (PROJECT_ROOT / model_path).resolve()

        tokenizer_path = Path(raw_tokenizer_dir)
        if not tokenizer_path.is_absolute():
            tokenizer_path = (PROJECT_ROOT / tokenizer_path).resolve()

        # Load model_config.json metadata if available
        config_data: Dict[str, Any] = {}
        if DEFAULT_CONFIG_JSON.exists():
            try:
                with open(DEFAULT_CONFIG_JSON, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                config_data = {}

        return cls(
            model_dir=model_path,
            checkpoint_name=raw_checkpoint,
            tokenizer_dir=tokenizer_path,
            offline=raw_offline,
            device=raw_device,
            config_data=config_data,
        )

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for the selected checkpoint variant."""
        return self.model_dir / self.checkpoint_name

    @property
    def pose_encoder_path(self) -> Path:
        """Path to pose_encoder.pt for the selected checkpoint."""
        return self.checkpoint_dir / "pose_encoder.pt"

    @property
    def t5_dir(self) -> Path:
        """Path to the t5 directory containing config.json and model.safetensors."""
        return self.checkpoint_dir / "t5"

    @property
    def t5_safetensors_path(self) -> Path:
        """Path to model.safetensors."""
        return self.t5_dir / "model.safetensors"

    @property
    def t5_config_path(self) -> Path:
        """Path to config.json."""
        return self.t5_dir / "config.json"

    @property
    def pose_dim(self) -> int:
        """Feature dimensionality expected by this checkpoint."""
        cp_info = self.config_data.get("available_checkpoints", {}).get(self.checkpoint_name, {})
        return cp_info.get("pose_dim", 600 if "motion" in self.checkpoint_name else 300)

    @property
    def use_motion(self) -> bool:
        """Whether this checkpoint incorporates temporal velocity features."""
        cp_info = self.config_data.get("available_checkpoints", {}).get(self.checkpoint_name, {})
        return cp_info.get("use_motion", "motion" in self.checkpoint_name)

    def validate_checkpoint(self) -> None:
        """
        Verify that all required files for the selected checkpoint exist.
        Raises clear FileNotFoundError with actionable message on missing assets.
        """
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model base directory not found at '{self.model_dir}'. "
                f"Check ISIGN_MODEL_DIR environment variable."
            )
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint variant directory '{self.checkpoint_name}' not found at '{self.checkpoint_dir}'. "
                f"Check ISIGN_CHECKPOINT_NAME environment variable."
            )
        if not self.pose_encoder_path.exists():
            raise FileNotFoundError(
                f"Missing pose encoder weights at '{self.pose_encoder_path}'."
            )
        if not self.t5_dir.exists():
            raise FileNotFoundError(
                f"Missing T5 subfolder at '{self.t5_dir}'."
            )
        if not self.t5_config_path.exists():
            raise FileNotFoundError(
                f"Missing T5 config at '{self.t5_config_path}'."
            )
        if not self.t5_safetensors_path.exists():
            raise FileNotFoundError(
                f"Missing T5 weights at '{self.t5_safetensors_path}'."
            )

    def validate_tokenizer(self) -> None:
        """
        Verify that offline tokenizer assets exist.
        Raises clear FileNotFoundError on missing assets.
        """
        if not self.tokenizer_dir.exists():
            raise FileNotFoundError(
                f"Tokenizer directory not found at '{self.tokenizer_dir}'. "
                f"Check ISIGN_TOKENIZER_DIR environment variable."
            )
        cfg_file = self.tokenizer_dir / "tokenizer_config.json"
        tok_file = self.tokenizer_dir / "tokenizer.json"
        sp_file = self.tokenizer_dir / "spiece.model"

        if not cfg_file.exists():
            raise FileNotFoundError(
                f"Missing tokenizer_config.json in '{self.tokenizer_dir}'."
            )
        # Fast tokenizer needs tokenizer.json; slow tokenizer needs spiece.model
        if not tok_file.exists() and not sp_file.exists():
            raise FileNotFoundError(
                f"Missing both tokenizer.json and spiece.model in '{self.tokenizer_dir}'."
            )

    def validate_all(self) -> None:
        """Validate both checkpoint and tokenizer assets."""
        self.validate_checkpoint()
        self.validate_tokenizer()

    def load_tokenizer(self, local_files_only: Optional[bool] = None):
        """
        Load the official tokenizer from the local directory.
        Defaults to local_files_only=True when offline=True.
        """
        from transformers import AutoTokenizer

        self.validate_tokenizer()
        if local_files_only is None:
            local_files_only = self.offline

        return AutoTokenizer.from_pretrained(
            str(self.tokenizer_dir),
            local_files_only=local_files_only,
        )


# Singleton instance loaded from environment
model_settings = ModelSettings.from_env()
