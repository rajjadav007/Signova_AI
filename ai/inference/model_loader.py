"""
Model Loading Module for Signova AI (F-013).
Loads the selected pretrained iSign T5 checkpoint (t5-small-motion) and offline tokenizer.
Provides thread-safe singleton state, device selection with fallback, and validation.
"""

import gc
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

from ai.config.model_settings import ModelSettings, model_settings as default_settings
from ai.inference.model import PoseEncoder, PoseT5

logger = logging.getLogger("signova.ai.model_loader")


def resolve_device(requested_device: Optional[str] = None) -> torch.device:
    """
    Select CUDA when available and compatible; reliably fallback to CPU.
    Exposes structured selection without modifying PyTorch build.
    """
    target = (
        requested_device
        or os.getenv("MODEL_DEVICE")
        or (default_settings.device if default_settings else "cpu")
        or "cpu"
    ).strip().lower()

    if target.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                # Test allocation to guarantee driver and compute capability compatibility
                test_tensor = torch.zeros(1, device=target)
                del test_tensor
                logger.info("CUDA device '%s' verified and active.", target)
                return torch.device(target)
            except Exception as exc:
                logger.warning(
                    "CUDA device '%s' initialization failed (%s). Falling back to CPU.",
                    target,
                    exc,
                )
                return torch.device("cpu")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")

    return torch.device("cpu")


def validate_loaded_model(
    model: PoseT5,
    tokenizer: AutoTokenizer,
    device: torch.device,
    expected_pose_dim: int = 300,
    expected_embedding_dim: int = 768,
) -> None:
    """
    Verify loaded model integrity and invariants:
    - Pose encoder loaded and correct dimensions (600 in, 512 out).
    - T5 model loaded.
    - Tokenizer loaded.
    - Model in evaluation mode.
    - Gradient calculation disabled.
    - Placed on selected device.
    """
    if model is None or not isinstance(model, PoseT5):
        raise ValueError("Loaded model instance is not a valid PoseT5 model.")

    if tokenizer is None:
        raise ValueError("Tokenizer instance is None.")

    # 1. Pose encoder checks
    if not hasattr(model, "pose_encoder") or model.pose_encoder is None:
        raise ValueError("Model has missing or uninitialized pose_encoder.")

    encoder_module = model.pose_encoder.encoder
    in_linear = encoder_module[0]
    out_linear = encoder_module[3]

    if in_linear.in_features != expected_pose_dim:
        raise ValueError(
            f"Pose encoder input dimension mismatch: expected {expected_pose_dim}, "
            f"got {in_linear.in_features}."
        )

    if out_linear.out_features != expected_embedding_dim:
        raise ValueError(
            f"Pose encoder output dimension mismatch: expected {expected_embedding_dim}, "
            f"got {out_linear.out_features}."
        )

    # 2. T5 model checks
    if not hasattr(model, "t5") or model.t5 is None:
        raise ValueError("Model has missing or uninitialized T5 model.")

    if model.t5.config.d_model != expected_embedding_dim:
        raise ValueError(
            f"T5 embedding dimension mismatch: expected {expected_embedding_dim}, "
            f"got {model.t5.config.d_model}."
        )

    # 3. Evaluation mode check
    if model.training:
        raise ValueError("PoseT5 model is not in evaluation mode.")
    if model.pose_encoder.training:
        raise ValueError("Pose encoder is not in evaluation mode.")
    if model.t5.training:
        raise ValueError("T5 model is not in evaluation mode.")

    # 4. Gradients disabled check
    for name, param in model.named_parameters():
        if param.requires_grad:
            raise ValueError(f"Parameter '{name}' still has requires_grad=True.")

    # 5. Device placement check
    pose_param_device = next(model.pose_encoder.parameters()).device
    t5_param_device = next(model.t5.parameters()).device

    if pose_param_device.type != device.type:
        raise ValueError(
            f"Pose encoder placed on '{pose_param_device}', expected '{device}'."
        )
    if t5_param_device.type != device.type:
        raise ValueError(
            f"T5 model placed on '{t5_param_device}', expected '{device}'."
        )


class ModelLoader:
    """
    Thread-safe model loading manager for iSign T5 pose-to-text model.
    Prevents repeated loading and ensures clean lifecycle management.
    """

    def __init__(self) -> None:
        self._model: Optional[PoseT5] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device: Optional[torch.device] = None
        self._settings: Optional[ModelSettings] = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are currently loaded in memory."""
        with self._lock:
            return self._model is not None and self._tokenizer is not None

    def get_model(self) -> Optional[PoseT5]:
        """Get the loaded model instance without reloading."""
        return self._model

    def get_tokenizer(self) -> Optional[AutoTokenizer]:
        """Get the loaded tokenizer instance without reloading."""
        return self._tokenizer

    def get_device(self) -> Optional[torch.device]:
        """Get the device on which the model is loaded."""
        return self._device

    def load_model(
        self,
        settings: Optional[ModelSettings] = None,
        device: Optional[str] = None,
        force_reload: bool = False,
    ) -> PoseT5:
        """
        Load the pretrained iSign T5 model and tokenizer.
        Guarantees single load, local-only assets, and evaluation mode.
        """
        with self._lock:
            if not force_reload and self._model is not None and self._tokenizer is not None:
                logger.info("Model already loaded. Returning cached instance.")
                return self._model

            active_settings = settings or default_settings
            # 1. Verify required files before attempting load
            active_settings.validate_all()

            # 2. Select device
            target_device = resolve_device(device or active_settings.device)
            logger.info("Selected device for model loading: %s", target_device)

            # 3. Load local offline tokenizer
            logger.info("Loading offline tokenizer from: %s", active_settings.tokenizer_dir)
            tokenizer = active_settings.load_tokenizer(local_files_only=True)

            # 4. Load PoseT5 model using official architecture
            # Load T5 weights once directly from local directory
            t5_path = str(active_settings.t5_dir)
            logger.info("Loading T5 model from: %s", t5_path)

            base_model_name = active_settings.config_data.get(
                "available_checkpoints", {}
            ).get(active_settings.checkpoint_name, {}).get("base_model", "t5-base")

            model = PoseT5(
                model_name=base_model_name,
                pose_dim=active_settings.pose_dim,
                local_model_path=t5_path,
                local_files_only=True,
            )

            # 5. Load pose encoder weights
            pose_encoder_path = str(active_settings.pose_encoder_path)
            logger.info("Loading pose encoder weights from: %s", pose_encoder_path)
            encoder_state_dict = torch.load(pose_encoder_path, map_location="cpu")
            model.pose_encoder.load_state_dict(encoder_state_dict)

            # 6. Move model to selected device
            model = model.to(target_device)

            # 7. Set evaluation mode and freeze weights
            model.eval()
            model.requires_grad_(False)

            # 8. Validate loaded components and dimensions
            validate_loaded_model(
                model=model,
                tokenizer=tokenizer,
                device=target_device,
                expected_pose_dim=active_settings.pose_dim,
                expected_embedding_dim=model.t5.config.d_model,
            )

            # 9. Commit state
            self._model = model
            self._tokenizer = tokenizer
            self._device = target_device
            self._settings = active_settings

            logger.info(
                "iSign PoseT5 model '%s' loaded successfully on device: %s.",
                active_settings.checkpoint_name,
                target_device,
            )
            return self._model

    def unload_model(self) -> None:
        """Unload model and tokenizer from memory and release resources."""
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None

            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None

            was_cuda = self._device is not None and self._device.type == "cuda"
            self._device = None
            self._settings = None

            gc.collect()
            if was_cuda and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.info("Model unloaded successfully.")

    def get_status(self) -> Dict[str, Any]:
        """Return structured model status information."""
        with self._lock:
            if self._model is None or self._tokenizer is None:
                return {
                    "is_loaded": False,
                    "checkpoint_name": None,
                    "checkpoint_dir": None,
                    "tokenizer_dir": None,
                    "device": None,
                    "device_type": None,
                    "cuda_available": torch.cuda.is_available(),
                    "model_family": "iSign-T5",
                    "components": {
                        "pose_encoder": False,
                        "t5": False,
                        "tokenizer": False,
                    },
                }

            encoder = self._model.pose_encoder.encoder
            return {
                "is_loaded": True,
                "checkpoint_name": self._settings.checkpoint_name if self._settings else "unknown",
                "checkpoint_dir": str(self._settings.checkpoint_dir) if self._settings else None,
                "tokenizer_dir": str(self._settings.tokenizer_dir) if self._settings else None,
                "device": str(self._device),
                "device_type": self._device.type if self._device else None,
                "cuda_available": torch.cuda.is_available(),
                "model_family": "iSign-T5",
                "base_model": self._model.model_name,
                "pose_dim": encoder[0].in_features,
                "embedding_dim": encoder[3].out_features,
                "eval_mode": not self._model.training,
                "requires_grad": any(p.requires_grad for p in self._model.parameters()),
                "components": {
                    "pose_encoder": True,
                    "t5": True,
                    "tokenizer": True,
                },
            }


# Singleton manager
_loader_singleton = ModelLoader()


def load_model(
    settings: Optional[ModelSettings] = None,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> PoseT5:
    """Convenience function: load model using global singleton."""
    return _loader_singleton.load_model(
        settings=settings,
        device=device,
        force_reload=force_reload,
    )


def is_model_loaded() -> bool:
    """Convenience function: check if model is loaded."""
    return _loader_singleton.is_loaded()


def get_model_status() -> Dict[str, Any]:
    """Convenience function: retrieve structured model status."""
    return _loader_singleton.get_status()


def unload_model() -> None:
    """Convenience function: unload model."""
    _loader_singleton.unload_model()


def get_model() -> Optional[PoseT5]:
    """Convenience function: get loaded model."""
    return _loader_singleton.get_model()


def get_tokenizer() -> Optional[AutoTokenizer]:
    """Convenience function: get loaded tokenizer."""
    return _loader_singleton.get_tokenizer()
