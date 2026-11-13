"""
Standalone Inference Module for iSign-T5 (F-015).
Performs Indian Sign Language pose-to-text translation using the official
t5-small-motion architecture, model manager (F-013), and preprocessor (F-014).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from ai.inference.model import PoseT5
from ai.inference.model_loader import (
    get_model,
    get_model_status,
    get_tokenizer,
    is_model_loaded,
    load_model,
    resolve_device,
)
from ai.preprocessing.pose_preprocessor import PreprocessedPose, preprocess_pose

logger = logging.getLogger("signova.ai.inference")

# Official generation parameters from model config and official inference.py
DEFAULT_MAX_LENGTH = 128
DEFAULT_NUM_BEAMS = 4
DEFAULT_LENGTH_PENALTY = 2.0
DEFAULT_EARLY_STOPPING = True


@dataclass(frozen=True)
class InferenceResult:
    """Structured result of pose-to-text translation."""
    text: str
    token_ids: List[int]
    latency_ms: float
    device: str
    input_frames: int
    input_shape: Tuple[int, ...]
    metadata: Optional[Dict[str, Any]] = None


class PoseTranslator:
    """
    Inference orchestrator for iSign-T5 pose-to-text translation.
    Reuses model manager from F-013 and preprocessor from F-014.
    """

    def __init__(
        self,
        auto_load: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = resolve_device(device if isinstance(device, str) else None) if not isinstance(device, torch.device) else device
        self.auto_load = auto_load

    def _get_active_model_and_tokenizer(
        self,
    ) -> Tuple[PoseT5, AutoTokenizer]:
        """Retrieve loaded model and tokenizer or auto-load if configured."""
        if not is_model_loaded():
            if not self.auto_load:
                raise RuntimeError(
                    "Model is not loaded. Call load_model() first or set auto_load=True."
                )
            logger.info("Model not loaded; auto-loading via F-013 model manager...")
            load_model(device=str(self.device))

        model = get_model()
        tokenizer = get_tokenizer()

        if model is None or tokenizer is None:
            raise RuntimeError("Model or tokenizer is unavailable after load_model().")

        return model, tokenizer

    def predict_with_metadata(
        self,
        pose_data: Union[PreprocessedPose, np.ndarray, Sequence[Any], Dict[str, Any], str, Path],
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        early_stopping: bool = DEFAULT_EARLY_STOPPING,
    ) -> InferenceResult:
        """
        Execute standalone inference on a pose sequence and return text with metadata.
        """
        start_time = time.perf_counter()

        model, tokenizer = self._get_active_model_and_tokenizer()
        model_device = next(model.parameters()).device

        # 1. Preprocess or adopt preprocessed input
        if isinstance(pose_data, PreprocessedPose):
            preprocessed = pose_data
            # Ensure tensors are on matching model device
            if preprocessed.pose_tensor.device != model_device:
                pose_tensor = preprocessed.pose_tensor.to(model_device)
                pose_length_tensor = preprocessed.pose_length.to(model_device)
            else:
                pose_tensor = preprocessed.pose_tensor
                pose_length_tensor = preprocessed.pose_length
        else:
            use_motion = getattr(model, "pose_dim", 300) == 600
            preprocessed = preprocess_pose(
                pose_data,
                use_motion=use_motion,
                device=model_device,
            )
            pose_tensor = preprocessed.pose_tensor
            pose_length_tensor = preprocessed.pose_length

        if preprocessed.actual_length == 0:
            raise ValueError("No valid pose frames available for translation.")

        # 2. Execute inference under torch.inference_mode (avoids gradients, reduces memory)
        with torch.inference_mode():
            generated_ids = model.generate(
                pose=pose_tensor,
                pose_length=pose_length_tensor,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )

        # 3. Decode output tokens with official tokenizer
        token_id_list = generated_ids[0].tolist()
        decoded_text = tokenizer.decode(
            token_id_list,
            skip_special_tokens=True,
        ).strip()

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return InferenceResult(
            text=decoded_text,
            token_ids=token_id_list,
            latency_ms=round(latency_ms, 2),
            device=str(model_device),
            input_frames=preprocessed.actual_length,
            input_shape=tuple(pose_tensor.shape),
            metadata=preprocessed.metadata,
        )

    def predict(
        self,
        pose_data: Union[PreprocessedPose, np.ndarray, Sequence[Any], Dict[str, Any], str, Path],
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
        early_stopping: bool = DEFAULT_EARLY_STOPPING,
    ) -> str:
        """Translate pose sequence and return clean English text string."""
        result = self.predict_with_metadata(
            pose_data=pose_data,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )
        return result.text

    def translate_pose_file(
        self,
        pose_path: Union[str, Path],
        max_pose_length: int = 500,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> str:
        """
        Official inference interface: Translate an iSign .pose file into English text.
        Conforms strictly to D:/iSign-t5-pose-to-text/inference/inference.py.
        """
        p = Path(pose_path)
        if not p.exists():
            raise FileNotFoundError(f"Pose file not found: {p}")

        return self.predict(
            pose_data=p,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=DEFAULT_LENGTH_PENALTY,
            early_stopping=DEFAULT_EARLY_STOPPING,
        )

    def translate_video_via_pose(
        self,
        video_path: Union[str, Path],
        output_pose_path: Optional[Union[str, Path]] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        num_beams: int = DEFAULT_NUM_BEAMS,
    ) -> InferenceResult:
        """
        End-to-end video inference via intermediate official .pose file.
        Workflow: Video -> MediaPipe -> Official .pose file -> Official Preprocessing -> T5 Model.
        """
        from ai.pose_extraction.pose_generator import video_to_pose_file

        saved_pose_path, _, meta = video_to_pose_file(
            video_path=video_path,
            output_path=output_pose_path,
        )

        result = self.predict_with_metadata(
            pose_data=saved_pose_path,
            max_length=max_length,
            num_beams=num_beams,
        )
        return result


# Default global translator instance
_default_translator = PoseTranslator(auto_load=True)


def predict_from_pose(
    pose_sequence: Union[PreprocessedPose, np.ndarray, Sequence[Any], Dict[str, Any], str, Path],
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams: int = DEFAULT_NUM_BEAMS,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    early_stopping: bool = DEFAULT_EARLY_STOPPING,
    auto_load: bool = True,
) -> str:
    """
    Convenience function: Translate prepared or raw pose sequence to English text.
    """
    if not auto_load and not is_model_loaded():
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    translator = _default_translator if auto_load else PoseTranslator(auto_load=False)
    return translator.predict(
        pose_data=pose_sequence,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
    )


def predict_with_metadata(
    pose_sequence: Union[PreprocessedPose, np.ndarray, Sequence[Any], Dict[str, Any], str, Path],
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams: int = DEFAULT_NUM_BEAMS,
    length_penalty: float = DEFAULT_LENGTH_PENALTY,
    early_stopping: bool = DEFAULT_EARLY_STOPPING,
    auto_load: bool = True,
) -> InferenceResult:
    """
    Convenience function: Translate pose sequence and return structured metadata.
    """
    if not auto_load and not is_model_loaded():
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    translator = _default_translator if auto_load else PoseTranslator(auto_load=False)
    return translator.predict_with_metadata(
        pose_data=pose_sequence,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=early_stopping,
    )


def translate_pose_file(
    pose_path: Union[str, Path],
    max_pose_length: int = 500,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams: int = DEFAULT_NUM_BEAMS,
    auto_load: bool = True,
) -> str:
    """Convenience function: Translate an iSign .pose file to English text."""
    if not auto_load and not is_model_loaded():
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    translator = _default_translator if auto_load else PoseTranslator(auto_load=False)
    return translator.translate_pose_file(
        pose_path=pose_path,
        max_pose_length=max_pose_length,
        max_length=max_length,
        num_beams=num_beams,
    )
