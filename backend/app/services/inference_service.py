"""
Inference Service for Signova AI Backend.
Integrates official iSign-T5 pose-to-text model (t5-small-motion) with MediaPipe
video-to-pose extraction and official .pose serialization.
Uses lazy/deferred imports to preserve modularity and fast router startup.
"""

import logging
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings

logger = logging.getLogger("signova.backend.inference")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
POSE_EXTENSIONS = {".pose"}


class InferenceService:
    """Production service wrapping iSign T5 pose-to-text translation."""

    def __init__(self) -> None:
        self.translator: Optional[Any] = None
        self.is_loaded: bool = False
        self.device_info: str = "uninitialized"

    def load_model(self) -> None:
        """Pre-warm model and tokenizer into memory once at application startup."""
        if self.is_loaded and self.translator is not None:
            logger.info("iSign T5 model already loaded in memory.")
            return

        try:
            from ai.inference.model_loader import load_model as load_isign_model
            from ai.inference.pose_translator import PoseTranslator

            logger.info("Loading iSign T5 model checkpoint: %s", settings.ISIGN_CHECKPOINT_NAME)
            device = settings.DEVICE
            load_isign_model(device=device)

            self.translator = PoseTranslator(auto_load=False, device=device)
            self.is_loaded = True
            self.device_info = str(self.translator.device)
            logger.info("iSign T5 model loaded successfully on device: %s", self.device_info)
        except Exception as exc:
            logger.warning(
                "Continuous iSign T5 translation model unavailable: %s. "
                "Isolated recognition remains 100%% operational.",
                exc,
            )
            self.is_loaded = False
            self.translator = None
            raise RuntimeError(
                f"Continuous iSign T5 translation model is unavailable: {exc}. "
                "Connect ISIGN_MODEL_DIR in .env to enable continuous translation."
            ) from exc

    def unload_model(self) -> None:
        """Unload model and free GPU/CPU memory on shutdown."""
        if self.is_loaded:
            from ai.inference.model_loader import unload_model as unload_isign_model
            unload_isign_model()
            self.translator = None
            self.is_loaded = False
            self.device_info = "unloaded"
            logger.info("iSign T5 model unloaded.")

    def predict_file(
        self,
        file_bytes: bytes,
        filename: str,
        decoding: str = "beam",
    ) -> Dict[str, Any]:
        """
        Process uploaded video or .pose file through official iSign T5 pipeline.
        Ephemeral storage: generates temporary files and deletes on completion.
        """
        if not self.is_loaded or self.translator is None:
            logger.info("Model not pre-loaded; auto-loading...")
            self.load_model()

        from ai.pose_extraction.pose_generator import video_to_pose_file

        t_total_start = time.perf_counter()
        suffix = Path(filename).suffix.lower()
        num_beams = 4 if decoding == "beam" else 1

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / f"upload_{filename}"
            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            if suffix in POSE_EXTENSIONS:
                # Direct .pose file inference
                t_infer_start = time.perf_counter()
                result = self.translator.predict_with_metadata(
                    pose_data=tmp_path,
                    max_length=128,
                    num_beams=num_beams,
                )
                t_infer = time.perf_counter() - t_infer_start
                t_extract = 0.0
                frame_count = result.input_frames
                metadata = result.metadata or {}

            elif suffix in VIDEO_EXTENSIONS:
                # Video -> MediaPipe -> .pose file -> T5 inference
                intermediate_pose_path = Path(tmp_dir) / "extracted.pose"

                t_ext_start = time.perf_counter()
                saved_pose, _, v_meta = video_to_pose_file(
                    video_path=tmp_path,
                    output_path=intermediate_pose_path,
                )
                t_extract = time.perf_counter() - t_ext_start

                t_infer_start = time.perf_counter()
                result = self.translator.predict_with_metadata(
                    pose_data=saved_pose,
                    max_length=128,
                    num_beams=num_beams,
                )
                t_infer = time.perf_counter() - t_infer_start
                frame_count = v_meta.get("read_frames", result.input_frames)
                metadata = v_meta

            else:
                raise ValueError(
                    f"Unsupported file format '{suffix}'. Allowed: {sorted(VIDEO_EXTENSIONS | POSE_EXTENSIONS)}"
                )

        total_duration = time.perf_counter() - t_total_start

        return {
            "prediction": result.text,
            "token_ids": result.token_ids,
            "timings": {
                "extraction_time_seconds": round(t_extract, 4),
                "inference_time_seconds": round(t_infer, 4),
                "total_processing_time_seconds": round(total_duration, 4),
            },
            "device": {
                "name": result.device,
            },
            "frames": frame_count,
            "metadata": metadata,
        }

    def predict_landmarks(
        self,
        landmarks: Any,
        decoding: str = "beam",
    ) -> Dict[str, Any]:
        """
        Process client landmark sequence through official iSign T5 pipeline.
        Expected shape: (frames, 300) or list of 300-dim vectors.
        """
        if not self.is_loaded or self.translator is None:
            logger.info("Model not pre-loaded; auto-loading...")
            self.load_model()

        import numpy as np

        t_total_start = time.perf_counter()
        num_beams = 4 if decoding == "beam" else 1

        landmarks_arr = np.asarray(landmarks, dtype=np.float32)
        if landmarks_arr.size == 0 or len(landmarks_arr) == 0:
            raise ValueError("Landmark sequence is empty.")

        t_infer_start = time.perf_counter()
        result = self.translator.predict_with_metadata(
            pose_data=landmarks_arr,
            max_length=128,
            num_beams=num_beams,
        )
        t_infer = time.perf_counter() - t_infer_start
        total_duration = time.perf_counter() - t_total_start

        return {
            "prediction": result.text,
            "token_ids": result.token_ids,
            "timings": {
                "extraction_time_seconds": 0.0,
                "inference_time_seconds": round(t_infer, 4),
                "total_processing_time_seconds": round(total_duration, 4),
            },
            "device": {
                "name": result.device,
            },
            "frames": result.input_frames,
            "metadata": result.metadata or {},
        }

    def predict_video_file(
        self,
        video_bytes: bytes,
        filename: str,
        decoding: str = "beam",
    ) -> Dict[str, Any]:
        """Backward-compatible entry point for video predictions."""
        return self.predict_file(
            file_bytes=video_bytes,
            filename=filename,
            decoding=decoding,
        )


# Global singleton instance
inference_service = InferenceService()
