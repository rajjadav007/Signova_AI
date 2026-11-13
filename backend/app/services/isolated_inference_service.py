"""
Service wrapper managing the Isolated Sign Recognition inference pipeline lifecycle.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

try:
    from ai.isolated.inference.classifier import IsolatedSignRecognizer
except ImportError:
    from backend.app.ml.recognition.isolated_inference import IsolatedSignRecognizer

logger = logging.getLogger("signova.isolated_service")


class IsolatedInferenceService:
    """Singleton service managing isolated ISL gesture recognition."""

    def __init__(self) -> None:
        self.recognizer: Optional[IsolatedSignRecognizer] = None
        self.is_loaded: bool = False

    def load_model(self) -> None:
        """Instantiates IsolatedSignRecognizer and loads best weights."""
        if self.is_loaded and self.recognizer is not None:
            logger.info("Isolated inference recognizer already loaded.")
            return

        try:
            self.recognizer = IsolatedSignRecognizer()
            self.recognizer.load_model()
            self.is_loaded = True
            logger.info(
                "Isolated inference pipeline loaded successfully (Model: %s, Device: %s).",
                self.recognizer.model_type,
                self.recognizer.device,
            )
        except Exception as exc:
            logger.error("Failed to load isolated recognition model: %s", exc, exc_info=True)
            self.is_loaded = False
            raise

    def predict_landmarks(
        self,
        raw_keypoints: Union[np.ndarray, List[Any]],
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if not self.is_loaded or self.recognizer is None:
            self.load_model()
        return self.recognizer.predict_landmarks(raw_keypoints, category=category, top_k=top_k)

    def predict_video_file(
        self,
        video_bytes: bytes,
        filename: str,
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        if not self.is_loaded or self.recognizer is None:
            self.load_model()
        return self.recognizer.predict_video(
            video_bytes=video_bytes,
            filename=filename,
            category=category,
            top_k=top_k,
        )


isolated_inference_service = IsolatedInferenceService()
