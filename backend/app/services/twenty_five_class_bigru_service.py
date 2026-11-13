"""
Signova AI - 25-Class BiGRU Attention Sentence Recognition Service.
Backend service wrapping TwentyFiveClassSequenceRecognizer for live webcam & video inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("signova.twenty_five_class_bigru_service")


class TwentyFiveClassBiGRUService:
    """
    Singleton service managing 25-Class ISL BiGRU Attention Sentence Recognition.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.model_dir = (
            self.project_root
            / "models"
            / "isl_csltr_continuous"
            / "twenty_five_class_sequence"
        )
        if not (self.model_dir / "best_model.pt").exists():
            root_model_dir = self.project_root.parent / "models" / "isl_csltr_continuous" / "twenty_five_class_sequence"
            if (root_model_dir / "best_model.pt").exists():
                self.model_dir = root_model_dir

        self.checkpoint_path = self.model_dir / "best_model.pt"
        self.is_loaded: bool = False
        self._load_lock = threading.Lock()
        self._recognizer: Optional[Any] = None

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load 25-class BiGRU model and configuration into memory."""
        with self._load_lock:
            checkpoint_path = self.checkpoint_path
            class_mapping_path = self.model_dir / "class_mapping.json"

            if not checkpoint_path.exists() or not class_mapping_path.exists():
                logger.warning(
                    "25-Class BiGRU artifacts not found in %s. Inference will not be available until trained.",
                    self.model_dir,
                )
                return

            try:
                from ai.continuous.twenty_five_class_sequence_recognizer import TwentyFiveClassSequenceRecognizer
                self._recognizer = TwentyFiveClassSequenceRecognizer(model_dir=self.model_dir)
                self.is_loaded = True
                logger.info(
                    "TwentyFiveClassBiGRUService successfully initialized with %d classes.",
                    self._recognizer.num_classes,
                )
            except Exception as e:
                logger.error("Failed to load TwentyFiveClassSequenceRecognizer: %s", e, exc_info=True)
                self.is_loaded = False

    def get_classes(self) -> List[Dict[str, Any]]:
        """Return the list of 25 supported sentence classes."""
        if not self.is_loaded:
            self.load_artifacts()
        if self._recognizer is not None:
            return self._recognizer.classes
        return []

    def predict_video(
        self,
        video_bytes: bytes,
        filename: str = "webcam.webm",
        confidence_threshold: float = 0.25,
        min_frames: int = 5,
    ) -> Dict[str, Any]:
        """
        Process recorded live webcam video or uploaded video through MediaPipe landmark extraction
        and TwentyFiveClassBiGRUAttentionModel sequence inference.
        """
        if not self.is_loaded:
            self.load_artifacts()

        if not self.is_loaded or self._recognizer is None:
            return {
                "success": False,
                "error": "25-Class BiGRU model is not loaded.",
                "frame_count": 0,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "bigru_25class",
            }

        suffix = Path(filename).suffix if Path(filename).suffix else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        frames_bgr: List[np.ndarray] = []
        try:
            cap = cv2.VideoCapture(str(tmp_path))
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frames_bgr.append(frame)
            cap.release()
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        if len(frames_bgr) < min_frames:
            return {
                "success": False,
                "error": f"Video contains only {len(frames_bgr)} frames (minimum required: {min_frames}).",
                "frame_count": len(frames_bgr),
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "bigru_25class",
            }

        return self._recognizer.predict_frames(
            frames_bgr,
            confidence_threshold=confidence_threshold,
            min_frames=min_frames,
        )

    def predict_landmarks(
        self,
        landmarks_sequence: List[List[List[float]]],
        confidence_threshold: float = 0.25,
        min_frames: int = 5,
    ) -> Dict[str, Any]:
        """
        Run inference directly from landmark sequence (T, 75, 4).
        """
        if not self.is_loaded:
            self.load_artifacts()

        if not self.is_loaded or self._recognizer is None:
            return {
                "success": False,
                "error": "25-Class BiGRU model is not loaded.",
                "frame_count": 0,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "bigru_25class",
            }

        landmarks_arr = np.array(landmarks_sequence, dtype=np.float32)
        if len(landmarks_arr.shape) != 3 or landmarks_arr.shape[1] != 75 or landmarks_arr.shape[2] != 4:
            return {
                "success": False,
                "error": f"Invalid landmark shape {landmarks_arr.shape}. Expected (T, 75, 4).",
                "frame_count": 0,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "bigru_25class",
            }

        return self._recognizer.predict_landmarks(
            landmarks_arr,
            confidence_threshold=confidence_threshold,
            min_frames=min_frames,
        )


# Global singleton instance
_instance: Optional[TwentyFiveClassBiGRUService] = None


def get_twenty_five_class_bigru_service() -> TwentyFiveClassBiGRUService:
    global _instance
    if _instance is None:
        _instance = TwentyFiveClassBiGRUService()
    return _instance
