"""
Signova AI - Five-Class BiGRU Attention Sentence Recognition Service.
Backend service wrapping FiveClassSequenceRecognizer for live webcam & video inference.
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

logger = logging.getLogger("signova.five_class_bigru_service")


ORIGINAL_SENTENCE_IDS = {
    0: 5,   # comb your hair
    1: 23,  # help me
    2: 35,  # i am fine. thank you sir.
    3: 53,  # i need water
    4: 91,  # who are you
}


class FiveClassBiGRUService:
    """
    Singleton service managing 5-Class ISL BiGRU Attention Sentence Recognition.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.model_dir = (
            self.project_root
            / "models"
            / "isl_csltr_continuous"
            / "five_class_sequence"
        )
        if not (self.model_dir / "best_model.pt").exists():
            root_model_dir = self.project_root.parent / "models" / "isl_csltr_continuous" / "five_class_sequence"
            if (root_model_dir / "best_model.pt").exists():
                self.model_dir = root_model_dir

        self.checkpoint_path = self.model_dir / "best_model.pt"
        self.is_loaded: bool = False
        self._load_lock = threading.Lock()
        self._recognizer: Optional[Any] = None

        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load 5-class BiGRU model and configuration into memory."""
        with self._load_lock:
            checkpoint_path = self.checkpoint_path
            class_mapping_path = self.model_dir / "class_mapping.json"

            if not checkpoint_path.exists() or not class_mapping_path.exists():
                logger.warning(
                    "BiGRU artifacts not found in %s. BiGRU inference will not be available.",
                    self.model_dir,
                )
                return

            try:
                from ai.continuous.five_class_sequence_recognizer import FiveClassSequenceRecognizer
                self._recognizer = FiveClassSequenceRecognizer(model_dir=self.model_dir)
                self.is_loaded = True
                logger.info(
                    "FiveClassBiGRUService successfully initialized with %d classes.",
                    self._recognizer.num_classes,
                )
            except Exception as e:
                logger.error("Failed to load FiveClassSequenceRecognizer: %s", e, exc_info=True)
                self.is_loaded = False

    def get_classes(self) -> List[Dict[str, Any]]:
        """Return the list of 5 supported sentence classes."""
        if not self.is_loaded:
            self.load_artifacts()
        if self._recognizer is not None:
            return self._recognizer.classes
        return []

    def predict_video(
        self,
        video_bytes: bytes,
        filename: str = "webcam.webm",
        confidence_threshold: float = 0.35,
        min_frames: int = 8,
    ) -> Dict[str, Any]:
        """
        Process recorded live webcam video or uploaded video through MediaPipe landmark extraction
        and FiveClassBiGRUAttentionModel sequence inference.
        """
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded or self._recognizer is None:
                raise RuntimeError(
                    "FiveClassBiGRU model artifacts not available at: " + str(self.model_dir)
                )

        t_start = time.perf_counter()
        suffix = Path(filename).suffix.lower() or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            # 1. Decode video frames chronologically via OpenCV
            cap = cv2.VideoCapture(str(tmp_path))
            rgb_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV uses BGR; convert to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb)
            cap.release()

            frame_count = len(rgb_frames)

            # 2. Quality gate: Insufficient frames
            if frame_count < min_frames:
                total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return {
                    "success": True,
                    "is_recognized": False,
                    "status": "INSUFFICIENT_FRAMES",
                    "predicted_sentence": None,
                    "predicted_class_id": None,
                    "original_sentence_id": None,
                    "confidence": 0.0,
                    "class_probabilities": {},
                    "frame_count": frame_count,
                    "processing_time_ms": total_ms,
                    "model_used": "FiveClassBiGRUAttentionModel",
                    "reason": f"Only {frame_count} frames captured. Minimum {min_frames} frames required for a sentence.",
                }

            # 3. Model Inference on chronological RGB frame sequence
            rec_result = self._recognizer.predict_frames(rgb_frames)
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            if not rec_result.get("valid_landmarks", True):
                return {
                    "success": True,
                    "is_recognized": False,
                    "status": "NO_VALID_LANDMARKS",
                    "predicted_sentence": None,
                    "predicted_class_id": None,
                    "original_sentence_id": None,
                    "confidence": 0.0,
                    "class_probabilities": rec_result.get("class_probabilities", {}),
                    "frame_count": frame_count,
                    "processing_time_ms": total_ms,
                    "model_inference_latency_ms": rec_result.get("latency_ms", 0.0),
                    "model_used": "FiveClassBiGRUAttentionModel",
                    "reason": "No valid pose or hand landmarks detected in video.",
                    "margin_to_second": 0.0,
                }

            conf = rec_result["confidence"]
            pred_id = rec_result["predicted_class_id"]
            pred_sentence = rec_result["predicted_sentence"]

            sorted_probs = sorted(rec_result["class_probabilities"].values(), reverse=True)
            top_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
            second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            margin = round(top_prob - second_prob, 4)

            # 4. Quality gate: Softmax Confidence Threshold
            is_recognized = conf >= confidence_threshold
            status_str = "SUCCESS" if is_recognized else "LOW_CONFIDENCE"
            reason_str = None if is_recognized else f"Highest confidence ({conf*100:.1f}%) is below acceptance threshold ({confidence_threshold*100:.1f}%)."

            logger.info(
                "[DIAGNOSTIC - BiGRU Service] Request Completed:\n"
                "  Requested Model: bigru\n"
                "  Actual Loaded Model: FiveClassBiGRUAttentionModel\n"
                "  Model Checkpoint Path: %s\n"
                "  Extracted Frames: %d\n"
                "  Predicted Class ID: %s\n"
                "  Predicted Sentence: %s\n"
                "  All Five Class Probabilities: %s",
                str(self.checkpoint_path),
                frame_count,
                str(pred_id),
                str(pred_sentence),
                json.dumps(rec_result["class_probabilities"]),
            )

            return {
                "success": True,
                "is_recognized": is_recognized,
                "status": status_str,
                "predicted_sentence": pred_sentence,
                "predicted_class_id": pred_id,
                "original_sentence_id": ORIGINAL_SENTENCE_IDS.get(pred_id),
                "confidence": conf,
                "class_probabilities": rec_result["class_probabilities"],
                "frame_count": frame_count,
                "processing_time_ms": total_ms,
                "model_inference_latency_ms": rec_result["latency_ms"],
                "model_used": "FiveClassBiGRUAttentionModel",
                "reason": reason_str,
                "margin_to_second": margin,
            }

        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def predict_landmarks(
        self,
        landmarks: Union[List[Any], np.ndarray],
        confidence_threshold: float = 0.35,
        min_frames: int = 8,
    ) -> Dict[str, Any]:
        """
        Process pre-extracted landmark sequence through FiveClassBiGRUAttentionModel.
        """
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded or self._recognizer is None:
                raise RuntimeError(
                    "FiveClassBiGRU model artifacts not available at: " + str(self.model_dir)
                )

        t_start = time.perf_counter()
        arr = np.asarray(landmarks, dtype=np.float32)
        frame_count = len(arr)

        if frame_count < min_frames:
            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            return {
                "success": True,
                "is_recognized": False,
                "status": "INSUFFICIENT_FRAMES",
                "predicted_sentence": None,
                "predicted_class_id": None,
                "original_sentence_id": None,
                "confidence": 0.0,
                "class_probabilities": {},
                "frame_count": frame_count,
                "processing_time_ms": total_ms,
                "model_used": "FiveClassBiGRUAttentionModel",
                "reason": f"Only {frame_count} frames captured. Minimum {min_frames} frames required for a sentence.",
                "margin_to_second": 0.0,
            }

        rec_result = self._recognizer.predict_landmarks(arr)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        if not rec_result.get("valid_landmarks", True):
            return {
                "success": True,
                "is_recognized": False,
                "status": "NO_VALID_LANDMARKS",
                "predicted_sentence": None,
                "predicted_class_id": None,
                "original_sentence_id": None,
                "confidence": 0.0,
                "class_probabilities": rec_result.get("class_probabilities", {}),
                "frame_count": frame_count,
                "processing_time_ms": total_ms,
                "model_inference_latency_ms": rec_result.get("latency_ms", 0.0),
                "model_used": "FiveClassBiGRUAttentionModel",
                "reason": "No valid landmarks detected in input sequence.",
                "margin_to_second": 0.0,
            }

        conf = rec_result["confidence"]
        pred_id = rec_result["predicted_class_id"]
        pred_sentence = rec_result["predicted_sentence"]

        sorted_probs = sorted(rec_result["class_probabilities"].values(), reverse=True)
        top_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
        second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        margin = round(top_prob - second_prob, 4)

        is_recognized = conf >= confidence_threshold
        status_str = "SUCCESS" if is_recognized else "LOW_CONFIDENCE"
        reason_str = None if is_recognized else f"Highest confidence ({conf*100:.1f}%) is below acceptance threshold ({confidence_threshold*100:.1f}%)."

        return {
            "success": True,
            "is_recognized": is_recognized,
            "status": status_str,
            "predicted_sentence": pred_sentence,
            "predicted_class_id": pred_id,
            "original_sentence_id": ORIGINAL_SENTENCE_IDS.get(pred_id),
            "confidence": conf,
            "class_probabilities": rec_result["class_probabilities"],
            "frame_count": frame_count,
            "processing_time_ms": total_ms,
            "model_inference_latency_ms": rec_result["latency_ms"],
            "model_used": "FiveClassBiGRUAttentionModel",
            "reason": reason_str,
            "margin_to_second": margin,
        }


five_class_bigru_service = FiveClassBiGRUService()
