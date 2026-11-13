"""
Inference engine for Individual ISL Sign Recognition.
Loads trained classifier checkpoint and executes prediction on landmark sequences or raw videos.
"""

import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ai.isolated.config.isolated_settings import (
    DEFAULT_SEQUENCE_LENGTH,
    IDX_TO_CLASS,
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    ISOLATED_IMPROVED_CHECKPOINT_PATH,
)
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample

logger = logging.getLogger("signova.isolated_inference")


class IsolatedSignRecognizer:
    """
    Isolated Sign Language Recognition inference wrapper.
    Accepts raw video bytes or (T, 75, 4) landmark sequences and returns top-k predictions.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path).resolve()
        elif ISOLATED_IMPROVED_CHECKPOINT_PATH.exists():
            self.checkpoint_path = ISOLATED_IMPROVED_CHECKPOINT_PATH
        else:
            self.checkpoint_path = ISOLATED_CHECKPOINT_PATH.resolve()

        self.sequence_length = sequence_length
        self.model: Optional[Any] = None
        self.is_loaded: bool = False
        self.device: str = "cpu"
        self.model_type: str = "Uninitialized"
        self.rejection_threshold: float = 0.50

    def load_model(self) -> None:
        """Loads classifier checkpoint if available."""
        if not self.checkpoint_path.exists():
            logger.warning(
                "Isolated classifier checkpoint not found at: %s. Using heuristic fallback.",
                self.checkpoint_path,
            )
            self.model = None
            self.model_type = "HeuristicBaseline"
            self.is_loaded = True
            return

        try:
            import json
            import joblib
            self.model = joblib.load(self.checkpoint_path)
            self.model_type = type(self.model).__name__

            # Load threshold if metadata file exists
            meta_path = self.checkpoint_path.parent / "model_metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, mode="r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.rejection_threshold = float(meta.get("optimal_rejection_threshold", 0.50))
                except Exception:
                    self.rejection_threshold = 0.50

            self.is_loaded = True
            logger.info("Loaded isolated classifier: %s from %s (threshold=%.2f)", self.model_type, self.checkpoint_path, self.rejection_threshold)
        except Exception as exc:
            logger.error("Failed to load isolated model from %s: %s", self.checkpoint_path, exc)
            self.is_loaded = False
            raise

    def predict_landmarks(
        self,
        raw_keypoints: Union[np.ndarray, List[Any]],
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Predict isolated ISL sign from landmark sequence.
        Input: (T, 75, 4) or (T, 300) array.
        """
        t0 = time.perf_counter()
        kps = np.asarray(raw_keypoints, dtype=np.float32)

        # Handle empty landmark input
        if kps.size == 0 or kps.shape[0] == 0:
            proc_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            safe_top3 = [{"label": "unknown", "sign": "unknown", "display_name": "Unknown", "confidence": 0.0}]
            return {
                "success": False,
                "predicted_label": "uncertain",
                "predicted_sign": "unknown",
                "display_name": "Uncertain",
                "category": category or "general",
                "confidence": 0.0,
                "status": "uncertain",
                "is_valid": False,
                "processing_time_ms": proc_time_ms,
                "model_used": self.model_type,
                "rejection_threshold": self.rejection_threshold,
                "top_3_predictions": safe_top3,
                "top_predictions": safe_top3,
                "reason": "Empty landmark sequence received",
            }

        # Handle invalid dimensions
        if kps.ndim not in (2, 3):
            proc_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            safe_top3 = [{"label": "unknown", "sign": "unknown", "display_name": "Unknown", "confidence": 0.0}]
            return {
                "success": False,
                "predicted_label": "uncertain",
                "predicted_sign": "unknown",
                "display_name": "Uncertain",
                "category": category or "general",
                "confidence": 0.0,
                "status": "uncertain",
                "is_valid": False,
                "processing_time_ms": proc_time_ms,
                "model_used": self.model_type,
                "rejection_threshold": self.rejection_threshold,
                "top_3_predictions": safe_top3,
                "top_predictions": safe_top3,
                "reason": f"Invalid landmark dimensions: shape {kps.shape}",
            }

        if kps.ndim == 2 and kps.shape[1] == 300:
            kps = kps.reshape(-1, 75, 4)
        elif kps.ndim == 2 and kps.shape[1] != 75:
            # Handle variable channel inputs
            T = kps.shape[0]
            padded = np.zeros((T, 75, 4), dtype=np.float32)
            n_vals = min(kps.shape[1], 300)
            padded.reshape(T, -1)[:, :n_vals] = kps[:, :n_vals]
            kps = padded

        # Guard against completely unobserved/zero landmarks
        if np.all(kps[..., :3] == 0.0):
            proc_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            safe_top3 = [{"label": "unknown", "sign": "unknown", "display_name": "Unknown", "confidence": 0.0}]
            return {
                "success": False,
                "predicted_label": "uncertain",
                "predicted_sign": "unknown",
                "display_name": "Uncertain",
                "category": category or "general",
                "confidence": 0.0,
                "status": "uncertain",
                "is_valid": False,
                "processing_time_ms": proc_time_ms,
                "model_used": self.model_type,
                "rejection_threshold": self.rejection_threshold,
                "top_3_predictions": safe_top3,
                "top_predictions": safe_top3,
                "reason": "No valid landmarks detected in input sequence",
            }

        # Preprocessing: Spatial normalization + uniform temporal resampling
        normed = spatial_normalize(kps)
        resampled = temporal_resample(normed, target_length=self.sequence_length)
        features = extract_isolated_features(resampled)  # (sequence_length, 600)

        # Inference
        if self.model is not None and hasattr(self.model, "predict_proba"):
            seq_input = features[np.newaxis, ...]
            try:
                probs = self.model.predict_proba(seq_input)[0]
            except Exception:
                probs = self.model.predict_proba(features.reshape(1, -1))[0]

            k_val = int(getattr(top_k, "default", top_k)) if hasattr(top_k, "default") or isinstance(top_k, (int, float, str)) else 3
            k_val = max(1, min(int(k_val), len(probs)))
            top_indices = np.argsort(probs)[::-1][:k_val]
            top_predictions = [
                {
                    "label": IDX_TO_CLASS.get(int(idx), "unknown"),
                    "sign": IDX_TO_CLASS.get(int(idx), "unknown"),
                    "display_name": IDX_TO_CLASS.get(int(idx), "unknown").capitalize(),
                    "confidence": round(float(probs[idx]), 4),
                }
                for idx in top_indices
            ]
            predicted_sign = top_predictions[0]["sign"]
            confidence = top_predictions[0]["confidence"]
        else:
            # Default baseline stub when model not yet trained
            top_predictions = [
                {"label": ISOLATED_CLASSES[0], "sign": ISOLATED_CLASSES[0], "display_name": ISOLATED_CLASSES[0].capitalize(), "confidence": 0.50},
                {"label": ISOLATED_CLASSES[1], "sign": ISOLATED_CLASSES[1], "display_name": ISOLATED_CLASSES[1].capitalize(), "confidence": 0.30},
                {"label": ISOLATED_CLASSES[2], "sign": ISOLATED_CLASSES[2], "display_name": ISOLATED_CLASSES[2].capitalize(), "confidence": 0.20},
            ]
            predicted_sign = top_predictions[0]["sign"]
            confidence = top_predictions[0]["confidence"]

        proc_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        is_recognized = confidence >= self.rejection_threshold
        cat_str = category if isinstance(category, str) else "general"

        top_3 = top_predictions[:3]
        confirmed_label = predicted_sign if is_recognized else "uncertain"

        return {
            "success": True,
            "predicted_label": confirmed_label,
            "predicted_sign": predicted_sign,
            "display_name": predicted_sign.capitalize() if is_recognized else f"Uncertain ({predicted_sign.capitalize()})",
            "category": cat_str,
            "confidence": confidence,
            "status": "recognized" if is_recognized else "uncertain",
            "is_valid": is_recognized,
            "processing_time_ms": proc_time_ms,
            "model_used": self.model_type,
            "rejection_threshold": self.rejection_threshold,
            "top_3_predictions": top_3,
            "top_predictions": top_predictions,
            "reason": None if is_recognized else f"Confidence {confidence:.2f} below rejection threshold {self.rejection_threshold:.2f}",
        }

    def predict_video(
        self,
        video_bytes: bytes,
        filename: str,
        category: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Extract MediaPipe landmarks from video bytes and run isolated sign recognition.
        """
        from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            extractor = ModernMediaPipeExtractor()
            raw_spatial, meta = extractor.extract_isign_video(tmp_path)
            landmarks_75x4 = raw_spatial.reshape(-1, 75, 4)
            return self.predict_landmarks(landmarks_75x4, category=category, top_k=top_k)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
