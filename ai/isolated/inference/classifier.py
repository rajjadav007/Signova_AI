"""
Inference engine for Individual ISL Sign Recognition.
Loads trained classifier checkpoint and executes prediction on landmark sequences or raw videos.
"""

import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ai.isolated.config.isolated_settings import (
    DEFAULT_SEQUENCE_LENGTH,
    IDX_TO_CLASS,
    ISOLATED_20CLASS_CHECKPOINT_PATH,
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    ISOLATED_IMPROVED_CHECKPOINT_PATH,
    PROJECT_ROOT,
)
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.preprocessing.normalizer import (
    compute_hand_motion_energy,
    detect_active_motion_window,
    spatial_normalize,
    temporal_resample,
)

logger = logging.getLogger("signova.isolated_inference")


class IsolatedSignRecognizer:
    """
    Isolated Sign Language Recognition inference wrapper.
    Accepts raw video bytes or (T, 75, 4) landmark sequences and returns top-k predictions.
    Supports dynamic class vocabularies (10-class or 20-class) and zero-downtime rollback.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path).resolve()
        elif os.environ.get("ISOLATED_MODEL_CHECKPOINT"):
            self.checkpoint_path = Path(os.environ["ISOLATED_MODEL_CHECKPOINT"]).resolve()
        elif os.environ.get("ISOLATED_MODEL_VERSION", "").lower() in ("20", "20class", "20_class") and ISOLATED_20CLASS_CHECKPOINT_PATH.exists():
            self.checkpoint_path = ISOLATED_20CLASS_CHECKPOINT_PATH.resolve()
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
        self.food_water_ambiguity_margin: float = 0.20
        self.enable_motion_window: bool = False
        self.classes: List[str] = list(ISOLATED_CLASSES)
        self.idx_to_class: Dict[int, str] = dict(IDX_TO_CLASS)

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

            # Check if model object has class_names
            if hasattr(self.model, "class_names") and self.model.class_names:
                self.classes = list(self.model.class_names)
                self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

            # Load threshold and classes if metadata file exists
            meta_candidates = [
                self.checkpoint_path.parent / "model_metadata_20class.json" if "20class" in self.checkpoint_path.name else None,
                self.checkpoint_path.parent / "model_metadata.json",
            ]
            for meta_path in meta_candidates:
                if meta_path is not None and meta_path.exists():
                    try:
                        with open(meta_path, mode="r", encoding="utf-8") as f:
                            meta = json.load(f)
                        self.rejection_threshold = float(meta.get("optimal_rejection_threshold", self.rejection_threshold))
                        if "food_water_ambiguity_margin" in meta:
                            self.food_water_ambiguity_margin = float(meta["food_water_ambiguity_margin"])
                        if "classes" in meta and meta["classes"]:
                            self.classes = list(meta["classes"])
                            self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
                        break
                    except Exception:
                        pass

            # Check 20-class labels file if classes not loaded yet
            if len(self.classes) <= 10 and "20class" in self.checkpoint_path.name:
                labels_20 = PROJECT_ROOT / "models" / "labels" / "isolated_20class_labels.json"
                if labels_20.exists():
                    try:
                        with open(labels_20, mode="r", encoding="utf-8") as f:
                            self.classes = json.load(f)
                            self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
                    except Exception:
                        pass

            self.is_loaded = True
            logger.info("Loaded isolated classifier: %s from %s (%d classes, threshold=%.2f)", self.model_type, self.checkpoint_path, len(self.classes), self.rejection_threshold)
        except Exception as exc:
            logger.error("Failed to load isolated model from %s: %s", self.checkpoint_path, exc)
            self.is_loaded = False
            raise

    def predict_landmarks(
        self,
        raw_keypoints: Union[np.ndarray, List[Any]],
        category: Optional[str] = None,
        top_k: int = 3,
        enable_motion_window: Optional[bool] = None,
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

        # Preprocessing: Spatial normalization + active motion window detection + uniform temporal resampling
        normed = spatial_normalize(kps)
        use_motion_window = self.enable_motion_window if enable_motion_window is None else enable_motion_window
        if use_motion_window:
            start_idx, end_idx = detect_active_motion_window(normed)
            active_normed = normed[start_idx:end_idx]
        else:
            active_normed = normed

        resampled = temporal_resample(active_normed, target_length=self.sequence_length)
        features = extract_isolated_features(resampled)  # (sequence_length, 600)

        # Inference
        is_food_water_ambiguity = False
        fw_margin = 0.0

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
                    "label": self.idx_to_class.get(int(idx), "unknown"),
                    "sign": self.idx_to_class.get(int(idx), "unknown"),
                    "display_name": self.idx_to_class.get(int(idx), "unknown").capitalize(),
                    "confidence": round(float(probs[idx]), 4),
                }
                for idx in top_indices
            ]
            predicted_sign = top_predictions[0]["sign"]
            confidence = top_predictions[0]["confidence"]

            # Food/Water ambiguity evaluation
            all_sorted_indices = np.argsort(probs)[::-1]
            top1_sign = self.idx_to_class.get(int(all_sorted_indices[0]), "unknown")
            top2_sign = self.idx_to_class.get(int(all_sorted_indices[1]), "unknown") if len(all_sorted_indices) > 1 else None
            top1_prob = float(probs[all_sorted_indices[0]])
            top2_prob = float(probs[all_sorted_indices[1]]) if len(all_sorted_indices) > 1 else 0.0

            if {top1_sign, top2_sign} == {"food", "water"}:
                fw_margin = top1_prob - top2_prob
                if fw_margin < self.food_water_ambiguity_margin:
                    is_food_water_ambiguity = True
        else:
            # Default baseline stub when model not yet trained
            top_predictions = [
                {"label": self.classes[0], "sign": self.classes[0], "display_name": self.classes[0].capitalize(), "confidence": 0.50},
                {"label": self.classes[1] if len(self.classes) > 1 else "unknown", "sign": self.classes[1] if len(self.classes) > 1 else "unknown", "display_name": (self.classes[1] if len(self.classes) > 1 else "unknown").capitalize(), "confidence": 0.30},
                {"label": self.classes[2] if len(self.classes) > 2 else "unknown", "sign": self.classes[2] if len(self.classes) > 2 else "unknown", "display_name": (self.classes[2] if len(self.classes) > 2 else "unknown").capitalize(), "confidence": 0.20},
            ]
            predicted_sign = top_predictions[0]["sign"]
            confidence = top_predictions[0]["confidence"]

        proc_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        is_recognized = (confidence >= self.rejection_threshold) and not is_food_water_ambiguity
        cat_str = category if isinstance(category, str) else "general"

        top_3 = top_predictions[:3]
        confirmed_label = predicted_sign if is_recognized else "uncertain"

        if is_recognized:
            reason = None
            display_name = predicted_sign.capitalize()
        elif is_food_water_ambiguity:
            reason = (
                f"Ambiguous between Food and Water: probability difference "
                f"({fw_margin:.2f}) below configured margin ({self.food_water_ambiguity_margin:.2f})"
            )
            display_name = f"Uncertain ({predicted_sign.capitalize()})"
        else:
            reason = f"Confidence {confidence:.2f} below rejection threshold {self.rejection_threshold:.2f}"
            display_name = f"Uncertain ({predicted_sign.capitalize()})"

        return {
            "success": True,
            "predicted_label": confirmed_label,
            "predicted_sign": predicted_sign,
            "display_name": display_name,
            "category": cat_str,
            "confidence": confidence,
            "status": "recognized" if is_recognized else "uncertain",
            "is_valid": is_recognized,
            "processing_time_ms": proc_time_ms,
            "model_used": self.model_type,
            "rejection_threshold": self.rejection_threshold,
            "top_3_predictions": top_3,
            "top_predictions": top_predictions,
            "reason": reason,
        }

    def predict_video(
        self,
        video_bytes: bytes,
        filename: str,
        category: Optional[str] = None,
        top_k: int = 3,
        enable_motion_window: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract MediaPipe landmarks from video bytes and run isolated sign recognition.
        Enables active-motion window detection by default to trim webcam resting frames.
        """
        from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            if not hasattr(self, "_extractor") or self._extractor is None:
                from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor
                self._extractor = ModernMediaPipeExtractor()
            raw_spatial, meta = self._extractor.extract_isign_video(tmp_path)
            landmarks_75x4 = raw_spatial.reshape(-1, 75, 4)
            return self.predict_landmarks(
                landmarks_75x4,
                category=category,
                top_k=top_k,
                enable_motion_window=enable_motion_window,
            )
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
