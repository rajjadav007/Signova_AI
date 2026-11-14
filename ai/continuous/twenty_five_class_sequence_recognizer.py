"""
Signova AI - 25-Class ISL Sentence Sequence Recognizer.
Velocity-Dominant Kinematic DTW Engine (Upgraded from BiGRU).

Accepts:
- An ordered list of RGB/BGR image frames (e.g. from webcam or video file)
- OR pre-extracted (T, 75, 4) landmark array
- OR an existing preprocessed feature sequence

Runs:
1. MediaPipe landmark extraction (Pose 33, LH 21, RH 21) with forward-fill if raw frames provided
2. 678-dimensional temporal feature extraction (225 spatial, 225 velocity, 225 acceleration, 3 masks)
3. 48-dimensional group-partitioned PCA projection (spatial 24, velocity 16, accel 8) with 2x temporal stride
4. Pairwise Velocity-Dominant DTW matching against the 126 canonical training exemplars
5. 1-NN / 5-NN class distance aggregation and temperature-scaled posterior probabilities
6. Calibrated uncertainty rejection (distance <= 0.40, margin >= 0.015)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.dtw_knn_classifier import DTWKNNClassifier, compute_dtw_distance
from ai.continuous.temporal_features import extract_temporal_features
from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor, apply_hand_forward_fill
from scripts.recognize_sentence_dtw import preprocess_sequence

logger = logging.getLogger("twenty_five_class_recognizer")


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', p.name)]


class TwentyFiveClassSequenceRecognizer:
    """
    Inference engine for 25-class ISL sentence sequence recognition using
    Velocity-Dominant Kinematic DTW with calibrated rejection.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if model_dir is None:
            self.model_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "twenty_five_class_sequence"
        else:
            self.model_dir = Path(model_dir)

        # Fallback to root models dir if needed
        if not (self.model_dir / "dtw_reference_dataset.pkl").exists():
            root_model_dir = Path(r"D:\Projects\models\isl_csltr_continuous\twenty_five_class_sequence")
            if (root_model_dir / "dtw_reference_dataset.pkl").exists():
                self.model_dir = root_model_dir

        # 1. Load class mapping
        class_mapping_path = self.model_dir / "class_mapping.json"
        if not class_mapping_path.exists():
            root_mapping = Path(r"D:\Projects\models\isl_csltr_continuous\twenty_five_class_sequence\class_mapping.json")
            if root_mapping.exists():
                class_mapping_path = root_mapping
            else:
                raise FileNotFoundError(f"class_mapping.json not found at {class_mapping_path}")

        with open(class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            self.classes = mapping_data["classes"]
            self.class_names = [c["sentence"] for c in self.classes]
            self.num_classes = len(self.class_names)

        # 2. Load preprocessors (improved group-partitioned PCA)
        preprocessors_path = (
            PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "ml_cache" / "preprocessors_improved.pkl"
        )
        if not preprocessors_path.exists():
            preprocessors_path = self.model_dir / "preprocessors_improved.pkl"

        if preprocessors_path.exists():
            with open(preprocessors_path, "rb") as f:
                self.preprocessors = pickle.load(f)
        else:
            logger.warning("preprocessors_improved.pkl not found at %s", preprocessors_path)
            self.preprocessors = None

        # 3. Load DTW reference dataset & fitted classifier
        dataset_path = self.model_dir / "dtw_reference_dataset.pkl"
        model_path = self.model_dir / "best_dtw_knn_model.pkl"

        if not dataset_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"DTW artifacts missing in {self.model_dir}. Please ensure dtw_reference_dataset.pkl and best_dtw_knn_model.pkl exist."
            )

        with open(dataset_path, "rb") as f:
            self.dataset_pkg = pickle.load(f)
            self.train_seqs = self.dataset_pkg["train"]["sequences"]
            self.train_labels = np.asarray(self.dataset_pkg["train"]["labels"], dtype=np.int64)

        with open(model_path, "rb") as f:
            self.clf: DTWKNNClassifier = pickle.load(f)

        self._extractor: Optional[ModernMediaPipeExtractor] = None
        self.distance_threshold = 0.40
        self.min_margin = 0.015

        logger.info(
            "TwentyFiveClassSequenceRecognizer initialized with %d classes and %d reference exemplars.",
            self.num_classes,
            len(self.train_seqs),
        )

    @property
    def extractor(self) -> ModernMediaPipeExtractor:
        if self._extractor is None:
            self._extractor = ModernMediaPipeExtractor()
        return self._extractor

    @property
    def model(self) -> Any:
        """Alias for compatibility with services checking for model attribute."""
        return self.clf

    def predict_features(
        self,
        features_48: np.ndarray,
        confidence_threshold: float = 0.25,
        min_frames: int = 5,
        fps: float = 25.0
    ) -> Dict[str, Any]:
        """
        Run inference on preprocessed (T, 48) feature array.
        """
        t0 = time.perf_counter()
        t, d = features_48.shape
        if d != 48:
            raise ValueError(f"Expected 48-dimensional features, got {d}")

        if t < min_frames:
            return {
                "success": False,
                "error": f"Insufficient frames ({t} < {min_frames}) for sentence recognition.",
                "frame_count": t,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "dtw_knn_25class",
                "model_used": "VelocityDominantDTWKNN"
            }

        # Pairwise DTW to all 126 train exemplars
        dists = [
            compute_dtw_distance(
                features_48, ref,
                metric="cosine",
                window=10,
                group_weights=(0.05, 1.0, 0.05),
                group_slices=(slice(0, 24), slice(24, 40), slice(40, 48))
            )
            for ref in self.train_seqs
        ]

        # Aggregate minimum distance per class
        class_min_dists = {}
        for c_id in range(self.num_classes):
            c_dists = [dists[i] for i in range(len(dists)) if self.train_labels[i] == c_id]
            class_min_dists[c_id] = float(min(c_dists)) if c_dists else 1.0

        sorted_classes = sorted(class_min_dists.items(), key=lambda x: x[1])
        best_cid, best_dist = sorted_classes[0]
        second_cid, second_dist = sorted_classes[1]
        margin = second_dist - best_dist

        # Temperature-scaled softmax posterior probabilities over negative distances
        tau = 0.05
        raw_logits = np.array([-class_min_dists[c] / tau for c in range(self.num_classes)], dtype=np.float32)
        raw_logits -= np.max(raw_logits)
        exp_logits = np.exp(raw_logits)
        probs = exp_logits / np.sum(exp_logits)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_sentence = self.class_names[best_cid]
        best_prob = float(probs[best_cid])

        # Top 5 candidates
        top5_candidates = [
            {
                "class_id": int(cid),
                "sentence": self.class_names[cid],
                "probability": round(float(probs[cid]), 4),
                "dtw_distance": round(float(dist), 4)
            }
            for cid, dist in sorted_classes[:5]
        ]

        class_probabilities = {
            self.class_names[i]: round(float(probs[i]), 4)
            for i in range(self.num_classes)
        }

        # Rejection & uncertainty thresholding
        is_confident = (best_dist <= self.distance_threshold) and (margin >= self.min_margin)
        uncertainty_warning = None
        if not is_confident:
            reasons = []
            if best_dist > self.distance_threshold:
                reasons.append(f"DTW distance {best_dist:.3f} > {self.distance_threshold}")
            if margin < self.min_margin:
                reasons.append(f"margin {margin:.3f} < {self.min_margin}")
            uncertainty_warning = (
                f"Low confidence prediction ({best_prob*100:.1f}%, {', '.join(reasons)}). Prediction may be uncertain."
            )

        return {
            "success": True,
            "predicted_class_id": best_cid,
            "predicted_sentence": pred_sentence,
            "confidence": best_prob,
            "dtw_distance": round(best_dist, 4),
            "margin": round(margin, 4),
            "class_probabilities": class_probabilities,
            "top5_candidates": top5_candidates,
            "frame_count": t,
            "is_confident": is_confident,
            "uncertainty_warning": uncertainty_warning,
            "latency_ms": round(latency_ms, 2),
            "model_identifier": "dtw_knn_25class",
            "model_type": "VelocityDominantDTWKNN",
            "model_used": "VelocityDominantDTWKNN"
        }

    def predict_landmarks(
        self,
        landmarks_seq: np.ndarray,
        confidence_threshold: float = 0.25,
        min_frames: int = 5,
        fps: float = 25.0
    ) -> Dict[str, Any]:
        """
        Inference on raw (T, 75, 4) landmark array.
        """
        arr = np.asarray(landmarks_seq, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 300:
            arr = arr.reshape(-1, 75, 4)

        if arr.ndim != 3 or arr.shape[1] != 75 or arr.shape[2] != 4:
            raise ValueError(f"Expected (T, 75, 4) landmarks, got {arr.shape}")

        t = arr.shape[0]
        if t < min_frames:
            return {
                "success": False,
                "error": f"Insufficient frames ({t} < {min_frames}) for sentence recognition.",
                "frame_count": t,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "dtw_knn_25class",
                "model_used": "VelocityDominantDTWKNN"
            }

        # 1. Forward-fill missing hands
        coords_flat = arr.reshape(t, 300)
        filled_flat = apply_hand_forward_fill(coords_flat, max_consecutive_frames=15)
        filled_landmarks = filled_flat.reshape(t, 75, 4)

        # 2. Extract 678-dim temporal features (spatial + velocity + acceleration + mask)
        raw_678 = extract_temporal_features(filled_landmarks, fps=fps)

        # 3. Project to 48-dim PCA with 2x temporal stride
        proc_48 = preprocess_sequence(raw_678, preprocessors=self.preprocessors, stride=2)

        return self.predict_features(
            proc_48,
            confidence_threshold=confidence_threshold,
            min_frames=min_frames,
            fps=fps
        )

    def predict_frames(
        self,
        frames_bgr: List[np.ndarray],
        confidence_threshold: float = 0.25,
        min_frames: int = 5,
        fps: float = 25.0
    ) -> Dict[str, Any]:
        """
        Extract MediaPipe landmarks from RGB/BGR frames and run 25-class Velocity DTW inference.
        """
        if not frames_bgr:
            return {
                "success": False,
                "error": "No frames provided.",
                "frame_count": 0,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "dtw_knn_25class",
                "model_used": "VelocityDominantDTWKNN"
            }

        landmarks_list = []
        for frame in frames_bgr:
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_75x4, _ = self.extractor.extract_isign_frame(rgb)
            landmarks_list.append(raw_75x4)

        if not landmarks_list or len(landmarks_list) < min_frames:
            return {
                "success": False,
                "error": f"Failed to extract sufficient frames ({len(landmarks_list)} < {min_frames}).",
                "frame_count": len(landmarks_list),
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "dtw_knn_25class",
                "model_used": "VelocityDominantDTWKNN"
            }

        landmarks_arr = np.array(landmarks_list, dtype=np.float32)
        return self.predict_landmarks(
            landmarks_arr,
            confidence_threshold=confidence_threshold,
            min_frames=min_frames,
            fps=fps
        )
