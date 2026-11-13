"""
Signova AI - 25-Class ISL Sentence Sequence Recognizer.
Phase 9 Inference Module:
Accepts:
- An ordered list of RGB/BGR image frames (e.g. from webcam or video frames folder)
- OR pre-extracted (T, 75, 4) landmark array
- OR an existing preprocessed (T, 300) feature sequence

Runs:
1. MediaPipe landmark extraction (Pose 33, LH 21, RH 21) if raw frames provided
2. Torso-centered coordinate normalization
3. Variable-length 25-Class BiGRU Attention Model forward pass
4. Softmax probability distribution and prediction
5. Always returns top prediction with clear uncertainty warning if confidence < 0.25
"""

import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.twenty_five_class_model import TwentyFiveClassBiGRUAttentionModel
from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

logger = logging.getLogger("twenty_five_class_recognizer")


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', p.name)]


class TwentyFiveClassSequenceRecognizer:
    """
    Inference engine for 25-class ISL sentence sequence recognition.
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
        if not (self.model_dir / "best_model.pt").exists():
            root_model_dir = Path(r"D:\Projects\models\isl_csltr_continuous\twenty_five_class_sequence")
            if (root_model_dir / "best_model.pt").exists():
                self.model_dir = root_model_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load configurations & class mapping
        class_mapping_path = self.model_dir / "class_mapping.json"
        if not class_mapping_path.exists():
            raise FileNotFoundError(f"class_mapping.json not found at {class_mapping_path}")

        with open(class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            self.classes = mapping_data["classes"]
            self.class_names = [c["sentence"] for c in self.classes]
            self.num_classes = len(self.class_names)

        model_config_path = self.model_dir / "model_config.json"
        if model_config_path.exists():
            with open(model_config_path, "r", encoding="utf-8") as f:
                self.model_config = json.load(f)
        else:
            self.model_config = {
                "input_dim": 300,
                "proj_dim": 128,
                "hidden_dim": 64,
                "num_layers": 1,
                "num_classes": self.num_classes,
                "dropout": 0.3
            }

        prep_config_path = self.model_dir / "preprocessing_config.json"
        if prep_config_path.exists():
            with open(prep_config_path, "r", encoding="utf-8") as f:
                self.prep_config = json.load(f)
        else:
            self.prep_config = {}

        # 2. Load Model
        checkpoint_path = self.model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"best_model.pt not found at {checkpoint_path}")

        self.model = TwentyFiveClassBiGRUAttentionModel(
            input_dim=self.model_config.get("input_dim", 300),
            proj_dim=self.model_config.get("proj_dim", 128),
            hidden_dim=self.model_config.get("hidden_dim", 64),
            num_layers=self.model_config.get("num_layers", 1),
            num_classes=self.num_classes,
            dropout=self.model_config.get("dropout", 0.3)
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._extractor: Optional[ModernMediaPipeExtractor] = None

    @property
    def extractor(self) -> ModernMediaPipeExtractor:
        if self._extractor is None:
            self._extractor = ModernMediaPipeExtractor()
        return self._extractor

    def normalize_torso_landmarks(self, keypoints_75x4: np.ndarray) -> np.ndarray:
        norm_kps = keypoints_75x4.copy()
        ls = norm_kps[11, :3]
        rs = norm_kps[12, :3]
        ls_vis = norm_kps[11, 3]
        rs_vis = norm_kps[12, 3]

        if ls_vis > 0.3 and rs_vis > 0.3:
            midpoint = (ls + rs) / 2.0
            shoulder_dist = float(np.linalg.norm(ls - rs))
            scale = shoulder_dist if shoulder_dist > 1e-3 else 1.0
        else:
            nose = norm_kps[0, :3]
            nose_vis = norm_kps[0, 3]
            midpoint = nose if nose_vis > 0.3 else np.array([0.5, 0.5, 0.0], dtype=np.float32)
            scale = 1.0

        for i in range(75):
            if norm_kps[i, 3] > 0.0:
                norm_kps[i, :3] = (norm_kps[i, :3] - midpoint) / scale
            else:
                norm_kps[i, :3] = 0.0
                norm_kps[i, 3] = 0.0

        return norm_kps

    def predict_features(
        self,
        features: np.ndarray,
        confidence_threshold: float = 0.25,
        min_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Run inference on preprocessed (T, 300) normalized feature array.
        """
        t0 = time.perf_counter()
        t, d = features.shape
        if d != 300:
            raise ValueError(f"Expected feature dim 300, got {d}")

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
                "model_identifier": "bigru_25class"
            }

        tensor_x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)  # (1, T, 300)

        with torch.no_grad():
            logits, attn_weights = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])
        pred_sentence = self.class_names[pred_idx]

        # Top 5 candidate sorting
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_candidates = [
            {
                "class_id": int(idx),
                "sentence": self.class_names[idx],
                "probability": float(probs[idx])
            }
            for idx in top5_indices
        ]

        class_probabilities = {
            self.class_names[i]: float(probs[i])
            for i in range(self.num_classes)
        }

        is_confident = pred_prob >= confidence_threshold
        uncertainty_warning = (
            None if is_confident
            else f"Low confidence prediction ({pred_prob*100:.1f}% < {confidence_threshold*100:.1f}% threshold). Prediction may be uncertain."
        )

        return {
            "success": True,
            "predicted_class_id": pred_idx,
            "predicted_sentence": pred_sentence,
            "confidence": pred_prob,
            "class_probabilities": class_probabilities,
            "top5_candidates": top5_candidates,
            "frame_count": t,
            "is_confident": is_confident,
            "uncertainty_warning": uncertainty_warning,
            "latency_ms": round(latency_ms, 2),
            "model_identifier": "bigru_25class",
            "model_type": "TwentyFiveClassBiGRUAttentionModel",
            "attention_weights": attn_weights.squeeze(0).cpu().numpy().tolist() if attn_weights is not None else []
        }

    def predict_landmarks(
        self,
        landmarks_seq: np.ndarray,
        confidence_threshold: float = 0.25,
        min_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Inference on raw (T, 75, 4) landmark array.
        """
        t, n_pts, n_ch = landmarks_seq.shape
        normalized = []
        for i in range(t):
            norm_frame = self.normalize_torso_landmarks(landmarks_seq[i])
            normalized.append(norm_frame.reshape(-1))
        features = np.array(normalized, dtype=np.float32)
        return self.predict_features(features, confidence_threshold=confidence_threshold, min_frames=min_frames)

    def predict_frames(
        self,
        frames_bgr: List[np.ndarray],
        confidence_threshold: float = 0.25,
        min_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Extract MediaPipe landmarks from RGB/BGR frames and run 25-class BiGRU inference.
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
                "model_identifier": "bigru_25class"
            }

        features = []
        for frame in frames_bgr:
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_75x4, _ = self.extractor.extract_isign_frame(rgb)
            norm_75x4 = self.normalize_torso_landmarks(raw_75x4)
            features.append(norm_75x4.reshape(-1))

        if not features:
            return {
                "success": False,
                "error": "Failed to extract landmarks from any frame.",
                "frame_count": 0,
                "predicted_sentence": "",
                "confidence": 0.0,
                "class_probabilities": {},
                "top5_candidates": [],
                "is_confident": False,
                "model_identifier": "bigru_25class"
            }

        features_arr = np.array(features, dtype=np.float32)
        return self.predict_features(features_arr, confidence_threshold=confidence_threshold, min_frames=min_frames)
