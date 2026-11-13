"""
Signova AI - Five-Class ISL Sentence Sequence Recognizer.
Phase 9 Inference Module:
Accepts:
- An ordered list of RGB/BGR image frames (e.g. from webcam or video frames folder)
- OR pre-extracted (T, 75, 4) landmark array
- OR an existing preprocessed (T, 300) feature sequence

Runs:
1. MediaPipe landmark extraction (Pose 33, LH 21, RH 21)
2. Torso-centered coordinate normalization
3. Variable-length BiGRU Attention Model forward pass
4. Softmax probability distribution and prediction

Returns:
- predicted_class_id
- predicted_sentence
- class_probabilities
- frame_count
- latency_ms
"""

import json
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

from ai.continuous.five_class_model import FiveClassBiGRUAttentionModel
from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor


def natural_sort_key(p: Path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', p.name)]


class FiveClassSequenceRecognizer:
    """
    Inference engine for 5-class ISL sentence sequence recognition.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        if model_dir is None:
            self.model_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "five_class_sequence"
        else:
            self.model_dir = Path(model_dir)

        # 1. Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Load configurations & class mapping
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

        # 3. Load Model
        checkpoint_path = self.model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.model = FiveClassBiGRUAttentionModel(
            input_dim=self.model_config.get("input_dim", 300),
            proj_dim=self.model_config.get("proj_dim", 128),
            hidden_dim=self.model_config.get("hidden_dim", 64),
            num_layers=self.model_config.get("num_layers", 1),
            num_classes=self.num_classes,
            dropout=self.model_config.get("dropout", 0.3)
        ).to(self.device)

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        # 4. Lazy Extractor
        self._extractor = None

    @property
    def extractor(self) -> ModernMediaPipeExtractor:
        if self._extractor is None:
            self._extractor = ModernMediaPipeExtractor()
        return self._extractor

    def normalize_landmarks(self, keypoints_75x4: np.ndarray) -> np.ndarray:
        """
        Torso-centered normalization exact match to training preprocessing.
        """
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

    def predict_features(self, sequence_300d: np.ndarray) -> Dict[str, Any]:
        """
        Predict on preprocessed (T, 300) feature sequence.
        """
        t_start = time.perf_counter()
        t_len = len(sequence_300d)
        if t_len == 0:
            raise ValueError("Empty sequence provided for prediction.")

        x = torch.from_numpy(sequence_300d).float().unsqueeze(0).to(self.device)  # (1, T, 300)
        mask = torch.ones(1, t_len, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits, attn = self.model(x, mask=mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        latency = (time.perf_counter() - t_start) * 1000.0

        class_probs = {self.class_names[i]: round(float(probs[i]), 4) for i in range(self.num_classes)}

        return {
            "predicted_class_id": pred_id,
            "predicted_sentence": self.class_names[pred_id],
            "confidence": round(float(probs[pred_id]), 4),
            "class_probabilities": class_probs,
            "frame_count": t_len,
            "latency_ms": round(latency, 2),
            "attention_weights": attn.squeeze(0).cpu().numpy().tolist()
        }

    def predict_frames(self, rgb_frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Predict on an ordered list of RGB image frames (e.g. from webcam capture).
        """
        t_start = time.perf_counter()
        if not rgb_frames:
            raise ValueError("No frames provided for prediction.")

        features_list = []
        valid_landmarks_count = 0
        for frame in rgb_frames:
            raw_75x4, quality_info = self.extractor.extract_isign_frame(frame)
            if quality_info.get("pose") or quality_info.get("left_hand") or quality_info.get("right_hand"):
                valid_landmarks_count += 1
            norm_75x4 = self.normalize_landmarks(raw_75x4)
            features_list.append(norm_75x4.reshape(-1))

        seq_300d = np.array(features_list, dtype=np.float32)
        if valid_landmarks_count == 0 or np.all(seq_300d == 0) or float(np.std(seq_300d)) < 1e-4:
            total_time_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "predicted_class_id": None,
                "predicted_sentence": None,
                "confidence": 0.0,
                "class_probabilities": {name: 0.0 for name in self.class_names},
                "frame_count": len(rgb_frames),
                "latency_ms": 0.0,
                "total_pipeline_time_ms": round(total_time_ms, 2),
                "attention_weights": [],
                "valid_landmarks": False,
                "reason": "No valid pose or hand landmarks detected in video.",
            }

        result = self.predict_features(seq_300d)
        result["valid_landmarks"] = True
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        result["total_pipeline_time_ms"] = round(total_time_ms, 2)
        return result

    def predict_landmarks(self, landmarks_seq: Union[List[Any], np.ndarray]) -> Dict[str, Any]:
        """
        Predict on a sequence of MediaPipe landmarks: shape (T, 75, 4) or (T, 300).
        """
        t_start = time.perf_counter()
        arr = np.asarray(landmarks_seq, dtype=np.float32)
        if len(arr) == 0:
            raise ValueError("Empty landmark sequence provided.")

        if np.all(arr == 0) or float(np.std(arr)) < 1e-4:
            total_time_ms = (time.perf_counter() - t_start) * 1000.0
            return {
                "predicted_class_id": None,
                "predicted_sentence": None,
                "confidence": 0.0,
                "class_probabilities": {name: 0.0 for name in self.class_names},
                "frame_count": len(arr),
                "latency_ms": 0.0,
                "total_pipeline_time_ms": round(total_time_ms, 2),
                "attention_weights": [],
                "valid_landmarks": False,
                "reason": "No valid landmarks detected in input sequence.",
            }

        if arr.ndim == 3 and arr.shape[1] == 75 and arr.shape[2] == 4:
            features = []
            for frame_75x4 in arr:
                norm_75x4 = self.normalize_landmarks(frame_75x4)
                features.append(norm_75x4.reshape(-1))
            seq_300d = np.array(features, dtype=np.float32)
        elif arr.ndim == 2 and arr.shape[1] == 300:
            seq_300d = arr
        else:
            raise ValueError(f"Invalid landmark dimensions: expected (T, 75, 4) or (T, 300), got {arr.shape}")

        result = self.predict_features(seq_300d)
        result["valid_landmarks"] = True
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        result["total_pipeline_time_ms"] = round(total_time_ms, 2)
        return result

    def predict_folder(self, folder_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predict on a directory containing chronologically numbered JPG frames.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        all_files = sorted(list(folder.glob("*")), key=natural_sort_key)
        frame_files = [f for f in all_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        if not frame_files:
            raise ValueError(f"No valid image files found in {folder}")

        rgb_frames = []
        for fp in frame_files:
            bgr = cv2.imread(str(fp))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_frames.append(rgb)

        return self.predict_frames(rgb_frames)
