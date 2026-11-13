"""
Service managing Sentence-Level Continuous Indian Sign Language Recognition inference.
Provides landmark and video sequence classification into 101 continuous ISL sentences.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

try:
    from ai.continuous.temporal_features import extract_temporal_features
    from ai.continuous.sentence_classifier import TemporalSentenceClassifier
except ImportError:
    # Relative fallback if root is not in sys.path
    import sys
    _p = Path(__file__).resolve().parent.parent.parent.parent
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
    from ai.continuous.temporal_features import extract_temporal_features
    from ai.continuous.sentence_classifier import TemporalSentenceClassifier

logger = logging.getLogger("signova.sentence_service")


class SentenceInferenceService:
    """
    Singleton inference service for ISL sentence-level continuous recognition.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.model_checkpoint_path = (
            self.project_root
            / "models"
            / "isl_csltr_continuous"
            / "best_temporal_sentence_classifier.pt"
        )
        self.sentence_classes_path = (
            self.project_root
            / "models"
            / "isl_csltr_continuous"
            / "sentence_classes.json"
        )
        self.manifest_path = (
            self.project_root
            / "datasets"
            / "isl_csltr_continuous"
            / "manifests"
            / "manifest.csv"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TemporalSentenceClassifier] = None
        self.is_loaded: bool = False
        self.model_used: str = "TemporalSentenceClassifier"
        self.confidence_threshold: float = 0.20

        # Sentence vocabulary mappings
        self.classes: List[Dict[str, Any]] = []
        self.id_to_sentence: Dict[int, str] = {}
        self.id_to_gloss: Dict[int, str] = {}
        self.id_to_tokens: Dict[int, List[str]] = {}

        self._extractor: Optional[Any] = None
        self._extractor_lock = threading.Lock()

        # Pre-load vocabulary metadata upon initialization
        self.load_sentence_mapping()

    def load_sentence_mapping(self) -> None:
        """Loads the 101 sentence classes mapping from JSON or CSV manifest."""
        if self.sentence_classes_path.exists():
            try:
                with open(self.sentence_classes_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_classes = data.get("classes", [])
                self.classes = []
                for item in raw_classes:
                    sid = int(item["sentence_id"])
                    sentence = str(item["sentence"])
                    gloss = item.get("isl_gloss") or item.get("gloss")
                    tokens = item.get("tokens")
                    self.classes.append({
                        "sentence_id": sid,
                        "sentence": sentence,
                        "gloss": gloss,
                        "tokens": tokens,
                    })
                    self.id_to_sentence[sid] = sentence
                    self.id_to_gloss[sid] = gloss or sentence.upper()
                    if tokens:
                        self.id_to_tokens[sid] = tokens
                logger.info("Loaded %d sentence classes from %s", len(self.classes), self.sentence_classes_path)
                return
            except Exception as exc:
                logger.warning("Could not read sentence_classes.json: %s. Falling back to manifest.", exc)

        # Fallback to manifest.csv
        if self.manifest_path.exists():
            try:
                seen_ids = set()
                self.classes = []
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sid = int(row["sentence_id"])
                        if sid not in seen_ids:
                            seen_ids.add(sid)
                            sentence = row.get("sentence", "").strip()
                            gloss = row.get("isl_gloss", "").strip()
                            tokens_str = row.get("tokens", "[]")
                            try:
                                tokens = eval(tokens_str) if tokens_str.startswith("[") else []
                            except Exception:
                                tokens = []
                            self.classes.append({
                                "sentence_id": sid,
                                "sentence": sentence,
                                "gloss": gloss,
                                "tokens": tokens,
                            })
                            self.id_to_sentence[sid] = sentence
                            self.id_to_gloss[sid] = gloss or sentence.upper()
                            self.id_to_tokens[sid] = tokens
                self.classes.sort(key=lambda x: x["sentence_id"])
                logger.info("Loaded %d sentence classes from %s", len(self.classes), self.manifest_path)
                return
            except Exception as exc:
                logger.error("Failed to load sentence classes from manifest.csv: %s", exc)

        # Emergency fallback: generate placeholder 101 classes if files are absent
        if not self.classes:
            for sid in range(101):
                s = f"Sentence {sid}"
                self.classes.append({"sentence_id": sid, "sentence": s, "gloss": s.upper(), "tokens": []})
                self.id_to_sentence[sid] = s
                self.id_to_gloss[sid] = s.upper()

    def load_model(self) -> None:
        """
        Instantiates TemporalSentenceClassifier and loads checkpoint weights.
        Falls back to initialized architecture if checkpoint is not yet generated.
        """
        if self.is_loaded and self.model is not None:
            return

        num_classes = len(self.classes) if self.classes else 101
        v2_path = (
            self.project_root
            / "models"
            / "isl_csltr_continuous"
            / "best_temporal_sentence_classifier_v2.pt"
        )
        target_path = v2_path if v2_path.exists() else self.model_checkpoint_path

        self.model = TemporalSentenceClassifier(
            input_dim=678,
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=2,
            dropout=0.4,
        )

        if target_path.exists():
            try:
                checkpoint = torch.load(target_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint)
                self.model_used = f"TemporalSentenceClassifier ({target_path.name})"
                logger.info("Successfully loaded sentence classifier checkpoint from %s", target_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load checkpoint state dict from %s: %s. Using initialized fallback.",
                    target_path,
                    exc,
                )
                self.model_used = "TemporalSentenceClassifier (fallback/eval)"
        else:
            self.model_used = "TemporalSentenceClassifier (fallback/eval)"
            logger.info(
                "Checkpoint '%s' not found on disk (training in progress). Running in initialized fallback mode.",
                target_path.name,
            )

        self.model.to(self.device).eval()
        self.is_loaded = True

    def get_vocabulary(self) -> List[Dict[str, Any]]:
        """Returns all supported sentence classes."""
        if not self.classes:
            self.load_sentence_mapping()
        return self.classes

    def predict_landmarks(
        self,
        landmarks: Union[np.ndarray, List[Any]],
        timestamps_ms: Optional[Union[np.ndarray, List[int]]] = None,
        fps: float = 25.0,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Recognize an ISL sentence from landmark sequences.
        Accepts: (T, 75, 4), (T, 300), or (T, 678) sequences.
        """
        t0 = time.perf_counter()

        if not self.is_loaded or self.model is None:
            self.load_model()

        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.size == 0:
            raise ValueError("Input landmark sequence is empty.")

        # Handle nested or flat representations
        if arr.ndim == 1:
            if arr.shape[0] == 678:
                arr = arr.reshape(1, 678)
            elif arr.shape[0] % 678 == 0:
                arr = arr.reshape(-1, 678)
            elif arr.shape[0] % 300 == 0:
                arr = arr.reshape(-1, 75, 4)
            else:
                raise ValueError(f"Cannot reshape 1D landmark array of size {arr.shape[0]} to (T, 300) or (T, 678).")
        elif arr.ndim == 2:
            if arr.shape[1] == 300:
                arr = arr.reshape(-1, 75, 4)
            elif arr.shape[1] == 75 * 4:
                arr = arr.reshape(-1, 75, 4)
            elif arr.shape[1] == 678:
                pass  # already 678 features
            else:
                raise ValueError(f"Unexpected 2D landmark shape {arr.shape}. Expected (T, 300) or (T, 678).")
        elif arr.ndim == 3:
            if arr.shape[1] != 75 or arr.shape[2] < 3:
                raise ValueError(f"Unexpected 3D landmark shape {arr.shape}. Expected (T, 75, 4).")

        # Extract 678-dimensional temporal features
        if arr.ndim == 2 and arr.shape[1] == 678:
            features = arr.astype(np.float32)
        else:
            features = extract_temporal_features(
                landmarks=arr,
                timestamps_ms=timestamps_ms,
                fps=fps,
            )

        T = features.shape[0]
        feat_tensor = torch.from_numpy(features).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        lengths = torch.tensor([T], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred_res = self.model.predict(feat_tensor, lengths=lengths)

        probs = pred_res["probabilities"][0].cpu().numpy()  # (num_classes,)
        k = max(1, min(int(top_k), len(probs)))
        top_indices = np.argsort(probs)[::-1][:k]

        top_predictions: List[Dict[str, Any]] = []
        for idx in top_indices:
            sid = int(idx)
            top_predictions.append({
                "sentence_id": sid,
                "sentence": self.id_to_sentence.get(sid, f"Sentence {sid}"),
                "gloss": self.id_to_gloss.get(sid),
                "confidence": round(float(probs[sid]), 4),
            })

        best = top_predictions[0]
        conf = best["confidence"]
        is_empty_or_static = bool(np.all(arr == 0) or float(np.std(arr)) < 1e-4)
        if is_empty_or_static:
            status = "uncertain"
            reason = "No hand movement or landmarks detected in video. Please ensure hands are clearly visible in the camera frame."
        else:
            status = "success" if conf >= self.confidence_threshold else "uncertain"
            reason = None if status == "success" else f"Confidence ({conf:.2f}) below threshold {self.confidence_threshold:.2f}"

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "success": True,
            "predicted_sentence_id": best["sentence_id"],
            "sentence": best["sentence"],
            "gloss": best["gloss"],
            "confidence": conf,
            "top_predictions": top_predictions,
            "processing_time_ms": elapsed_ms,
            "model_used": self.model_used,
            "status": status,
            "reason": reason,
        }

    def predict_video(
        self,
        video_bytes: bytes,
        filename: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Extract landmarks from video bytes using ModernMediaPipeExtractor and run sentence recognition.
        """
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            if self._extractor is None:
                with self._extractor_lock:
                    if self._extractor is None:
                        from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor
                        self._extractor = ModernMediaPipeExtractor()

            raw_spatial, meta = self._extractor.extract_isign_video(tmp_path)
            landmarks_75x4 = raw_spatial.reshape(-1, 75, 4)
            fps = float(meta.get("fps", 25.0))

            return self.predict_landmarks(
                landmarks=landmarks_75x4,
                fps=fps,
                top_k=top_k,
            )
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass


sentence_inference_service = SentenceInferenceService()
