"""
Service for Signova AI 5-Class Hackathon Demo Recognition.

Wraps the 5-class DTW-KNN model and reference dataset for backend API inference.
Reuses existing MediaPipe landmark extraction, preprocessing, and rejection logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("signova.hackathon_demo_service")


class HackathonDemoService:
    """
    Singleton service managing 5-Class ISL Continuous Sentence Recognition.
    """

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.demo_dir = (
            self.project_root
            / "models"
            / "isl_csltr_continuous"
            / "hackathon_demo"
        )
        self.classes_path = self.demo_dir / "demo_classes.json"
        self.dataset_path = self.demo_dir / "reference_dataset.pkl"

        self.classes: List[Dict[str, Any]] = []
        self.references: List[Dict[str, Any]] = []
        self.ref_seqs: List[np.ndarray] = []
        self.ref_labels: np.ndarray = np.array([], dtype=np.int64)
        self.preprocessors: Optional[Dict[str, Any]] = None

        self.is_loaded: bool = False
        self._load_lock = threading.Lock()
        self._extractor: Optional[Any] = None
        self._extractor_lock = threading.Lock()

        # Load metadata and reference dataset
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load 5-class demo metadata, references, and sequences into memory."""
        with self._load_lock:
            if not self.classes_path.exists() or not self.dataset_path.exists():
                logger.warning(
                    "Demo artifacts not found in %s. Demo inference will not be available.",
                    self.demo_dir,
                )
                return

            with open(self.classes_path, "r", encoding="utf-8") as f:
                self.classes = json.load(f)["classes"]

            with open(self.dataset_path, "rb") as f:
                ds = pickle.load(f)

            self.references = ds.get("references", [])
            self.ref_seqs = ds["sequences"]
            self.ref_labels = ds["labels"]

            # Preload preprocessors
            improved_path = (
                self.project_root
                / "datasets"
                / "isl_csltr_continuous"
                / "ml_cache"
                / "preprocessors_improved.pkl"
            )
            if improved_path.exists():
                with open(improved_path, "rb") as f:
                    self.preprocessors = pickle.load(f)

            self.is_loaded = True
            logger.info(
                "HackathonDemoService loaded %d classes, %d reference sequences.",
                len(self.classes),
                len(self.ref_seqs),
            )

    def get_classes(self) -> List[Dict[str, Any]]:
        """Return the list of 5 supported sentence classes."""
        if not self.is_loaded:
            self.load_artifacts()
        return self.classes

    def _get_extractor(self) -> Any:
        """Lazily initialize MediaPipe extractor in a thread-safe manner."""
        if self._extractor is None:
            with self._extractor_lock:
                if self._extractor is None:
                    from ai.pose_extraction.mediapipe_extractor import ModernMediaPipeExtractor

                    self._extractor = ModernMediaPipeExtractor()
        return self._extractor

    def predict_video(
        self,
        video_bytes: bytes,
        filename: str = "upload.mp4",
        threshold: float = 0.40,
        min_margin: float = 0.015,
    ) -> Dict[str, Any]:
        """
        Process uploaded video through MediaPipe landmark extraction,
        Frame PCA preprocessing, and DTW matching against the 10 references.
        """
        if not self.is_loaded:
            self.load_artifacts()
            if not self.is_loaded:
                raise RuntimeError(
                    "Demo model artifacts could not be loaded. Please run "
                    "scripts/build_hackathon_demo_references.py first."
                )

        t_total_start = time.perf_counter()
        suffix = Path(filename).suffix.lower() or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            # 1. Feature extraction using existing pipeline
            from scripts.recognize_sentence_dtw import (
                extract_features_from_video,
                preprocess_sequence,
            )
            from scripts.demo_recognize_sentence import recognize_sequence

            raw_feat = extract_features_from_video(tmp_path)
            frame_count = len(raw_feat)
            proc_seq = preprocess_sequence(raw_feat, preprocessors=self.preprocessors)

            # 2. DTW sequence recognition & rejection evaluation
            rec_result = recognize_sequence(
                proc_seq=proc_seq,
                ref_seqs=self.ref_seqs,
                ref_labels=self.ref_labels,
                classes=self.classes,
                distance_threshold=threshold,
                min_margin=min_margin,
            )

            total_elapsed_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

            rankings = rec_result["rankings"]
            runner_up_dist = rankings[1]["dtw_distance"] if len(rankings) > 1 else None

            # Map to original sentence ID
            orig_sid = None
            if rec_result["is_recognized"] and rec_result["predicted_class_id"] is not None:
                cls_id = rec_result["predicted_class_id"]
                if 0 <= cls_id < len(self.classes):
                    orig_sid = self.classes[cls_id]["original_sentence_id"]

            logger.info(
                "[DIAGNOSTIC - DTW Service] Request Completed:\n"
                "  Requested Model: dtw\n"
                "  Actual Loaded Model: DTWKNNClassifier-5Class\n"
                "  Model Checkpoint Path: %s\n"
                "  Extracted Frames: %d\n"
                "  Predicted Class ID: %s\n"
                "  Predicted Sentence: %s",
                str(self.dataset_path),
                frame_count,
                str(rec_result["predicted_class_id"]),
                str(rec_result["predicted_sentence"]),
            )

            return {
                "success": True,
                "is_recognized": rec_result["is_recognized"],
                "status": rec_result["status"],
                "predicted_sentence": rec_result["predicted_sentence"],
                "predicted_class_id": rec_result["predicted_class_id"],
                "original_sentence_id": orig_sid,
                "top_similarity": rec_result["top_similarity"],
                "top_distance": rec_result["top_distance"],
                "runner_up_distance": runner_up_dist,
                "margin_to_second": rec_result["margin_to_second"],
                "matching_latency_ms": rec_result["latency_ms"],
                "processing_time_ms": total_elapsed_ms,
                "frame_count": frame_count,
                "rankings": rankings,
                "model_used": "DTWKNNClassifier-5Class",
                "note": rec_result["note"],
            }
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass


hackathon_demo_service = HackathonDemoService()
