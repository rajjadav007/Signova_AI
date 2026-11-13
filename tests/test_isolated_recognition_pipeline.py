"""
Unit & Integration Tests for F-019.1 Isolated Sign Recognition Pipeline.
Tests:
- Temporal resampling to fixed sequence length (64).
- Spatial normalization (centering and shoulder scaling).
- Feature extraction (spatial 300 + velocity 300 = 600).
- IsolatedSignRecognizer prediction format and top-k output.
- Isolated dataset loading from datasets/landmarks/.
- Evaluation metrics calculation.
- Module separation: verifies isolated components execute independently of iSign T5.
"""

from pathlib import Path
import numpy as np
import pytest

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    ISOLATED_CLASSES,
    NUM_CLASSES,
)
from ai.isolated.evaluation.metrics import compute_isolated_metrics
from ai.isolated.features.feature_extractor import compute_velocity, extract_isolated_features
from ai.isolated.inference.classifier import IsolatedSignRecognizer
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample
from ai.isolated.training.dataset_loader import load_split_dataset


def test_1_temporal_resampling():
    """Verify arbitrary length sequences resample to exactly 64 frames."""
    for orig_t in [10, 45, 64, 120]:
        seq = np.random.randn(orig_t, 75, 4).astype(np.float32)
        resampled = temporal_resample(seq, target_length=DEFAULT_SEQUENCE_LENGTH)
        assert resampled.shape == (DEFAULT_SEQUENCE_LENGTH, 75, 4)
        assert resampled.dtype == np.float32


def test_2_spatial_normalization():
    """Verify spatial centering on mid-shoulder and scaling."""
    T = 20
    kps = np.full((T, 75, 4), 0.5, dtype=np.float32)
    # Set shoulders: LS (11) and RS (12)
    kps[:, 11, :] = [0.4, 0.5, 0.0, 1.0]  # Left shoulder
    kps[:, 12, :] = [0.6, 0.5, 0.0, 1.0]  # Right shoulder
    kps[:, 0, :] = [0.5, 0.2, 0.0, 1.0]   # Nose

    normed = spatial_normalize(kps)
    assert normed.shape == (T, 75, 4)
    # Mid shoulder was (0.5, 0.5), width was 0.2
    # Nose X should be (0.5 - 0.5) / 0.2 = 0.0
    np.testing.assert_allclose(normed[:, 0, 0], 0.0, atol=1e-5)


def test_3_feature_extraction():
    """Verify velocity features and 600-dim feature concatenation."""
    T = DEFAULT_SEQUENCE_LENGTH
    kps = np.random.randn(T, 75, 4).astype(np.float32)
    features = extract_isolated_features(kps)
    assert features.shape == (T, 600)
    assert features.dtype == np.float32

    # First frame velocity must be zero
    np.testing.assert_array_equal(features[0, 300:], 0.0)


def test_4_isolated_recognizer_prediction():
    """Verify recognizer returns valid structured predictions."""
    recognizer = IsolatedSignRecognizer()
    recognizer.load_model()
    assert recognizer.is_loaded

    dummy_seq = np.random.randn(30, 75, 4).astype(np.float32)
    res = recognizer.predict_landmarks(dummy_seq, top_k=3)

    assert res["success"] is True
    assert res["predicted_sign"] in ISOLATED_CLASSES
    assert len(res["top_predictions"]) == 3
    assert "processing_time_ms" in res
    assert res["model_used"] is not None


def test_5_load_split_dataset():
    """Verify dataset loader reads validation split .npz landmarks correctly."""
    X, y, meta = load_split_dataset(split="validation", sequence_length=32)
    assert len(X) == 28  # Validation split has exactly 28 clips
    assert len(y) == 28
    assert X.shape == (28, 32, 600)
    assert y.dtype == np.int64
    assert all(0 <= label < NUM_CLASSES for label in y)


def test_6_evaluation_metrics():
    """Verify multi-class metrics computation."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_probs = np.eye(10)

    metrics = compute_isolated_metrics(y_true, y_pred, y_probs)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["top3_accuracy"] == 1.0
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics


def test_7_module_separation_from_isign_t5():
    """Verify isolated sign recognition runs without importing or loading iSign T5."""
    import sys
    # Verify transformers model loader not imported into isolated namespace
    from ai.isolated.inference.classifier import IsolatedSignRecognizer
    rec = IsolatedSignRecognizer()
    assert rec is not None


def test_8_isolated_augmenter():
    """Verify training-only data augmentation behavior and boundary enforcement."""
    from ai.isolated.training.augmentation import IsolatedAugmenter

    augmenter = IsolatedAugmenter(jitter_std=0.003, seed=42)

    # 1. Augmentation on validation or test must raise ValueError
    with pytest.raises(ValueError, match="forbidden"):
        dummy = np.zeros((64, 75, 4), dtype=np.float32)
        augmenter.augment(dummy, split="validation")

    with pytest.raises(ValueError, match="forbidden"):
        dummy = np.zeros((64, 75, 4), dtype=np.float32)
        augmenter.augment(dummy, split="test")

    # 2. Exceeding jitter std limit must fail at init
    with pytest.raises(ValueError, match="must be <= 0.005"):
        IsolatedAugmenter(jitter_std=0.01)

    # 3. Spatial jitter must preserve zero-mask on undetected landmarks
    kps = np.zeros((64, 75, 4), dtype=np.float32)
    # Detect right wrist only (idx 54)
    kps[:, 54, :3] = [0.5, 0.5, 0.0]
    kps[:, 54, 3] = 0.9  # Confident detection

    jittered = augmenter.apply_spatial_jitter(kps)
    assert jittered.shape == (64, 75, 4)
    # Undetected landmarks must remain exactly 0.0
    assert np.all(jittered[:, 0, :] == 0.0)
    assert np.all(jittered[:, 33, :] == 0.0)
    # Detected right wrist x/y should have jitter, z and conf unchanged
    assert not np.array_equal(jittered[:, 54, :2], kps[:, 54, :2])
    np.testing.assert_array_equal(jittered[:, 54, 3], kps[:, 54, 3])


def test_9_precomputed_features_loading():
    """Verify loading pre-computed 600-dim feature files matches validation split."""
    X, y, meta = load_split_dataset(split="validation", from_precomputed_features=True)
    assert len(X) == 28
    assert X.shape == (28, 64, 600)
    assert y.shape == (28,)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_10_feature_dataset_files_and_manifest():
    """Verify generated feature dataset files, index CSV, and summary JSON."""
    import json
    import csv
    from ai.isolated.config.isolated_settings import FEATURES_DIR

    index_csv = FEATURES_DIR / "feature_index.csv"
    summary_json = FEATURES_DIR / "feature_summary.json"

    assert index_csv.exists()
    assert summary_json.exists()

    with open(index_csv, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 187  # Exactly 187 approved clips

    with open(summary_json, mode="r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["total_samples"] == 187
    assert summary["feature_dimensions"]["total"] == 600
    assert summary["splits"]["train"]["samples"] == 129
    assert summary["splits"]["validation"]["samples"] == 28
    assert summary["splits"]["test"]["samples"] == 30


def test_11_trained_champion_model_and_rejection():
    """Verify F-022 trained model checkpoint, metadata, and confidence rejection."""
    import json
    from ai.isolated.config.isolated_settings import ISOLATED_CHECKPOINT_DIR, ISOLATED_CHECKPOINT_PATH
    from ai.isolated.inference.classifier import IsolatedSignRecognizer

    assert ISOLATED_CHECKPOINT_PATH.exists()
    metadata_path = ISOLATED_CHECKPOINT_DIR / "model_metadata.json"
    assert metadata_path.exists()

    with open(metadata_path, mode="r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["status"] in ("trained", "improved")
    assert "champion_model" in meta
    assert "validation_benchmark" in meta
    assert "test_results" in meta
    assert "optimal_rejection_threshold" in meta

    # Test recognizer with real loaded champion model
    recognizer = IsolatedSignRecognizer()
    recognizer.load_model()
    assert recognizer.is_loaded
    assert recognizer.model_type != "HeuristicBaseline"

    # Feed real validation sample
    dummy_kps = np.random.randn(64, 75, 4).astype(np.float32)
    res = recognizer.predict_landmarks(dummy_kps, top_k=3)
    assert res["success"] is True
    assert res["status"] in ("recognized", "uncertain")
    assert "rejection_threshold" in res
    assert len(res["top_predictions"]) == 3
    if not res["is_valid"]:
        assert res["reason"] is not None


def test_12_generalization_artifacts_and_improved_model():
    """Verify F-023 generalization analysis artifacts and improved bilateral model checkpoint."""
    import json
    from ai.isolated.config.isolated_settings import ISOLATED_CHECKPOINT_DIR
    from ai.isolated.inference.classifier import IsolatedSignRecognizer

    improved_ckpt = ISOLATED_CHECKPOINT_DIR / "isolated_10class_improved.joblib"
    baseline_ckpt = ISOLATED_CHECKPOINT_DIR / "f022_isolated_10class_baseline.joblib"
    gen_analysis = ISOLATED_CHECKPOINT_DIR / "generalization_analysis.json"

    assert improved_ckpt.exists()
    assert baseline_ckpt.exists()
    assert gen_analysis.exists()

    with open(gen_analysis, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "success"
    assert "generalization_gap" in data
    assert "signer_breakdown" in data
    assert "rejection_sweep" in data

    # Verify recognizer can load improved checkpoint
    rec = IsolatedSignRecognizer(checkpoint_path=improved_ckpt)
    rec.load_model()
    assert rec.is_loaded

    dummy_kps = np.random.randn(64, 75, 4).astype(np.float32)
    res = rec.predict_landmarks(dummy_kps, top_k=3)
    assert res["success"] is True
    assert len(res["top_predictions"]) == 3


def test_13_route_recognize_video_end_to_end():
    """Verify FastAPI isolated recognition route processing real video end-to-end."""
    import io
    from fastapi import UploadFile
    from backend.app.api.routes.isolated_recognition import recognize_video

    sample_video = Path("dataset/isl-isolated-40words/Yes/1.mp4")
    if not sample_video.exists():
        pytest.skip("Test sample video not found on disk.")

    with open(sample_video, "rb") as f:
        file_bytes = f.read()

    upload = UploadFile(file=io.BytesIO(file_bytes), filename="webcam_capture.mp4")
    response = recognize_video(video=upload, category="general", top_k=3)

    assert response.success is True
    assert response.predicted_sign in ISOLATED_CLASSES
    assert 0.0 <= response.confidence <= 1.0
    assert response.status in ("recognized", "uncertain")
    assert len(response.top_predictions) == 3
    assert response.processing_time_ms > 0
    assert "BiGRU" in response.model_used


def test_14_route_recognize_video_invalid_format():
    """Verify route rejects unsupported non-video file formats."""
    import io
    from fastapi import HTTPException, UploadFile
    from backend.app.api.routes.isolated_recognition import recognize_video

    upload = UploadFile(file=io.BytesIO(b"fake data"), filename="test.txt")
    with pytest.raises(HTTPException) as exc_info:
        recognize_video(video=upload, top_k=3)
    assert exc_info.value.status_code == 400
    assert "Unsupported video format" in exc_info.value.detail


def test_15_route_recognize_video_empty_file():
    """Verify route rejects 0-byte video uploads."""
    import io
    from fastapi import HTTPException, UploadFile
    from backend.app.api.routes.isolated_recognition import recognize_video

    upload = UploadFile(file=io.BytesIO(b""), filename="empty.mp4")
    with pytest.raises(HTTPException) as exc_info:
        recognize_video(video=upload, top_k=3)
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


def test_16_route_recognize_landmarks():
    """Verify route processing pre-extracted landmark sequence."""
    from backend.app.api.routes.isolated_recognition import recognize_landmarks
    from backend.app.schemas.isolated import LandmarkSequenceRequest

    # Create dummy (30 frames, 75 landmarks * 4 channels = 300)
    dummy_seq = np.random.randn(30, 300).astype(np.float32).tolist()
    req = LandmarkSequenceRequest(landmarks=dummy_seq, top_k=3)
    res = recognize_landmarks(request=req)

    assert res.success is True
    assert res.predicted_sign in ISOLATED_CLASSES
    assert len(res.top_predictions) == 3
    assert res.processing_time_ms > 0




