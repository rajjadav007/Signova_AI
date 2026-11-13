"""
Focused Backend Tests for Production Isolated ISL Recognition (F-034).
Tests:
- Checkpoint loading and model instance reuse (singleton).
- Valid landmark sequence prediction.
- Accepted prediction structure (status='recognized', is_valid=True, predicted_label).
- Low-confidence rejection (status='uncertain', is_valid=False, predicted_label='uncertain').
- All-zero landmark input handling.
- Empty landmark sequence handling.
- Invalid landmark dimensions handling.
- Top-3 candidate output format.
- Continuous inference pipeline regression protection.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

backend_dir = str(Path(__file__).resolve().parent.parent)
project_root = Path(backend_dir).parent
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.main import app
from app.services.isolated_inference_service import isolated_inference_service

client = TestClient(app)


def test_checkpoint_loading_and_model_reuse():
    """Verify model loads successfully and reuses the existing instance."""
    isolated_inference_service.load_model()
    assert isolated_inference_service.is_loaded is True
    assert isolated_inference_service.recognizer is not None
    assert isolated_inference_service.recognizer.model_type == "GRUClassifierWrapper"
    assert isolated_inference_service.recognizer.rejection_threshold == 0.80

    # Ensure second call does not re-instantiate or overwrite
    first_instance = isolated_inference_service.recognizer
    isolated_inference_service.load_model()
    assert isolated_inference_service.recognizer is first_instance


def test_valid_landmark_accepted_prediction():
    """Verify genuine sign landmark input produces accepted prediction (status='recognized', is_valid=True)."""
    # Use real held-out test sample ISL10_0136 ('hello' by User015)
    sample_npz = project_root / "datasets" / "landmarks" / "test" / "ISL10_0136.npz"
    if not sample_npz.exists():
        pytest.skip(f"Test sample {sample_npz} not found.")

    data = np.load(sample_npz)
    kps = data["landmarks"].reshape(-1, 300).tolist()  # (T, 300) list

    res = client.post(
        "/api/v1/recognize/isolated/landmarks",
        json={"landmarks": kps, "top_k": 3},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is True
    assert body["predicted_label"] == "hello"
    assert body["predicted_sign"] == "hello"
    assert body["confidence"] >= 0.80
    assert body["status"] == "recognized"
    assert body["is_valid"] is True
    assert body["rejection_threshold"] == 0.80
    assert body["reason"] is None
    assert body["processing_time_ms"] > 0


def test_top_3_output_structure():
    """Verify top_3_predictions contains exactly 3 candidates with required fields."""
    sample_npz = project_root / "datasets" / "landmarks" / "test" / "ISL10_0136.npz"
    if not sample_npz.exists():
        pytest.skip(f"Test sample {sample_npz} not found.")

    data = np.load(sample_npz)
    kps = data["landmarks"].reshape(-1, 300).tolist()

    res = client.post(
        "/api/v1/recognize/isolated/landmarks",
        json={"landmarks": kps, "top_k": 3},
    )
    assert res.status_code == 200
    body = res.json()

    top3 = body["top_3_predictions"]
    assert len(top3) == 3
    for pred in top3:
        assert "label" in pred
        assert "confidence" in pred
        assert "display_name" in pred
        assert 0.0 <= pred["confidence"] <= 1.0

    # Ensure sorted descending
    confs = [p["confidence"] for p in top3]
    assert confs == sorted(confs, reverse=True)


def test_low_confidence_rejection():
    """Verify uncertain/ambiguous input triggers status='uncertain' and is_valid=False."""
    # Synthetic random noise produces low confidence across classes
    np.random.seed(42)
    noise = np.random.randn(64, 300).astype(np.float32).tolist()

    res = client.post(
        "/api/v1/recognize/isolated/landmarks",
        json={"landmarks": noise, "top_k": 3},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["status"] == "uncertain"
    assert body["is_valid"] is False
    assert body["predicted_label"] == "uncertain"
    assert body["confidence"] < 0.80
    assert body["reason"] is not None
    assert "below rejection threshold" in body["reason"]


def test_all_zero_landmark_input():
    """Verify all-zero landmark array returns safe invalid response."""
    zeros = np.zeros((30, 300), dtype=np.float32).tolist()

    res = client.post(
        "/api/v1/recognize/isolated/landmarks",
        json={"landmarks": zeros, "top_k": 3},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is False
    assert body["status"] == "uncertain"
    assert body["is_valid"] is False
    assert body["predicted_label"] == "uncertain"
    assert body["confidence"] == 0.0
    assert "No valid landmarks" in body["reason"]


def test_empty_landmark_sequence():
    """Verify empty landmark list returns safe invalid response without crashing."""
    res = client.post(
        "/api/v1/recognize/isolated/landmarks",
        json={"landmarks": [], "top_k": 3},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is False
    assert body["status"] == "uncertain"
    assert body["is_valid"] is False
    assert body["predicted_label"] == "uncertain"
    assert "Empty landmark sequence" in body["reason"]


def test_invalid_dimensions():
    """Verify non-2D/3D landmark payload returns safe invalid response."""
    # 1D list instead of sequence of frames
    res = client.post(
        "/api/v1/recognize/isolated/landmarks",
        json={"landmarks": [1.0, 2.0, 3.0], "top_k": 3},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["success"] is False
    assert body["status"] == "uncertain"
    assert body["is_valid"] is False
    assert "Invalid landmark dimensions" in body["reason"]


def test_continuous_t5_inference_regression_check():
    """Verify continuous T5 predict route returns 200 when connected, or 503 when disconnected."""
    synthetic_landmarks = np.random.randn(20, 300).tolist()
    res = client.post(
        "/api/v1/predict/landmarks",
        json={"landmarks": synthetic_landmarks, "decoding": "beam"},
    )
    assert res.status_code in (200, 503)
    if res.status_code == 200:
        body = res.json()
        assert body["success"] is True
        assert isinstance(body["predicted_text"], str)
    else:
        assert res.status_code == 503
        err_msg = res.json().get("detail") or res.json().get("error", {}).get("message", "")
        assert "unavailable" in err_msg.lower()
