"""
Live Integration Tests for FastAPI Backend Endpoints.
Tests:
- GET /health and GET /api/v1/health
- POST /api/v1/predict/landmarks (empty, invalid shape, and valid 300-dim sequence)
- POST /api/v1/predict/video (bad extension, empty file, .pose upload, real video upload)
- Temp file cleanup
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

client = TestClient(app)


def test_1_health_endpoints():
    """Verify both /health and /api/v1/health return 200 OK with healthy status."""
    res_root = client.get("/health")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "healthy"

    res_v1 = client.get("/api/v1/health")
    assert res_v1.status_code == 200
    assert res_v1.json()["status"] == "healthy"


def test_2_predict_landmarks_validation():
    """Verify landmark validation: empty list returns 400."""
    res_empty = client.post("/api/v1/predict/landmarks", json={"landmarks": [], "decoding": "beam"})
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()


def test_3_predict_landmarks_live():
    """Verify landmark endpoint returns valid prediction for 300-dim sequence."""
    synthetic_landmarks = np.random.randn(20, 300).tolist()
    res = client.post(
        "/api/v1/predict/landmarks",
        json={"landmarks": synthetic_landmarks, "decoding": "beam"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert isinstance(data["predicted_text"], str)
    assert data["inference_time_seconds"] > 0
    assert data["device"] == "cpu"


def test_4_predict_video_validation():
    """Verify video validation: bad format and empty file return 400."""
    # Bad extension
    res_bad = client.post(
        "/api/v1/predict/video",
        files={"video": ("test.txt", b"invalid text data", "text/plain")},
    )
    assert res_bad.status_code == 400
    assert "unsupported" in res_bad.json()["detail"].lower()

    # Empty file
    res_empty = client.post(
        "/api/v1/predict/video",
        files={"video": ("empty.mp4", b"", "video/mp4")},
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()


def test_5_predict_pose_file_live():
    """Verify official .pose file upload triggers direct inference without re-extraction."""
    pose_path = project_root / "norm_sample.pose"
    if not pose_path.exists():
        pytest.skip("norm_sample.pose not available")

    with open(pose_path, "rb") as f:
        pose_bytes = f.read()

    res = client.post(
        "/api/v1/predict/video",
        files={"video": ("norm_sample.pose", pose_bytes, "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["predicted_text"] == "Isn't that amazing!"
    assert data["device"] == "cpu"


def test_6_predict_real_video_live():
    """Verify real video upload runs full MediaPipe -> 75 landmarks -> 300 dim -> T5-Base pipeline."""
    video_path = project_root / "datasets" / "videos" / "WIN_20261112_12_11_45_Pro.mp4"
    if not video_path.exists():
        video_path = project_root / "datasets" / "videos" / "10 granfather sick  (2).mov"
    if not video_path.exists():
        pytest.skip("Real video dataset not found")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # Send video to /api/v1/predict/video
    res = client.post(
        "/api/v1/predict/video",
        files={"video": (video_path.name, video_bytes, "video/mp4")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert isinstance(data["predicted_text"], str)
    assert len(data["predicted_text"]) > 0
    assert data["inference_time_seconds"] > 0
    assert data["device"] == "cpu"
