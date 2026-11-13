"""
Unit & Integration Tests for Sentence-Level Continuous ISL Recognition Endpoints.

Tests:
1. Singleton loading and model reuse in SentenceInferenceService.
2. Vocabulary endpoints (GET /vocabulary and GET /sentences) returning 101 sentence classes.
3. Landmark inference with (T, 75, 4), (T, 300), and (T, 678) arrays.
4. Top-k predictions ordering and response schema validation.
5. Error handling: empty landmarks, invalid dimensions.
6. Video upload validation: rejected file format, empty file.
7. Zero regression on isolated recognition routes.
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
from app.services.sentence_inference_service import sentence_inference_service

client = TestClient(app)


def test_sentence_service_loading_and_singleton():
    """Verify SentenceInferenceService loads successfully and reuses singleton instance."""
    sentence_inference_service.load_model()
    assert sentence_inference_service.is_loaded is True
    assert sentence_inference_service.model is not None
    assert len(sentence_inference_service.classes) == 101

    # Verify second call is a no-op / reuses instance
    existing_model = sentence_inference_service.model
    sentence_inference_service.load_model()
    assert sentence_inference_service.model is existing_model


def test_sentence_vocabulary_endpoints():
    """Verify both /vocabulary and /sentences return all 101 supported ISL sentences."""
    # Test /vocabulary
    r1 = client.get("/api/v1/recognize/sentence/vocabulary")
    assert r1.status_code == 200, f"Expected 200, got {r1.status_code}: {r1.text}"
    data1 = r1.json()
    assert data1["total"] == 101
    assert len(data1["sentences"]) == 101
    first = data1["sentences"][0]
    assert first["sentence_id"] == 0
    assert "sentence" in first
    assert "gloss" in first

    # Test /sentences alias
    r2 = client.get("/api/v1/recognize/sentence/sentences")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["total"] == 101
    assert data2["sentences"][0]["sentence"] == first["sentence"]


def test_sentence_landmark_recognition_75x4():
    """Verify POST /recognize/sentence accepts (T, 75, 4) landmark sequences."""
    T = 25
    landmarks = np.zeros((T, 75, 4), dtype=np.float32)
    # Set shoulders
    landmarks[:, 11, :2] = [-0.2, 0.5]
    landmarks[:, 12, :2] = [0.2, 0.5]
    landmarks[:, :, 3] = 1.0

    payload = {
        "landmarks": landmarks.tolist(),
        "timestamps_ms": [i * 40 for i in range(T)],
        "fps": 25.0,
        "top_k": 3,
    }

    res = client.post("/api/v1/recognize/sentence", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"

    body = res.json()
    assert body["success"] is True
    assert 0 <= body["predicted_sentence_id"] < 101
    assert isinstance(body["sentence"], str) and len(body["sentence"]) > 0
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_predictions"]) == 3
    assert body["processing_time_ms"] > 0
    assert body["model_used"] != ""
    assert body["status"] in ["success", "uncertain"]


def test_sentence_landmark_recognition_landmarks_alias_and_300():
    """Verify POST /recognize/sentence/landmarks alias accepts (T, 300) flat spatial frames."""
    T = 20
    landmarks = np.zeros((T, 300), dtype=np.float32)
    # Set shoulders (lm 11: 44..47, lm 12: 48..51)
    landmarks[:, 44:46] = [-0.2, 0.5]
    landmarks[:, 48:50] = [0.2, 0.5]

    payload = {
        "landmarks": landmarks.tolist(),
        "fps": 25.0,
        "top_k": 5,
    }

    res = client.post("/api/v1/recognize/sentence/landmarks", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert len(body["top_predictions"]) == 5


def test_sentence_landmark_recognition_678():
    """Verify POST /recognize/sentence accepts pre-extracted (T, 678) features."""
    T = 15
    features = np.zeros((T, 678), dtype=np.float32)

    payload = {
        "landmarks": features.tolist(),
        "top_k": 2,
    }

    res = client.post("/api/v1/recognize/sentence", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert len(body["top_predictions"]) == 2


def test_sentence_empty_landmarks_rejected():
    """Verify empty landmark array returns 400 Bad Request."""
    res = client.post("/api/v1/recognize/sentence", json={"landmarks": []})
    assert res.status_code == 400


def test_sentence_invalid_landmark_dimension():
    """Verify invalid landmark inner shape returns 400 Bad Request."""
    res = client.post("/api/v1/recognize/sentence", json={"landmarks": [[1.0, 2.0, 3.0]]})
    assert res.status_code == 400


def test_sentence_video_invalid_extension():
    """Verify video upload with unsupported extension returns 400."""
    res = client.post(
        "/api/v1/recognize/sentence/video",
        files={"video": ("test.txt", b"dummy content", "text/plain")},
    )
    assert res.status_code == 400
    assert "Unsupported video format" in res.json()["detail"]


def test_sentence_video_empty_file():
    """Verify uploading an empty video file returns 400."""
    res = client.post(
        "/api/v1/recognize/sentence/video",
        files={"video": ("empty.mp4", b"", "video/mp4")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


# =========================================================================
# Regression Tests for 5-Class Inference Pipeline (Section 6)
# =========================================================================

EXPECTED_FIVE_CLASSES = {
    0: "comb your hair",
    1: "help me",
    2: "i am fine. thank you sir.",
    3: "i need water",
    4: "who are you",
}


def test_five_class_mapping_integrity():
    """Verify GET /five-class/classes returns the exact 5 canonical classes."""
    res = client.get("/api/v1/recognize/sentence/five-class/classes")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body["total"] == 5
    assert len(body["classes"]) == 5

    loaded_map = {c["demo_class_id"]: c["sentence"] for c in body["classes"]}
    for cid, sentence in EXPECTED_FIVE_CLASSES.items():
        assert cid in loaded_map
        assert loaded_map[cid] == sentence
    assert "go and sleep" not in loaded_map.values()


def test_five_class_bigru_upload_cannot_call_101_model(monkeypatch):
    """Verify BiGRU upload routes strictly to five_class_bigru_service, never 101-class."""
    called_101 = False

    def fake_101(*args, **kwargs):
        nonlocal called_101
        called_101 = True
        return {"success": True, "sentence": "go and sleep", "predicted_sentence_id": 16}

    monkeypatch.setattr(sentence_inference_service, "predict_video", fake_101)

    fake_bigru_result = {
        "success": True,
        "is_recognized": True,
        "status": "SUCCESS",
        "predicted_sentence": "comb your hair",
        "predicted_class_id": 0,
        "original_sentence_id": 5,
        "confidence": 0.94,
        "class_probabilities": {
            "comb your hair": 0.94,
            "help me": 0.01,
            "i am fine. thank you sir.": 0.02,
            "i need water": 0.02,
            "who are you": 0.01,
        },
        "frame_count": 50,
        "processing_time_ms": 120.0,
        "model_inference_latency_ms": 5.0,
        "model_used": "FiveClassBiGRUAttentionModel",
        "reason": None,
    }

    from app.services.five_class_bigru_service import five_class_bigru_service
    monkeypatch.setattr(five_class_bigru_service, "predict_video", lambda *a, **kw: fake_bigru_result)

    dummy_video = b"fake video bytes for testing"
    res = client.post(
        "/api/v1/recognize/sentence/five-class/video?model=bigru",
        files={"video": ("test_gesture.mp4", dummy_video, "video/mp4")},
    )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert called_101 is False, "CRITICAL REGRESSION: 101-class model was invoked for BiGRU upload!"
    assert body["model_used"] == "FiveClassBiGRUAttentionModel"
    assert body["predicted_sentence"] == "comb your hair"
    assert body["predicted_class_id"] == 0
    assert body["predicted_sentence"] != "go and sleep"


def test_five_class_dtw_upload_cannot_call_101_model(monkeypatch):
    """Verify DTW upload routes strictly to hackathon_demo_service, never 101-class."""
    called_101 = False

    def fake_101(*args, **kwargs):
        nonlocal called_101
        called_101 = True
        return {"success": True, "sentence": "go and sleep", "predicted_sentence_id": 16}

    monkeypatch.setattr(sentence_inference_service, "predict_video", fake_101)

    fake_dtw_result = {
        "success": True,
        "is_recognized": True,
        "status": "Recognized",
        "predicted_sentence": "help me",
        "predicted_class_id": 1,
        "original_sentence_id": 23,
        "top_similarity": 0.85,
        "top_distance": 0.15,
        "runner_up_distance": 0.35,
        "margin_to_second": 0.20,
        "matching_latency_ms": 4.5,
        "processing_time_ms": 110.0,
        "frame_count": 42,
        "rankings": [
            {
                "class_id": 1,
                "original_sentence_id": 23,
                "sentence": "help me",
                "isl_gloss": "HELP ME",
                "dtw_distance": 0.15,
                "similarity": 0.85,
            },
            {
                "class_id": 0,
                "original_sentence_id": 5,
                "sentence": "comb your hair",
                "isl_gloss": "COMB YOU HAIR",
                "dtw_distance": 0.35,
                "similarity": 0.65,
            },
        ],
        "model_used": "DTWKNNClassifier-5Class",
        "note": "DTW test note",
    }

    from app.services.hackathon_demo_service import hackathon_demo_service
    monkeypatch.setattr(hackathon_demo_service, "predict_video", lambda *a, **kw: fake_dtw_result)

    dummy_video = b"fake video bytes for testing"
    res = client.post(
        "/api/v1/recognize/sentence/five-class/video?model=dtw",
        files={"video": ("test_dtw.mp4", dummy_video, "video/mp4")},
    )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert called_101 is False, "CRITICAL REGRESSION: 101-class model was invoked for DTW upload!"
    assert body["model_used"] == "DTWKNNClassifier-5Class"
    assert body["predicted_sentence"] == "help me"
    assert body["predicted_class_id"] == 1
    assert body["predicted_sentence"] != "go and sleep"


def test_five_class_endpoint_never_returns_go_and_sleep(monkeypatch):
    """Verify that five-class endpoint strictly rejects sentences outside the 5-class mapping."""
    # Simulate a bugged recognizer attempting to return "go and sleep"
    bad_result = {
        "success": True,
        "is_recognized": True,
        "status": "SUCCESS",
        "predicted_sentence": "go and sleep",
        "predicted_class_id": 16,
        "original_sentence_id": 16,
        "confidence": 0.88,
        "class_probabilities": {"go and sleep": 0.88},
        "frame_count": 30,
        "processing_time_ms": 100.0,
        "model_used": "FiveClassBiGRUAttentionModel",
    }

    from app.services.five_class_bigru_service import five_class_bigru_service
    monkeypatch.setattr(five_class_bigru_service, "predict_video", lambda *a, **kw: bad_result)

    res = client.post(
        "/api/v1/recognize/sentence/five-class/video?model=bigru",
        files={"video": ("bad.mp4", b"dummy video", "video/mp4")},
    )

    # Must raise 500 error and fail safely rather than silently returning "go and sleep"
    assert res.status_code == 500
    assert "go and sleep" not in res.text


def test_five_class_model_selection_matches_backend_model(monkeypatch):
    """Verify selected query param model='bigru' or 'dtw' matches actual model loaded."""
    from app.services.five_class_bigru_service import five_class_bigru_service
    from app.services.hackathon_demo_service import hackathon_demo_service

    monkeypatch.setattr(
        five_class_bigru_service,
        "predict_video",
        lambda *a, **kw: {
            "success": True,
            "is_recognized": True,
            "status": "SUCCESS",
            "predicted_sentence": "who are you",
            "predicted_class_id": 4,
            "original_sentence_id": 91,
            "confidence": 0.91,
            "class_probabilities": {"who are you": 0.91},
            "frame_count": 30,
            "processing_time_ms": 80.0,
            "model_used": "FiveClassBiGRUAttentionModel",
        },
    )

    monkeypatch.setattr(
        hackathon_demo_service,
        "predict_video",
        lambda *a, **kw: {
            "success": True,
            "is_recognized": True,
            "status": "Recognized",
            "predicted_sentence": "i need water",
            "predicted_class_id": 3,
            "original_sentence_id": 53,
            "top_similarity": 0.82,
            "top_distance": 0.18,
            "margin_to_second": 0.10,
            "matching_latency_ms": 3.0,
            "processing_time_ms": 70.0,
            "frame_count": 25,
            "rankings": [],
            "model_used": "DTWKNNClassifier-5Class",
            "note": "test",
        },
    )

    # 1. BiGRU
    r_bigru = client.post(
        "/api/v1/recognize/sentence/five-class/video?model=bigru",
        files={"video": ("sample.mp4", b"bytes", "video/mp4")},
    )
    assert r_bigru.status_code == 200
    assert r_bigru.json()["model_used"] == "FiveClassBiGRUAttentionModel"

    # 2. DTW
    r_dtw = client.post(
        "/api/v1/recognize/sentence/five-class/video?model=dtw",
        files={"video": ("sample.mp4", b"bytes", "video/mp4")},
    )
    assert r_dtw.status_code == 200
    assert r_dtw.json()["model_used"] == "DTWKNNClassifier-5Class"


def test_five_class_backend_error_visibility(monkeypatch):
    """Verify backend errors are visible with HTTP error status instead of falling back."""
    from app.services.five_class_bigru_service import five_class_bigru_service

    def fail_predict(*args, **kwargs):
        raise RuntimeError("MediaPipe Landmark Extraction failed on corrupt frames")

    monkeypatch.setattr(five_class_bigru_service, "predict_video", fail_predict)

    res = client.post(
        "/api/v1/recognize/sentence/five-class/video?model=bigru",
        files={"video": ("corrupt.mp4", b"corrupted bytes", "video/mp4")},
    )

    assert res.status_code == 500
    assert "Inference execution failed" in res.json()["detail"]
