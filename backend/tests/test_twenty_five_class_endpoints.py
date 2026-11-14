"""
Unit & Integration Tests for 25-Class Continuous Sentence Recognition Endpoints.

Direct route and service execution avoiding TestClient event loop deadlocks on Windows.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
from fastapi import HTTPException

backend_dir = str(Path(__file__).resolve().parent.parent)
project_root = Path(backend_dir).parent
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.api.routes.sentence_recognition import (
    get_twenty_five_class_classes,
    recognize_twenty_five_class_landmarks,
    get_five_class_classes,
)
from app.schemas.sentence import (
    SentenceLandmarkRequest,
    TwentyFiveClassesResponse,
    TwentyFiveClassSentenceRecognitionResponse,
)
from app.services.twenty_five_class_bigru_service import twenty_five_class_bigru_service


def test_twenty_five_class_service_loading():
    """Verify TwentyFiveClassBiGRUService loads model and 25 classes successfully."""
    twenty_five_class_bigru_service.load_model()
    assert twenty_five_class_bigru_service.is_loaded is True
    assert twenty_five_class_bigru_service.model is not None
    assert len(twenty_five_class_bigru_service.classes) == 25
    assert twenty_five_class_bigru_service.classes[0] == "comb your hair"
    assert twenty_five_class_bigru_service.classes[4] == "who are you"
    assert twenty_five_class_bigru_service.classes[5] == "are you free today"
    assert twenty_five_class_bigru_service.classes[24] == "what are you doing"


def test_twenty_five_class_get_classes_endpoint():
    """Verify get_twenty_five_class_classes returns all 25 classes with canonical mapping."""
    res = get_twenty_five_class_classes()
    assert isinstance(res, TwentyFiveClassesResponse)
    assert res.total == 25
    assert len(res.classes) == 25
    assert res.classes[0].class_id == 0
    assert res.classes[0].sentence == "comb your hair"
    assert res.classes[1].sentence == "help me"
    assert res.classes[2].sentence == "i am fine. thank you sir."
    assert res.classes[3].sentence == "i need water"
    assert res.classes[4].sentence == "who are you"
    assert res.classes[5].sentence == "are you free today"
    assert res.classes[24].sentence == "what are you doing"


def test_twenty_five_class_landmarks_endpoint():
    """Verify recognize_twenty_five_class_landmarks performs inference on valid sequence."""
    T = 25
    dummy_landmarks = np.zeros((T, 75, 4), dtype=np.float32)
    # Set torso points to simulate real coordinate frame
    dummy_landmarks[:, 11, :2] = [-0.2, 0.5]
    dummy_landmarks[:, 12, :2] = [0.2, 0.5]
    dummy_landmarks[:, :, 3] = 1.0

    request = SentenceLandmarkRequest(
        landmarks=dummy_landmarks.tolist(),
        fps=25.0,
        top_k=5,
    )

    res = recognize_twenty_five_class_landmarks(request=request, confidence_threshold=0.25)
    assert isinstance(res, TwentyFiveClassSentenceRecognitionResponse)
    assert res.success is True
    assert res.model_identifier == "bigru_25class"
    assert res.model_used == "TwentyFiveClassBiGRUAttentionModel"
    assert isinstance(res.predicted_sentence, str)
    assert 0 <= res.predicted_class_id < 25
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top5_candidates) == 5
    assert len(res.class_probabilities) == 25

    # Top candidate matches predicted sentence
    top_cand = res.top5_candidates[0]
    assert top_cand.class_id == res.predicted_class_id
    assert top_cand.sentence == res.predicted_sentence

    # Uncertainty handling
    if not res.is_confident:
        assert res.status == "LOW_CONFIDENCE"
        assert res.uncertainty_warning is not None
    else:
        assert res.status == "SUCCESS"
        assert res.uncertainty_warning is None


def test_twenty_five_class_empty_landmarks_error():
    """Verify recognize_twenty_five_class_landmarks raises HTTPException on empty data."""
    request = SentenceLandmarkRequest(
        landmarks=[],
        fps=25.0,
    )
    with pytest.raises(HTTPException) as exc_info:
        recognize_twenty_five_class_landmarks(request=request)
    assert exc_info.value.status_code == 400
    assert "Landmark sequence cannot be empty" in exc_info.value.detail


def test_five_class_endpoint_unaffected():
    """Verify existing 5-class endpoint still returns 5 classes unchanged."""
    res = get_five_class_classes()
    assert res.total == 5
    assert len(res.classes) == 5
    assert res.classes[0].sentence == "comb your hair"
    assert res.classes[4].sentence == "who are you"
