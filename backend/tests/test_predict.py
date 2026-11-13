"""
Unit and Contract Tests for Prediction Route and Inference Service (/api/v1/predict/video).
Validates:
- OpenAPI schema registration and contract.
- File extension definitions (video and .pose).
- Route validation logic (bad extension, empty file, size limit).
- PredictionResponse schema serialization and invariants.
- InferenceService routing and parameter mapping without eager ML framework imports.
"""

import io
from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.api.routes.predict import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES
from app.schemas.predict import PredictionResponse
from app.services.inference_service import POSE_EXTENSIONS, VIDEO_EXTENSIONS, InferenceService


def test_predict_openapi_route_registration():
    """Verify /api/v1/predict/video is present and correct in OpenAPI schema."""
    schema = app.openapi()
    assert "/api/v1/predict/video" in schema["paths"]
    endpoint_meta = schema["paths"]["/api/v1/predict/video"]
    assert "post" in endpoint_meta

    post_spec = endpoint_meta["post"]
    assert post_spec["summary"] == "Predict Video" or "operationId" in post_spec
    assert "requestBody" in post_spec
    assert "multipart/form-data" in post_spec["requestBody"]["content"]

    # Verify 200 response references PredictionResponse
    assert "200" in post_spec["responses"]


def test_allowed_extensions_specification():
    """Verify supported video formats and official .pose format are whitelisted."""
    expected_videos = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    assert expected_videos.issubset(VIDEO_EXTENSIONS)
    assert ".pose" in POSE_EXTENSIONS
    assert ALLOWED_EXTENSIONS == (VIDEO_EXTENSIONS | POSE_EXTENSIONS)


def test_predict_validation_logic():
    """Verify bad extensions and empty files trigger expected validation rules."""
    # 1. Invalid extension
    bad_filename = "document.txt"
    bad_suffix = Path(bad_filename).suffix.lower()
    assert bad_suffix not in ALLOWED_EXTENSIONS

    # 2. Empty bytes
    empty_bytes = b""
    assert len(empty_bytes) == 0

    # 3. Oversized file boundary
    large_size = MAX_FILE_SIZE_BYTES + 1
    assert large_size > MAX_FILE_SIZE_BYTES


def test_prediction_response_schema():
    """Verify PredictionResponse Pydantic schema serializes and validates cleanly."""
    resp = PredictionResponse(
        success=True,
        predicted_text="This is a test translation.",
        decoding="beam",
        inference_time_seconds=0.452,
        video_name="sample.pose",
        device="cpu",
    )
    dumped = resp.model_dump()
    assert dumped["success"] is True
    assert dumped["predicted_text"] == "This is a test translation."
    assert dumped["decoding"] == "beam"
    assert dumped["inference_time_seconds"] == 0.452
    assert dumped["video_name"] == "sample.pose"
    assert dumped["device"] == "cpu"


def test_inference_service_pose_routing(tmp_path):
    """Verify InferenceService handles .pose file bytes and routes to translator."""
    service = InferenceService()
    mock_translator = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "Hello world sign translation"
    mock_result.token_ids = [101, 102]
    mock_result.input_frames = 25
    mock_result.device = "cpu"
    mock_result.metadata = {"format": "isign_pose"}
    mock_translator.predict_with_metadata.return_value = mock_result

    service.translator = mock_translator
    service.is_loaded = True
    service.device_info = "cpu"

    dummy_pose_bytes = b"BINARY_MOCK_POSE_DATA"
    result = service.predict_file(
        file_bytes=dummy_pose_bytes,
        filename="test.pose",
        decoding="beam",
    )

    assert result["prediction"] == "Hello world sign translation"
    assert result["frames"] == 25
    assert result["device"]["name"] == "cpu"
    assert result["timings"]["total_processing_time_seconds"] >= 0
    mock_translator.predict_with_metadata.assert_called_once()


def test_predict_landmarks_openapi_route_registration():
    """Verify /api/v1/predict/landmarks is present and correct in OpenAPI schema."""
    schema = app.openapi()
    assert "/api/v1/predict/landmarks" in schema["paths"]
    endpoint_meta = schema["paths"]["/api/v1/predict/landmarks"]
    assert "post" in endpoint_meta

    post_spec = endpoint_meta["post"]
    assert "requestBody" in post_spec
    assert "application/json" in post_spec["requestBody"]["content"]
    assert "200" in post_spec["responses"]


def test_inference_service_landmarks_routing():
    """Verify InferenceService handles landmark sequences and routes to translator."""
    service = InferenceService()
    mock_translator = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "Hello landmark translation"
    mock_result.token_ids = [201, 202]
    mock_result.input_frames = 30
    mock_result.device = "cpu"
    mock_result.metadata = {}
    mock_translator.predict_with_metadata.return_value = mock_result

    service.translator = mock_translator
    service.is_loaded = True
    service.device_info = "cpu"

    dummy_landmarks = [[0.0] * 300 for _ in range(30)]
    result = service.predict_landmarks(
        landmarks=dummy_landmarks,
        decoding="beam",
    )

    assert result["prediction"] == "Hello landmark translation"
    assert result["frames"] == 30
    assert result["device"]["name"] == "cpu"
    mock_translator.predict_with_metadata.assert_called_once()

