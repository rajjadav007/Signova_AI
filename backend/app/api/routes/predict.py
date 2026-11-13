import asyncio
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

try:
    from app.schemas.predict import LandmarkSequencePredictionRequest, PredictionResponse
    from app.services.inference_service import POSE_EXTENSIONS, VIDEO_EXTENSIONS, inference_service
except ImportError:
    from backend.app.schemas.predict import LandmarkSequencePredictionRequest, PredictionResponse
    from backend.app.services.inference_service import POSE_EXTENSIONS, VIDEO_EXTENSIONS, inference_service

logger = logging.getLogger("signova.routes.predict")

router = APIRouter(prefix="/predict", tags=["prediction"])

ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | POSE_EXTENSIONS
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit


@router.post("/video", response_model=PredictionResponse)
async def predict_video(
    video: UploadFile = File(..., description="Uploaded video (.mp4, .mov, .webm, .avi, .mkv) or .pose file"),
    decoding: Literal["beam", "greedy"] = Query("beam", description="T5 text decoding strategy (beam or greedy)"),
) -> PredictionResponse:
    """
    Process an uploaded video or .pose file through the official iSign T5 translation pipeline.
    """
    filename = video.filename or "input.mp4"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    try:
        file_bytes = await video.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file stream: {exc}",
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )

    # Offload heavy synchronous ML inference to worker thread pool
    try:
        raw_result = await asyncio.to_thread(
            inference_service.predict_file,
            file_bytes=file_bytes,
            filename=filename,
            decoding=decoding,
        )
    except RuntimeError as rerr:
        logger.warning("Continuous translation unavailable: %s", rerr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(rerr),
        )
    except Exception as exc:
        logger.error("Inference failure for '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    predicted_text = raw_result.get("prediction", "")
    timings = raw_result.get("timings", {})
    duration_sec = timings.get("total_processing_time_seconds", 0.0)
    device_name = raw_result.get("device", {}).get("name", inference_service.device_info)

    return PredictionResponse(
        success=True,
        predicted_text=predicted_text,
        decoding=decoding,
        inference_time_seconds=duration_sec,
        video_name=filename,
        device=device_name,
    )


@router.post("/landmarks", response_model=PredictionResponse)
async def predict_landmarks(
    request: LandmarkSequencePredictionRequest,
) -> PredictionResponse:
    """
    Process a 75-landmark sequence (300 coordinates per frame) through the official iSign T5 simple model.
    Expected landmark shape: (T, 300) or (T, 75, 4).
    """
    if len(request.landmarks) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Landmark sequence is empty.",
        )

    try:
        raw_result = await asyncio.to_thread(
            inference_service.predict_landmarks,
            landmarks=request.landmarks,
            decoding=request.decoding,
        )
    except RuntimeError as rerr:
        logger.warning("Continuous translation unavailable: %s", rerr)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(rerr),
        )
    except Exception as exc:
        logger.error("Landmark inference failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    predicted_text = raw_result.get("prediction", "")
    timings = raw_result.get("timings", {})
    duration_sec = timings.get("total_processing_time_seconds", 0.0)
    device_name = raw_result.get("device", {}).get("name", inference_service.device_info)

    return PredictionResponse(
        success=True,
        predicted_text=predicted_text,
        decoding=request.decoding,
        inference_time_seconds=duration_sec,
        video_name="client_landmarks_stream",
        device=device_name,
    )
