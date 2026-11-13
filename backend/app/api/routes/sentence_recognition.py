"""
API routes for Continuous / Sentence-Level Indian Sign Language Recognition.
Supports landmark sequence arrays, video file uploads, and sentence vocabulary querying.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

try:
    from app.schemas.sentence import (
        SentenceLandmarkRequest,
        SentenceRecognitionResponse,
        SentenceVocabularyItem,
        SentenceVocabularyResponse,
        TopSentencePrediction,
        DemoClassesResponse,
        DemoClassItem,
        DemoSentenceRecognitionResponse,
        DemoRankingItem,
        FiveClassSentenceRecognitionResponse,
    )
    from app.services.sentence_inference_service import sentence_inference_service
    from app.services.hackathon_demo_service import hackathon_demo_service
    from app.services.five_class_bigru_service import five_class_bigru_service
except ModuleNotFoundError:
    from backend.app.schemas.sentence import (
        SentenceLandmarkRequest,
        SentenceRecognitionResponse,
        SentenceVocabularyItem,
        SentenceVocabularyResponse,
        TopSentencePrediction,
        DemoClassesResponse,
        DemoClassItem,
        DemoSentenceRecognitionResponse,
        DemoRankingItem,
        FiveClassSentenceRecognitionResponse,
    )
    from backend.app.services.sentence_inference_service import sentence_inference_service
    from backend.app.services.hackathon_demo_service import hackathon_demo_service
    from backend.app.services.five_class_bigru_service import five_class_bigru_service

logger = logging.getLogger("signova.routes.sentence")

router = APIRouter(prefix="/recognize/sentence", tags=["sentence_recognition"])

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("", response_model=SentenceRecognitionResponse)
@router.post("/", response_model=SentenceRecognitionResponse)
@router.post("/landmarks", response_model=SentenceRecognitionResponse)
def recognize_sentence_landmarks(
    request: SentenceLandmarkRequest,
) -> SentenceRecognitionResponse:
    """
    Recognize continuous ISL sentence from a temporal sequence of MediaPipe landmarks.
    Expected landmark shape: (T, 75, 4), (T, 300), or (T, 678).
    """
    if not request.landmarks or len(request.landmarks) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Landmark sequence cannot be empty.",
        )

    try:
        raw_result = sentence_inference_service.predict_landmarks(
            landmarks=request.landmarks,
            timestamps_ms=request.timestamps_ms,
            fps=request.fps,
            top_k=request.top_k,
        )
    except ValueError as exc:
        logger.warning("Invalid landmark sequence input: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Sentence landmark recognition failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    top_predictions = [
        TopSentencePrediction(
            sentence_id=p["sentence_id"],
            sentence=p["sentence"],
            gloss=p.get("gloss"),
            confidence=float(p.get("confidence", 0.0)),
        )
        for p in raw_result.get("top_predictions", [])
    ]

    return SentenceRecognitionResponse(
        success=raw_result["success"],
        predicted_sentence_id=raw_result["predicted_sentence_id"],
        sentence=raw_result["sentence"],
        gloss=raw_result.get("gloss"),
        confidence=raw_result["confidence"],
        top_predictions=top_predictions,
        processing_time_ms=raw_result["processing_time_ms"],
        model_used=raw_result["model_used"],
        status=raw_result["status"],
        reason=raw_result.get("reason"),
    )


@router.post("/video", response_model=SentenceRecognitionResponse)
def recognize_sentence_video(
    video: UploadFile = File(..., description="Uploaded sign video (.mp4, .webm, .avi, .mov, .mkv)"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top candidates to return"),
) -> SentenceRecognitionResponse:
    """
    Extract landmarks from an uploaded continuous ISL signing video and recognize the sentence.
    """
    filename = video.filename or "input.mp4"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    try:
        video_bytes = video.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded video stream: {exc}",
        )

    if len(video_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty.",
        )

    if len(video_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )

    actual_top_k = int(getattr(top_k, "default", top_k)) if hasattr(top_k, "default") or isinstance(top_k, (int, float, str)) else 3

    try:
        raw_result = sentence_inference_service.predict_video(
            video_bytes=video_bytes,
            filename=filename,
            top_k=actual_top_k,
        )
    except Exception as exc:
        logger.error("Sentence video recognition failure for '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    top_predictions = [
        TopSentencePrediction(
            sentence_id=p["sentence_id"],
            sentence=p["sentence"],
            gloss=p.get("gloss"),
            confidence=float(p.get("confidence", 0.0)),
        )
        for p in raw_result.get("top_predictions", [])
    ]

    return SentenceRecognitionResponse(
        success=raw_result["success"],
        predicted_sentence_id=raw_result["predicted_sentence_id"],
        sentence=raw_result["sentence"],
        gloss=raw_result.get("gloss"),
        confidence=raw_result["confidence"],
        top_predictions=top_predictions,
        processing_time_ms=raw_result["processing_time_ms"],
        model_used=raw_result["model_used"],
        status=raw_result["status"],
        reason=raw_result.get("reason"),
    )


@router.get("/vocabulary", response_model=SentenceVocabularyResponse)
@router.get("/sentences", response_model=SentenceVocabularyResponse)
def get_sentence_vocabulary() -> SentenceVocabularyResponse:
    """
    Get the list of all supported 101 continuous ISL sentence classes.
    """
    vocab = sentence_inference_service.get_vocabulary()
    items = [
        SentenceVocabularyItem(
            sentence_id=item["sentence_id"],
            sentence=item["sentence"],
            gloss=item.get("gloss"),
            tokens=item.get("tokens"),
        )
        for item in vocab
    ]
    return SentenceVocabularyResponse(
        total=len(items),
        sentences=items,
    )


@router.get("/demo/classes", response_model=DemoClassesResponse)
def get_demo_classes() -> DemoClassesResponse:
    """
    Get the 5 supported continuous ISL sentence classes for Hackathon Demo Mode.
    """
    raw_classes = hackathon_demo_service.get_classes()
    items = [
        DemoClassItem(
            demo_class_id=c["demo_class_id"],
            original_sentence_id=c["original_sentence_id"],
            sentence=c["sentence"],
            isl_gloss=c["isl_gloss"],
        )
        for c in raw_classes
    ]
    return DemoClassesResponse(
        total=len(items),
        classes=items,
    )


@router.post("/demo/video", response_model=DemoSentenceRecognitionResponse)
def recognize_demo_sentence_video(
    video: UploadFile = File(..., description="Uploaded sign video (.mp4, .webm, .avi, .mov, .mkv)"),
    threshold: float = Query(0.40, ge=0.1, le=1.0, description="Rejection distance threshold (dist > threshold rejects)"),
    min_margin: float = Query(0.015, ge=0.0, le=0.5, description="Minimum margin between rank 1 and 2"),
) -> DemoSentenceRecognitionResponse:
    """
    Recognize ISL sentence using the 5-class DTW-KNN model with MediaPipe extraction and unknown input rejection.
    """
    filename = video.filename or "input.mp4"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    try:
        video_bytes = video.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded video stream: {exc}",
        )

    if len(video_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty.",
        )

    if len(video_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )

    try:
        raw_result = hackathon_demo_service.predict_video(
            video_bytes=video_bytes,
            filename=filename,
            threshold=threshold,
            min_margin=min_margin,
        )
    except Exception as exc:
        logger.error("Demo sentence recognition error for '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )

    rankings = [
        DemoRankingItem(
            class_id=r["class_id"],
            original_sentence_id=r["original_sentence_id"],
            sentence=r["sentence"],
            isl_gloss=r["isl_gloss"],
            dtw_distance=r["dtw_distance"],
            similarity=r["similarity"],
        )
        for r in raw_result.get("rankings", [])
    ]

    return DemoSentenceRecognitionResponse(
        success=raw_result["success"],
        is_recognized=raw_result["is_recognized"],
        status=raw_result["status"],
        predicted_sentence=raw_result["predicted_sentence"],
        predicted_class_id=raw_result["predicted_class_id"],
        original_sentence_id=raw_result.get("original_sentence_id"),
        top_similarity=raw_result["top_similarity"],
        top_distance=raw_result["top_distance"],
        runner_up_distance=raw_result.get("runner_up_distance"),
        margin_to_second=raw_result["margin_to_second"],
        matching_latency_ms=raw_result["matching_latency_ms"],
        processing_time_ms=raw_result["processing_time_ms"],
        rankings=rankings,
        model_used=raw_result["model_used"],
        note=raw_result["note"],
    )


@router.get("/five-class/classes", response_model=DemoClassesResponse)
def get_five_class_classes() -> DemoClassesResponse:
    """
    Get the 5 supported continuous ISL sentence classes for the five-class sequence models.
    """
    raw_classes = five_class_bigru_service.get_classes()
    if not raw_classes:
        raw_classes = hackathon_demo_service.get_classes()

    items = [
        DemoClassItem(
            demo_class_id=c.get("class_id", c.get("demo_class_id", i)),
            original_sentence_id=c.get("original_sentence_id", i),
            sentence=c.get("sentence", c.get("name", "")),
            isl_gloss=c.get("isl_gloss", c.get("sentence", "").upper()),
        )
        for i, c in enumerate(raw_classes)
    ]
    return DemoClassesResponse(
        total=len(items),
        classes=items,
    )


CANONICAL_FIVE_CLASSES = {
    0: "comb your hair",
    1: "help me",
    2: "i am fine. thank you sir.",
    3: "i need water",
    4: "who are you",
}


@router.post("/five-class/video", response_model=FiveClassSentenceRecognitionResponse)
def recognize_five_class_video(
    video: UploadFile = File(..., description="Uploaded or live webcam recorded video (.mp4, .webm, .avi, .mov)"),
    model: str = Query("bigru", pattern="^(bigru|dtw)$", description="Recognition model architecture: 'bigru' or 'dtw'"),
    confidence_threshold: float = Query(0.35, ge=0.1, le=1.0, description="Softmax confidence rejection threshold (BiGRU)"),
) -> FiveClassSentenceRecognitionResponse:
    """
    Recognize 5-class ISL sentence from live webcam recording or uploaded video.
    Selectable between BiGRU Attention Model and DTW-KNN.
    Guarantees strict 5-class output mapping, zero silent fallbacks, and diagnostic logging.
    """
    filename = video.filename or "webcam_gesture.webm"
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    try:
        video_bytes = video.file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded video stream: {exc}",
        )

    if len(video_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded video file is empty.",
        )

    if len(video_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit.",
        )

    try:
        if model == "bigru":
            actual_model = "FiveClassBiGRUAttentionModel"
            checkpoint_path = str(getattr(five_class_bigru_service, "checkpoint_path", five_class_bigru_service.model_dir / "best_model.pt"))

            raw_result = five_class_bigru_service.predict_video(
                video_bytes=video_bytes,
                filename=filename,
                confidence_threshold=confidence_threshold,
            )

            is_rec = raw_result["is_recognized"]
            pred_id = raw_result.get("predicted_class_id")
            pred_sentence = raw_result.get("predicted_sentence")
            frame_count = raw_result.get("frame_count", 0)
            class_probs = raw_result.get("class_probabilities", {})

            # Strict 5-class output verification (Never return "go and sleep" or 101-class outputs)
            if pred_id is not None:
                if pred_id not in CANONICAL_FIVE_CLASSES:
                    raise ValueError(f"BiGRU predicted invalid class ID {pred_id} outside 5-class mapping.")
                pred_sentence = CANONICAL_FIVE_CLASSES[pred_id]

            # Diagnostic logging (Section 5)
            logger.info(
                "[DIAGNOSTIC] Sentence Recognition Request:\n"
                "  Requested model: %s\n"
                "  Actual loaded model: %s\n"
                "  Model checkpoint path: %s\n"
                "  API endpoint: %s\n"
                "  Number of extracted frames: %d\n"
                "  Predicted class ID: %s\n"
                "  Predicted sentence: %s\n"
                "  All five class probabilities: %s",
                model,
                actual_model,
                checkpoint_path,
                "/api/v1/recognize/sentence/five-class/video",
                frame_count,
                str(pred_id),
                str(pred_sentence),
                class_probs,
            )

            return FiveClassSentenceRecognitionResponse(
                success=raw_result["success"],
                is_recognized=is_rec,
                status=raw_result["status"],
                predicted_sentence=pred_sentence,
                predicted_class_id=pred_id,
                original_sentence_id=raw_result.get("original_sentence_id"),
                confidence=raw_result["confidence"],
                class_probabilities=class_probs,
                frame_count=frame_count,
                processing_time_ms=raw_result["processing_time_ms"],
                model_inference_latency_ms=raw_result.get("model_inference_latency_ms"),
                model_used=actual_model,
                reason=raw_result.get("reason"),
                margin_to_second=raw_result.get("margin_to_second"),
            )
        else:
            # DTW-KNN model path
            actual_model = "DTWKNNClassifier-5Class"
            checkpoint_path = str(getattr(hackathon_demo_service, "dataset_path", hackathon_demo_service.demo_dir / "reference_dataset.pkl"))

            dtw_result = hackathon_demo_service.predict_video(
                video_bytes=video_bytes,
                filename=filename,
            )

            is_rec = dtw_result["is_recognized"]
            pred_id = dtw_result.get("predicted_class_id")
            pred_sentence = dtw_result.get("predicted_sentence")
            frame_count = dtw_result.get("frame_count", 0)

            # Map DTW rankings to canonical 5-class pseudo-probabilities
            class_probs = {}
            for r in dtw_result.get("rankings", []):
                cid = r["class_id"]
                c_name = CANONICAL_FIVE_CLASSES.get(cid, r["sentence"])
                class_probs[c_name] = round(float(r.get("similarity", 0.0)), 4)

            # Ensure all 5 canonical classes present in class_probs
            for cid, c_name in CANONICAL_FIVE_CLASSES.items():
                if c_name not in class_probs:
                    class_probs[c_name] = 0.0

            # Strict 5-class output verification (Never return "go and sleep" or 101-class outputs)
            if pred_id is None and dtw_result.get("rankings"):
                pred_id = dtw_result["rankings"][0].get("class_id")
            if pred_id is not None:
                if pred_id not in CANONICAL_FIVE_CLASSES:
                    raise ValueError(f"DTW-KNN predicted invalid class ID {pred_id} outside 5-class mapping.")
                pred_sentence = CANONICAL_FIVE_CLASSES[pred_id]

            # Diagnostic logging (Section 5)
            logger.info(
                "[DIAGNOSTIC] Sentence Recognition Request:\n"
                "  Requested model: %s\n"
                "  Actual loaded model: %s\n"
                "  Model checkpoint path: %s\n"
                "  API endpoint: %s\n"
                "  Number of extracted frames: %d\n"
                "  Predicted class ID: %s\n"
                "  Predicted sentence: %s\n"
                "  All five class probabilities: %s",
                model,
                actual_model,
                checkpoint_path,
                "/api/v1/recognize/sentence/five-class/video",
                frame_count,
                str(pred_id),
                str(pred_sentence),
                class_probs,
            )

            formatted_rankings = []
            for r in dtw_result.get("rankings", []):
                cid = r["class_id"]
                formatted_rankings.append(
                    DemoRankingItem(
                        class_id=cid,
                        original_sentence_id=r["original_sentence_id"],
                        sentence=CANONICAL_FIVE_CLASSES.get(cid, r["sentence"]),
                        isl_gloss=r["isl_gloss"],
                        dtw_distance=r["dtw_distance"],
                        similarity=r["similarity"],
                    )
                )

            return FiveClassSentenceRecognitionResponse(
                success=dtw_result["success"],
                is_recognized=is_rec,
                status=dtw_result["status"],
                predicted_sentence=pred_sentence,
                predicted_class_id=pred_id,
                original_sentence_id=dtw_result.get("original_sentence_id"),
                confidence=dtw_result["top_similarity"],
                class_probabilities=class_probs,
                frame_count=frame_count,
                processing_time_ms=dtw_result["processing_time_ms"],
                model_inference_latency_ms=dtw_result.get("matching_latency_ms"),
                model_used=actual_model,
                reason=dtw_result.get("note"),
                top_distance=dtw_result.get("top_distance"),
                runner_up_distance=dtw_result.get("runner_up_distance"),
                margin_to_second=dtw_result.get("margin_to_second"),
                matching_latency_ms=dtw_result.get("matching_latency_ms"),
                rankings=formatted_rankings,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Five-class recognition error (%s) for '%s': %s", model, filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )


@router.post("/five-class/landmarks", response_model=FiveClassSentenceRecognitionResponse)
def recognize_five_class_landmarks(
    request: SentenceLandmarkRequest,
    model: str = Query("bigru", pattern="^(bigru|dtw)$", description="Recognition model architecture: 'bigru' or 'dtw'"),
    confidence_threshold: float = Query(0.35, ge=0.1, le=1.0, description="Softmax confidence rejection threshold (BiGRU)"),
) -> FiveClassSentenceRecognitionResponse:
    """
    Recognize 5-class ISL sentence from pre-extracted landmark sequence array.
    """
    if not request.landmarks or len(request.landmarks) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Landmark sequence cannot be empty.",
        )

    try:
        if model == "bigru":
            actual_model = "FiveClassBiGRUAttentionModel"
            raw_result = five_class_bigru_service.predict_landmarks(
                landmarks=request.landmarks,
                confidence_threshold=confidence_threshold,
            )
            is_rec = raw_result["is_recognized"]
            pred_id = raw_result.get("predicted_class_id")
            pred_sentence = raw_result.get("predicted_sentence")
            frame_count = raw_result.get("frame_count", 0)
            class_probs = raw_result.get("class_probabilities", {})

            if pred_id is not None and pred_id in CANONICAL_FIVE_CLASSES:
                pred_sentence = CANONICAL_FIVE_CLASSES[pred_id]

            return FiveClassSentenceRecognitionResponse(
                success=raw_result["success"],
                is_recognized=is_rec,
                status=raw_result["status"],
                predicted_sentence=pred_sentence,
                predicted_class_id=pred_id,
                original_sentence_id=raw_result.get("original_sentence_id"),
                confidence=raw_result["confidence"],
                class_probabilities=class_probs,
                frame_count=frame_count,
                processing_time_ms=raw_result["processing_time_ms"],
                model_inference_latency_ms=raw_result.get("model_inference_latency_ms"),
                model_used=actual_model,
                reason=raw_result.get("reason"),
                margin_to_second=raw_result.get("margin_to_second"),
            )
        else:
            actual_model = "DTWKNNClassifier-5Class"
            from scripts.recognize_sentence_dtw import preprocess_sequence
            from scripts.demo_recognize_sentence import recognize_sequence

            arr = np.asarray(request.landmarks, dtype=np.float32)
            if np.all(arr == 0) or float(np.std(arr)) < 1e-4:
                return FiveClassSentenceRecognitionResponse(
                    success=True,
                    is_recognized=False,
                    status="NO_VALID_LANDMARKS",
                    predicted_sentence=None,
                    predicted_class_id=None,
                    original_sentence_id=None,
                    confidence=0.0,
                    class_probabilities={name: 0.0 for name in CANONICAL_FIVE_CLASSES.values()},
                    frame_count=len(arr),
                    processing_time_ms=0.0,
                    model_used=actual_model,
                    reason="No valid landmarks detected in input sequence.",
                    margin_to_second=0.0,
                )

            proc_seq = preprocess_sequence(arr, preprocessors=hackathon_demo_service.preprocessors)
            rec_result = recognize_sequence(
                proc_seq=proc_seq,
                ref_seqs=hackathon_demo_service.ref_seqs,
                ref_labels=hackathon_demo_service.ref_labels,
                classes=hackathon_demo_service.classes,
            )
            is_rec = rec_result["is_recognized"]
            top_r = rec_result.get("rankings", [{}])[0]
            pred_id = rec_result.get("predicted_class_id")
            if pred_id is None and top_r:
                pred_id = top_r.get("class_id")
            pred_sentence = CANONICAL_FIVE_CLASSES.get(pred_id, rec_result.get("predicted_sentence"))

            class_probs = {}
            for r in rec_result.get("rankings", []):
                cid = r["class_id"]
                class_probs[CANONICAL_FIVE_CLASSES.get(cid, r["sentence"])] = round(float(r.get("similarity", 0.0)), 4)
            for cid, c_name in CANONICAL_FIVE_CLASSES.items():
                if c_name not in class_probs:
                    class_probs[c_name] = 0.0

            formatted_rankings = []
            for r in rec_result.get("rankings", []):
                cid = r["class_id"]
                formatted_rankings.append(
                    DemoRankingItem(
                        class_id=cid,
                        original_sentence_id=r["original_sentence_id"],
                        sentence=CANONICAL_FIVE_CLASSES.get(cid, r["sentence"]),
                        isl_gloss=r["isl_gloss"],
                        dtw_distance=r["dtw_distance"],
                        similarity=r["similarity"],
                    )
                )

            return FiveClassSentenceRecognitionResponse(
                success=True,
                is_recognized=is_rec,
                status=rec_result["status"],
                predicted_sentence=pred_sentence,
                predicted_class_id=pred_id,
                original_sentence_id=hackathon_demo_service.classes[pred_id]["original_sentence_id"] if pred_id is not None and pred_id < len(hackathon_demo_service.classes) else None,
                confidence=rec_result["top_similarity"],
                class_probabilities=class_probs,
                frame_count=len(arr),
                processing_time_ms=rec_result["latency_ms"],
                model_used=actual_model,
                reason=rec_result.get("note"),
                margin_to_second=rec_result.get("margin_to_second"),
                top_distance=rec_result.get("top_distance"),
                runner_up_distance=rec_result.get("runner_up_distance"),
                matching_latency_ms=rec_result.get("latency_ms"),
                rankings=formatted_rankings,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Five-class landmark recognition error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )


