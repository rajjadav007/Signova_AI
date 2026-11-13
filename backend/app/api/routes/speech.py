"""
API routes for Sarvam AI Bulbul Text-to-Speech (FE-08).
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

try:
    from app.schemas.sarvam import (
        SUPPORTED_SARVAM_SPEAKERS,
        STTResponse,
        TranscribeAndTranslateResponse,
        TTSRequest,
        TTSResponse,
    )
    from app.services.sarvam_service import sarvam_service
except ModuleNotFoundError:
    from backend.app.schemas.sarvam import (
        SUPPORTED_SARVAM_SPEAKERS,
        STTResponse,
        TranscribeAndTranslateResponse,
        TTSRequest,
        TTSResponse,
    )
    from backend.app.services.sarvam_service import sarvam_service

logger = logging.getLogger("signova.routes.speech")

router = APIRouter(prefix="/speech", tags=["speech"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".webm", ".mp3", ".ogg", ".m4a"}


@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest) -> TTSResponse:
    """
    Synthesizes translated or user-edited text into spoken audio via Sarvam Bulbul TTS.
    """
    return await sarvam_service.synthesize_speech(request)


@router.post("/transcribe", response_model=STTResponse)
async def transcribe_speech(
    audio: UploadFile = File(..., description="Recorded audio file (.wav, .webm, .mp3, .ogg)"),
    language_code: Optional[str] = Query(None, description="Spoken language code (e.g. 'en-IN', 'hi-IN')"),
) -> STTResponse:
    """
    Transcribes spoken audio into text using Sarvam Saaras Speech-to-Text.
    """
    filename = audio.filename or "audio.webm"
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )
    return await sarvam_service.speech_to_text(
        audio_bytes=audio_bytes,
        filename=filename,
        language_code=language_code,
    )


@router.post("/transcribe-and-translate", response_model=TranscribeAndTranslateResponse)
async def transcribe_and_translate_speech(
    audio: UploadFile = File(..., description="Recorded audio file from Other Person (.wav, .webm, .mp3, .ogg)"),
    source_language_code: str = Query("en-IN", description="Language spoken by the other person"),
    target_language_code: str = Query("gu-IN", description="Deaf user's readable language"),
) -> TranscribeAndTranslateResponse:
    """
    Direction B: Transcribes other person's speech and translates to Deaf user's readable language.
    """
    filename = audio.filename or "speech.webm"
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded speech recording is empty.",
        )
    return await sarvam_service.transcribe_and_translate(
        audio_bytes=audio_bytes,
        filename=filename,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )


@router.get("/speakers", response_model=List[str])
def get_supported_speakers() -> List[str]:
    """
    Returns the list of supported Sarvam Bulbul voice speaker names.
    """
    return sorted(SUPPORTED_SARVAM_SPEAKERS)
