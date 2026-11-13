"""
Sarvam AI Service Layer (FE-08).
Manages backend interactions with Sarvam AI Translation and Bulbul Text-to-Speech APIs.
Ensures API keys are never exposed to clients or logged.
"""

import logging
import time
from typing import Any, Dict, Optional
from fastapi import HTTPException, status
import httpx

try:
    from app.core.config import settings
    from app.schemas.sarvam import (
        SUPPORTED_SARVAM_LANGUAGES,
        SUPPORTED_SARVAM_SPEAKERS,
        STTResponse,
        TranscribeAndTranslateResponse,
        TranslationRequest,
        TranslationResponse,
        TTSRequest,
        TTSResponse,
    )
except ModuleNotFoundError:
    from backend.app.core.config import settings
    from backend.app.schemas.sarvam import (
        SUPPORTED_SARVAM_LANGUAGES,
        SUPPORTED_SARVAM_SPEAKERS,
        STTResponse,
        TranscribeAndTranslateResponse,
        TranslationRequest,
        TranslationResponse,
        TTSRequest,
        TTSResponse,
    )

logger = logging.getLogger("signova.services.sarvam")


class SarvamService:
    """Service client for external Sarvam AI translation and TTS endpoints."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: Optional[float] = None) -> None:
        self._api_key = api_key if api_key is not None else settings.SARVAM_API_KEY
        self._base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self._timeout = timeout if timeout is not None else settings.SARVAM_TIMEOUT_SECONDS

    @property
    def is_configured(self) -> bool:
        """Returns True if a non-empty SARVAM_API_KEY is configured."""
        return bool(self._api_key and self._api_key.strip())

    def _ensure_api_key(self) -> str:
        """Enforces presence of API key; raises 503 if missing."""
        key = self._api_key if self._api_key is not None else settings.SARVAM_API_KEY
        if not key or not key.strip():
            logger.warning("Sarvam API request attempted but SARVAM_API_KEY is not configured.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Sarvam AI service is not configured. Please set SARVAM_API_KEY in the backend environment (.env).",
            )
        return key.strip()

    def _get_headers(self) -> Dict[str, str]:
        """Constructs headers without exposing secrets to callers or loggers."""
        key = self._ensure_api_key()
        return {
            "api-subscription-key": key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _handle_upstream_error(self, response: httpx.Response, service_name: str) -> None:
        """Parses upstream error and raises mapped HTTPException with sanitized message."""
        logger.error("Sarvam %s API error status: %d", service_name, response.status_code)
        try:
            err_data = response.json()
            detail_msg = (
                err_data.get("detail")
                or err_data.get("message")
                or (err_data.get("error", {}).get("message") if isinstance(err_data.get("error"), dict) else None)
                or f"Upstream error {response.status_code}"
            )
        except Exception:
            detail_msg = f"Upstream {service_name} error with status code {response.status_code}."

        lower_msg = str(detail_msg).lower()
        # API credit exhaustion detection (HTTP 402 or keywords)
        if response.status_code == 402 or any(k in lower_msg for k in ("credit", "quota", "balance", "insufficient_fund", "payment")):
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Sarvam AI credit limit exhausted. Please top up API credits to continue.",
            )
        # Rate limit detection
        if response.status_code == 429 or "rate limit" in lower_msg:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Sarvam AI rate limit exceeded. Please wait a moment before trying again.",
            )
        # Auth / key failure
        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Sarvam AI authentication failed. Please verify your API key.",
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Sarvam AI {service_name} error: {detail_msg}",
        )

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translates input text using Sarvam Translate API.
        """
        text = request.input.strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input text cannot be empty.",
            )

        source_lang = (request.source_language_code or "en-IN").strip()
        if source_lang not in SUPPORTED_SARVAM_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported source language '{source_lang}'. Supported: {sorted(SUPPORTED_SARVAM_LANGUAGES.keys())}",
            )

        target_lang = request.target_language_code.strip()
        if target_lang not in SUPPORTED_SARVAM_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported target language '{target_lang}'. Supported: {sorted(SUPPORTED_SARVAM_LANGUAGES.keys())}",
            )

        headers = self._get_headers()
        payload = {
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "model": request.model,
        }

        url = f"{self._base_url}/translate"
        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            logger.error("Sarvam translation request timed out after %.1fs", self._timeout)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Sarvam translation request timed out after {self._timeout} seconds.",
            )
        except httpx.RequestError as exc:
            logger.error("Network error communicating with Sarvam AI: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Unable to reach Sarvam AI translation service.",
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if response.status_code != 200:
            self._handle_upstream_error(response, "translation")

        data = response.json()
        translated_text = data.get("translated_text", "")
        if not translated_text:
            translated_text = text

        return TranslationResponse(
            success=True,
            translated_text=translated_text,
            source_language_code=data.get("source_language_code", source_lang),
            target_language_code=target_lang,
            processing_time_ms=elapsed_ms,
        )

    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """
        Converts input text to speech using Sarvam Bulbul TTS API.
        """
        text = request.text.strip()
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text to convert to speech cannot be empty.",
            )

        target_lang = request.target_language_code.strip()
        if target_lang not in SUPPORTED_SARVAM_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported TTS language '{target_lang}'. Supported: {sorted(SUPPORTED_SARVAM_LANGUAGES.keys())}",
            )

        speaker = request.speaker.strip().lower()
        if speaker not in SUPPORTED_SARVAM_SPEAKERS:
            speaker = "priya"

        # Map legacy v1 voice names to modern v3 speakers for Sarvam API compatibility
        legacy_speaker_map = {
            "meera": "priya",
            "arvind": "aditya",
            "pavithra": "pooja",
            "maitreyi": "neha",
            "amol": "rahul",
            "amartya": "amit",
        }
        sarvam_speaker = legacy_speaker_map.get(speaker, speaker)
        sarvam_model = "bulbul:v3" if request.model in ("bulbul:v1", "bulbul:v2", "", None) else request.model

        headers = self._get_headers()
        payload = {
            "text": text,
            "target_language_code": target_lang,
            "speaker": sarvam_speaker,
            "pace": request.pace,
            "speech_sample_rate": 8000,
            "enable_preprocessing": True,
            "model": sarvam_model,
        }

        url = f"{self._base_url}/text-to-speech"
        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            logger.error("Sarvam TTS request timed out after %.1fs", self._timeout)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Sarvam TTS request timed out after {self._timeout} seconds.",
            )
        except httpx.RequestError as exc:
            logger.error("Network error communicating with Sarvam AI TTS: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Unable to reach Sarvam AI text-to-speech service.",
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if response.status_code != 200:
            self._handle_upstream_error(response, "text-to-speech")

        data = response.json()
        audios = data.get("audios") or []
        if not audios or not isinstance(audios, list) or len(audios) == 0:
            logger.error("Sarvam TTS returned response without audio data: %s", list(data.keys()))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Sarvam AI returned an invalid audio response (no audio stream).",
            )

        audio_base64 = audios[0]

        return TTSResponse(
            success=True,
            audio_base64=audio_base64,
            audio_format="wav",
            target_language_code=target_lang,
            speaker=speaker,
            processing_time_ms=elapsed_ms,
        )

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: Optional[str] = None,
        model: str = "saaras:v4",
    ) -> STTResponse:
        """
        Transcribes speech audio using Sarvam Saaras STT API.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio data cannot be empty.",
            )

        key = self._ensure_api_key()
        headers = {
            "api-subscription-key": key,
            "Accept": "application/json",
        }

        # Determine MIME type from filename extension
        fn_lower = filename.lower()
        if fn_lower.endswith(".webm"):
            content_type = "audio/webm"
        elif fn_lower.endswith(".mp3"):
            content_type = "audio/mpeg"
        elif fn_lower.endswith(".ogg"):
            content_type = "audio/ogg"
        else:
            content_type = "audio/wav"

        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data_fields = {
            "model": model,
            "mode": "transcribe",
        }
        if language_code and language_code in SUPPORTED_SARVAM_LANGUAGES:
            data_fields["language_code"] = language_code

        url = f"{self._base_url}/speech-to-text"
        t0 = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, files=files, data=data_fields)
        except httpx.TimeoutException:
            logger.error("Sarvam STT request timed out after %.1fs", self._timeout)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Sarvam speech-to-text request timed out after {self._timeout} seconds.",
            )
        except httpx.RequestError as exc:
            logger.error("Network error communicating with Sarvam AI STT: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Unable to reach Sarvam AI speech-to-text service.",
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if response.status_code != 200:
            self._handle_upstream_error(response, "speech-to-text")

        resp_data = response.json()
        transcript = resp_data.get("transcript", "").strip()

        return STTResponse(
            success=True,
            transcript=transcript,
            language_code=resp_data.get("language_code", language_code),
            model=model,
            processing_time_ms=elapsed_ms,
        )

    async def transcribe_and_translate(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        source_language_code: str = "en-IN",
        target_language_code: str = "gu-IN",
        model: str = "saaras:v4",
    ) -> TranscribeAndTranslateResponse:
        """
        Executes unified speech-to-text followed by translation into target language.
        """
        if source_language_code not in SUPPORTED_SARVAM_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported speech source language '{source_language_code}'. Supported: {sorted(SUPPORTED_SARVAM_LANGUAGES.keys())}",
            )
        if target_language_code not in SUPPORTED_SARVAM_LANGUAGES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported target language '{target_language_code}'. Supported: {sorted(SUPPORTED_SARVAM_LANGUAGES.keys())}",
            )

        t0 = time.perf_counter()

        stt_result = await self.speech_to_text(
            audio_bytes=audio_bytes,
            filename=filename,
            language_code=source_language_code,
            model=model,
        )

        transcript = stt_result.transcript
        if not transcript:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return TranscribeAndTranslateResponse(
                success=False,
                original_transcript="",
                translated_text="",
                error_message="No speech detected in audio recording. Please speak clearly into the microphone.",
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                processing_time_ms=elapsed_ms,
            )

        # If source and target are the same language, no translation needed
        if source_language_code == target_language_code:
            translated_text = transcript
        else:
            trans_request = TranslationRequest(
                input=transcript,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            trans_result = await self.translate(trans_request)
            translated_text = trans_result.translated_text

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return TranscribeAndTranslateResponse(
            success=True,
            original_transcript=transcript,
            translated_text=translated_text,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            processing_time_ms=elapsed_ms,
        )


sarvam_service = SarvamService()
