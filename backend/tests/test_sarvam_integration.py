"""
Unit and Integration Tests for Sarvam AI Services & Endpoints (FE-08).
All external HTTP calls are strictly mocked to preserve API credits.
"""

from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

backend_dir = str(Path(__file__).resolve().parent.parent)
project_root = Path(backend_dir).parent
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.api.routes.speech import get_supported_speakers, synthesize_speech, transcribe_speech, transcribe_and_translate_speech
from app.api.routes.translation import get_supported_languages, translate_text
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
from app.services.sarvam_service import SarvamService


@pytest.fixture
def mock_configured_service():
    """Returns a SarvamService configured with a mock API key."""
    return SarvamService(api_key="mock_sarvam_secret_key_12345")


@pytest.fixture
def unconfigured_service():
    """Returns a SarvamService with no API key set."""
    return SarvamService(api_key="")


# ---------------------------------------------------------------------------
# 1. Configuration & Key Validation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_missing_api_key_raises_503(unconfigured_service):
    """Verify that requests fail with 503 when SARVAM_API_KEY is not configured."""
    req = TranslationRequest(
        input="Hello water please",
        target_language_code="hi-IN",
    )
    with pytest.raises(HTTPException) as exc_info:
        await unconfigured_service.translate(req)

    assert exc_info.value.status_code == 503
    assert "SARVAM_API_KEY" in exc_info.value.detail
    assert "mock_sarvam_secret_key" not in exc_info.value.detail


@pytest.mark.anyio
async def test_missing_api_key_tts_raises_503(unconfigured_service):
    """Verify that TTS fails with 503 when SARVAM_API_KEY is not configured."""
    req = TTSRequest(
        text="नमस्ते",
        target_language_code="hi-IN",
        speaker="meera",
    )
    with pytest.raises(HTTPException) as exc_info:
        await unconfigured_service.synthesize_speech(req)

    assert exc_info.value.status_code == 503
    assert "SARVAM_API_KEY" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 2. Input & Language Validation
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_empty_translation_input_rejected(mock_configured_service):
    """Verify that empty or whitespace-only input text is rejected with 400."""
    req = TranslationRequest(
        input="   ",
        target_language_code="hi-IN",
    )
    with pytest.raises(HTTPException) as exc_info:
        await mock_configured_service.translate(req)

    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_unsupported_target_language_rejected(mock_configured_service):
    """Verify that an unsupported target language code returns 400."""
    req = TranslationRequest(
        input="Water please",
        target_language_code="fr-FR",  # French not in Sarvam supported list
    )
    with pytest.raises(HTTPException) as exc_info:
        await mock_configured_service.translate(req)

    assert exc_info.value.status_code == 400
    assert "unsupported" in exc_info.value.detail.lower()
    assert "fr-FR" in exc_info.value.detail


@pytest.mark.anyio
async def test_unsupported_tts_language_rejected(mock_configured_service):
    """Verify that unsupported TTS language code returns 400."""
    req = TTSRequest(
        text="Bonjour",
        target_language_code="de-DE",
        speaker="meera",
    )
    with pytest.raises(HTTPException) as exc_info:
        await mock_configured_service.synthesize_speech(req)

    assert exc_info.value.status_code == 400
    assert "unsupported" in exc_info.value.detail.lower()


def test_schema_validation_constraints():
    """Verify Pydantic schema validation for minimum length and field limits."""
    with pytest.raises(ValidationError):
        TranslationRequest(input="", target_language_code="hi-IN")

    with pytest.raises(ValidationError):
        TTSRequest(text="", target_language_code="hi-IN")


# ---------------------------------------------------------------------------
# 3. Successful Mocked Translation Response
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_successful_mocked_translation(mock_configured_service):
    """Verify successful translation returns well-formed TranslationResponse."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "request_id": "test_req_001",
        "translated_text": "कृपया मुझे पानी चाहिए",
        "source_language_code": "en-IN",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        req = TranslationRequest(
            input="Please water",
            target_language_code="hi-IN",
            source_language_code="en-IN",
        )
        res = await mock_configured_service.translate(req)

        assert isinstance(res, TranslationResponse)
        assert res.success is True
        assert res.translated_text == "कृपया मुझे पानी चाहिए"
        assert res.target_language_code == "hi-IN"
        assert res.source_language_code == "en-IN"
        assert res.processing_time_ms >= 0.0

        # Verify payload sent to Sarvam
        mock_post.assert_called_once()
        sent_json = mock_post.call_args.kwargs.get("json", {})
        assert sent_json["input"] == "Please water"
        assert sent_json["target_language_code"] == "hi-IN"


# ---------------------------------------------------------------------------
# 4. Successful Mocked TTS Response
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_successful_mocked_tts(mock_configured_service):
    """Verify successful Bulbul TTS converts text to base64 audio."""
    mock_audio_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA="
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "request_id": "test_req_tts_002",
        "audios": [mock_audio_b64],
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        req = TTSRequest(
            text="कृपया मुझे पानी चाहिए",
            target_language_code="hi-IN",
            speaker="meera",
        )
        res = await mock_configured_service.synthesize_speech(req)

        assert isinstance(res, TTSResponse)
        assert res.success is True
        assert res.audio_base64 == mock_audio_b64
        assert res.audio_format == "wav"
        assert res.target_language_code == "hi-IN"
        assert res.speaker == "meera"

        # Verify payload sent to Sarvam Bulbul
        mock_post.assert_called_once()
        sent_json = mock_post.call_args.kwargs.get("json", {})
        assert sent_json["text"] == "कृपया मुझे पानी चाहिए"
        assert sent_json["target_language_code"] == "hi-IN"
        assert sent_json["speaker"] in ("meera", "priya")


# ---------------------------------------------------------------------------
# 5. Upstream Provider Error & Timeout Handling
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upstream_timeout_handling(mock_configured_service):
    """Verify that network timeout raises 504 Gateway Timeout."""
    import httpx

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        req = TranslationRequest(
            input="Help please",
            target_language_code="ta-IN",
        )
        with pytest.raises(HTTPException) as exc_info:
            await mock_configured_service.translate(req)

        assert exc_info.value.status_code == 504
        assert "timed out" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_upstream_error_sanitization_no_credentials_leaked(mock_configured_service):
    """Verify that upstream 500 error from Sarvam is safely sanitized without leaking credentials."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"detail": "Internal processing failure in Sarvam cluster."}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        req = TranslationRequest(
            input="Food water",
            target_language_code="gu-IN",
        )
        with pytest.raises(HTTPException) as exc_info:
            await mock_configured_service.translate(req)

        assert exc_info.value.status_code == 502
        # Ensure secret API key is never in detail
        assert "mock_sarvam_secret_key" not in exc_info.value.detail
        assert "Sarvam AI translation error" in exc_info.value.detail


@pytest.mark.anyio
async def test_empty_audio_response_raises_502(mock_configured_service):
    """Verify that upstream returning no audios list triggers 502."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"audios": []}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        req = TTSRequest(
            text="Hello",
            target_language_code="en-IN",
        )
        with pytest.raises(HTTPException) as exc_info:
            await mock_configured_service.synthesize_speech(req)

        assert exc_info.value.status_code == 502
        assert "invalid audio response" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# 6. Route Handlers & Catalogs
# ---------------------------------------------------------------------------

def test_supported_languages_catalog():
    """Verify languages endpoint returns required languages (en, hi, gu, kn, ta)."""
    languages = get_supported_languages()
    assert isinstance(languages, dict)
    assert "en-IN" in languages
    assert "hi-IN" in languages
    assert "gu-IN" in languages
    assert "kn-IN" in languages
    assert "ta-IN" in languages


def test_supported_speakers_catalog():
    """Verify speakers endpoint returns known Bulbul speakers."""
    speakers = get_supported_speakers()
    assert isinstance(speakers, list)
    assert "meera" in speakers
    assert "arvind" in speakers


# ---------------------------------------------------------------------------
# 7. Saaras Speech-to-Text & Direction B Unified Flow
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_missing_api_key_stt_raises_503(unconfigured_service):
    """Verify STT fails with 503 when API key is missing."""
    with pytest.raises(HTTPException) as exc_info:
        await unconfigured_service.speech_to_text(b"RIFF_FAKE_AUDIO", filename="test.wav")
    assert exc_info.value.status_code == 503
    assert "SARVAM_API_KEY" in exc_info.value.detail


@pytest.mark.anyio
async def test_empty_stt_audio_raises_400(mock_configured_service):
    """Verify empty audio bytes trigger 400 Bad Request."""
    with pytest.raises(HTTPException) as exc_info:
        await mock_configured_service.speech_to_text(b"", filename="empty.wav")
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


@pytest.mark.anyio
async def test_successful_mocked_stt(mock_configured_service):
    """Verify successful speech-to-text response parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "transcript": "Hello, how are you?",
        "language_code": "en-IN",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await mock_configured_service.speech_to_text(
            b"RIFF_DUMMY_AUDIO_DATA",
            filename="input.webm",
            language_code="en-IN",
        )

        assert isinstance(res, STTResponse)
        assert res.success is True
        assert res.transcript == "Hello, how are you?"
        assert res.language_code == "en-IN"
        assert res.processing_time_ms >= 0


@pytest.mark.anyio
async def test_successful_mocked_transcribe_and_translate(mock_configured_service):
    """Verify Direction B end-to-end: Speech in English -> Transcribe -> Translate to Gujarati."""
    mock_stt_resp = MagicMock()
    mock_stt_resp.status_code = 200
    mock_stt_resp.json.return_value = {
        "transcript": "Please bring water",
        "language_code": "en-IN",
    }

    mock_translate_resp = MagicMock()
    mock_translate_resp.status_code = 200
    mock_translate_resp.json.return_value = {
        "translated_text": "કૃપા કરીને પાણી લાવો",
        "source_language_code": "en-IN",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [mock_stt_resp, mock_translate_resp]

        res = await mock_configured_service.transcribe_and_translate(
            audio_bytes=b"RIFF_SPEECH_DATA",
            filename="speech.wav",
            source_language_code="en-IN",
            target_language_code="gu-IN",
        )

        assert isinstance(res, TranscribeAndTranslateResponse)
        assert res.success is True
        assert res.original_transcript == "Please bring water"
        assert res.translated_text == "કૃપા કરીને પાણી લાવો"
        assert res.source_language_code == "en-IN"
        assert res.target_language_code == "gu-IN"


@pytest.mark.anyio
async def test_transcribe_and_translate_same_language_skips_translation(mock_configured_service):
    """Verify that when source == target language, translation API call is skipped."""
    mock_stt_resp = MagicMock()
    mock_stt_resp.status_code = 200
    mock_stt_resp.json.return_value = {
        "transcript": "Hello friend",
        "language_code": "en-IN",
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_stt_resp

        res = await mock_configured_service.transcribe_and_translate(
            audio_bytes=b"RIFF_SPEECH_DATA",
            filename="speech.wav",
            source_language_code="en-IN",
            target_language_code="en-IN",
        )

        assert res.success is True
        assert res.original_transcript == "Hello friend"
        assert res.translated_text == "Hello friend"
        # Only STT called once, translate skipped
        assert mock_post.call_count == 1

