"""
Pydantic schemas for Sarvam AI Translation and Text-to-Speech endpoints (FE-08).
"""

from typing import Optional
from pydantic import BaseModel, Field


SUPPORTED_SARVAM_LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ta-IN": "Tamil",
    "bn-IN": "Bengali",
    "te-IN": "Telugu",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
}

SUPPORTED_SARVAM_SPEAKERS = {
    # Sarvam Bulbul v3 / modern voices
    "aditya", "priya", "neha", "rahul", "pooja", "rohan", "simran",
    "anushka", "amit", "dev", "ishita", "shreya", "ratan", "varun",
    "kabir", "aayan", "shubh", "ashutosh", "advait", "anand", "tanya",
    "tarun", "sunny", "mani", "gokul", "vijay", "shruti", "suhani",
    "mohit", "kavitha", "rehan", "soham", "rupali", "niharika",
    # Legacy v1 speaker aliases (mapped transparently in service layer)
    "meera", "arvind", "pavithra", "maitreyi", "amol", "amartya",
}


class TranslationRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=2000, description="Source text to translate")
    target_language_code: str = Field(..., description="Target BCP-47 language code (e.g. 'hi-IN')")
    source_language_code: str = Field("en-IN", description="Source BCP-47 language code (default 'en-IN')")
    model: str = Field("sarvam-translate:v1", description="Sarvam translation model identifier")


class TranslationResponse(BaseModel):
    success: bool = Field(True, description="Whether translation completed successfully")
    translated_text: str = Field(..., description="Translated output string")
    source_language_code: str = Field(..., description="Source language code")
    target_language_code: str = Field(..., description="Target language code")
    processing_time_ms: float = Field(0.0, description="Roundtrip processing time in milliseconds")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2500, description="Text string to convert to speech")
    target_language_code: str = Field(..., description="BCP-47 language code for synthesized voice")
    speaker: str = Field("priya", description="Voice persona identifier")
    pace: float = Field(1.0, ge=0.5, le=2.0, description="Speech pace multiplier")
    model: str = Field("bulbul:v3", description="Sarvam TTS model identifier")


class TTSResponse(BaseModel):
    success: bool = Field(True, description="Whether speech synthesis completed successfully")
    audio_base64: str = Field(..., description="Base64-encoded WAV/MP3 audio payload")
    audio_format: str = Field("wav", description="Audio container format (default 'wav')")
    target_language_code: str = Field(..., description="Language code of spoken voice")
    speaker: str = Field("priya", description="Voice persona used for generation")
    processing_time_ms: float = Field(0.0, description="Roundtrip processing time in milliseconds")


class STTResponse(BaseModel):
    success: bool = Field(True, description="Whether transcription completed successfully")
    transcript: str = Field(..., description="Recognized speech transcript")
    language_code: Optional[str] = Field(None, description="Language code of spoken audio")
    model: str = Field("saaras:v4", description="Sarvam STT model identifier")
    processing_time_ms: float = Field(0.0, description="Roundtrip processing time in milliseconds")


class TranscribeAndTranslateResponse(BaseModel):
    success: bool = Field(True, description="Whether speech-to-text and translation succeeded")
    original_transcript: str = Field(..., description="Recognized speech transcript in Other Person's Language")
    translated_text: str = Field(..., description="Translated text in Your Language")
    source_language_code: str = Field(..., description="Source spoken language code")
    target_language_code: str = Field(..., description="Target readable language code")
    processing_time_ms: float = Field(0.0, description="End-to-end processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Optional error description when success is False")

