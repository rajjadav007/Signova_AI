try:
    from app.schemas.health import HealthResponse
    from app.schemas.error import ErrorResponse, ErrorDetail
    from app.schemas.predict import PredictionResponse
    from app.schemas.isolated import (
        IsolatedRecognitionResponse,
        LandmarkSequenceRequest,
        TopPrediction,
    )
    from app.schemas.sarvam import (
        TranslationRequest,
        TranslationResponse,
        TTSRequest,
        TTSResponse,
        SUPPORTED_SARVAM_LANGUAGES,
        SUPPORTED_SARVAM_SPEAKERS,
    )
except ModuleNotFoundError:
    from backend.app.schemas.health import HealthResponse
    from backend.app.schemas.error import ErrorResponse, ErrorDetail
    from backend.app.schemas.predict import PredictionResponse
    from backend.app.schemas.isolated import (
        IsolatedRecognitionResponse,
        LandmarkSequenceRequest,
        TopPrediction,
    )
    from backend.app.schemas.sarvam import (
        TranslationRequest,
        TranslationResponse,
        TTSRequest,
        TTSResponse,
        SUPPORTED_SARVAM_LANGUAGES,
        SUPPORTED_SARVAM_SPEAKERS,
    )

__all__ = [
    "HealthResponse",
    "ErrorResponse",
    "ErrorDetail",
    "PredictionResponse",
    "IsolatedRecognitionResponse",
    "LandmarkSequenceRequest",
    "TopPrediction",
    "TranslationRequest",
    "TranslationResponse",
    "TTSRequest",
    "TTSResponse",
    "SUPPORTED_SARVAM_LANGUAGES",
    "SUPPORTED_SARVAM_SPEAKERS",
]
