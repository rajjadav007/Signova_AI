from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Resolve project root relative to backend/app/core
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Load local .env file if present
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
elif (BACKEND_DIR / ".env").exists():
    load_dotenv(BACKEND_DIR / ".env")

DEFAULT_MODEL_PATH = PROJECT_ROOT / "research" / "ISLiM" / "checkpoints" / "ISLiM_full_posestitch_pre_processed.pt"
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "research" / "ISLiM" / "assets" / "t5-small"

# iSign T5 Defaults (F-012)
DEFAULT_ISIGN_MODEL_DIR = Path("D:/iSign-t5-pose-to-text")
DEFAULT_ISIGN_TOKENIZER_DIR = PROJECT_ROOT / "models" / "tokenizer" / "t5-small"

# Ensure MPLCONFIGDIR is set to a writable temp path
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TEMP", "C:/Temp"))


def _parse_cors_origins() -> List[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "")
    origins: List[str] = [
        o.strip()
        for o in raw_origins.split(",")
        if o.strip() and o.strip() != "*"
    ]
    frontend_origin = os.getenv("FRONTEND_URL", "http://localhost:5173").strip()
    if frontend_origin and frontend_origin != "*" and frontend_origin not in origins:
        origins.append(frontend_origin)
    # Ensure local development origins are present
    if "http://localhost:5173" not in origins:
        origins.append("http://localhost:5173")
    if "http://127.0.0.1:5173" not in origins:
        origins.append("http://127.0.0.1:5173")
    return origins


@dataclass(frozen=True)
class Settings:
    # Application & Environment
    APP_NAME: str = os.getenv("APP_NAME", "Signova AI")
    PROJECT_NAME: str = "Signova AI API"
    APP_ENV: str = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "development"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    # API Metadata
    API_TITLE: str = "Signova AI API"
    API_DESCRIPTION: str = "Signova AI API - Indian Sign Language (ISL) recognition and translation services."
    API_VERSION: str = "0.1.0"
    API_V1_STR: str = os.getenv("API_V1_PREFIX", os.getenv("API_V1_STR", "/api/v1"))

    # Frontend Origin & CORS
    FRONTEND_ORIGIN: str = os.getenv("FRONTEND_URL", "http://localhost:5173")
    CORS_ORIGINS: List[str] = field(default_factory=_parse_cors_origins)
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = field(
        default_factory=lambda: ["*"]
    )
    CORS_ALLOW_HEADERS: List[str] = field(
        default_factory=lambda: ["*"]
    )

    # Server Configuration
    HOST: str = os.getenv("BACKEND_HOST", os.getenv("HOST", "127.0.0.1"))
    PORT: int = int(os.getenv("BACKEND_PORT", os.getenv("PORT", "8000")))

    # ML Model Config (Preserved for future phases)
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    TOKENIZER_DIR: str = os.getenv("TOKENIZER_DIR", str(DEFAULT_TOKENIZER_DIR))
    DEVICE: str = os.getenv("MODEL_DEVICE", os.getenv("DEVICE", "cpu"))

    # iSign T5 Pretrained Pose-to-Text Model Config (F-012)
    ISIGN_MODEL_DIR: str = os.getenv("ISIGN_MODEL_DIR", str(DEFAULT_ISIGN_MODEL_DIR))
    ISIGN_CHECKPOINT_NAME: str = os.getenv("ISIGN_CHECKPOINT_NAME", "t5-base")
    ISIGN_TOKENIZER_DIR: str = os.getenv("ISIGN_TOKENIZER_DIR", str(DEFAULT_ISIGN_TOKENIZER_DIR))
    ISIGN_OFFLINE: bool = os.getenv("ISIGN_OFFLINE", "true").lower() in ("true", "1", "yes")

    # Sarvam AI Cloud Services (FE-08)
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "").strip()
    SARVAM_BASE_URL: str = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai").rstrip("/")
    SARVAM_TIMEOUT_SECONDS: float = float(os.getenv("SARVAM_TIMEOUT_SECONDS", "15.0"))


settings = Settings()
