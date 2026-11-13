"""
API routes for Sarvam AI Text Translation (FE-08).
"""

import logging
from typing import Dict
from fastapi import APIRouter

try:
    from app.schemas.sarvam import (
        SUPPORTED_SARVAM_LANGUAGES,
        TranslationRequest,
        TranslationResponse,
    )
    from app.services.sarvam_service import sarvam_service
except ModuleNotFoundError:
    from backend.app.schemas.sarvam import (
        SUPPORTED_SARVAM_LANGUAGES,
        TranslationRequest,
        TranslationResponse,
    )
    from backend.app.services.sarvam_service import sarvam_service

logger = logging.getLogger("signova.routes.translation")

router = APIRouter(prefix="/translate", tags=["translation"])


@router.post("", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest) -> TranslationResponse:
    """
    Translates an edited ISL word sequence or message into a selected Indian language using Sarvam AI.
    """
    return await sarvam_service.translate(request)


@router.get("/languages", response_model=Dict[str, str])
def get_supported_languages() -> Dict[str, str]:
    """
    Returns the catalog of supported target language codes and human-readable names.
    """
    return SUPPORTED_SARVAM_LANGUAGES
