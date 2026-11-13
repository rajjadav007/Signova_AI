from fastapi import APIRouter

from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check Signova AI backend availability and runtime metadata.",
)
def get_health() -> HealthResponse:
    """Return backend liveness and basic application runtime status."""
    return HealthResponse(
        status="healthy",
        application=settings.API_TITLE,
        version=settings.API_VERSION,
        environment=settings.APP_ENV,
    )
