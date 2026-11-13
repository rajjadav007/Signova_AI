import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.api.routes.health import get_health
from app.core.config import settings
from app.main import app
from app.schemas.health import HealthResponse


def test_health_response_success():
    """Test 1: Verify successful health check execution and structure."""
    response = get_health()
    assert isinstance(response, HealthResponse)
    assert response.status == "healthy"
    data = response.model_dump()
    assert set(data.keys()) == {"status", "application", "version", "environment"}


def test_health_application_metadata():
    """Test 2: Verify configured application metadata in health response."""
    response = get_health()
    assert response.application == settings.API_TITLE
    assert response.version == settings.API_VERSION
    assert response.environment == settings.APP_ENV


def test_health_response_consistency():
    """Test 3: Verify required fields and types are consistent."""
    response = get_health()
    assert isinstance(response.status, str)
    assert isinstance(response.application, str)
    assert isinstance(response.version, str)
    assert isinstance(response.environment, str)
    assert len(response.application) > 0
    assert len(response.version) > 0


def test_health_independence_from_ai_and_db():
    """Test 4: Verify health check runs without loading ML libraries or requiring database."""
    # Forbidden heavy dependencies
    forbidden = ["torch.cuda", "mediapipe", "transformers", "cv2"]
    loaded = [m for m in forbidden if m in sys.modules]
    assert len(loaded) == 0, f"ML libraries unexpectedly loaded: {loaded}"

    # Verify no database connection dependency in health endpoint
    response = get_health()
    assert response.status == "healthy"


def test_health_openapi_route_registration():
    """Verify endpoint is registered in OpenAPI schema at /api/v1/health and /health."""
    schema = app.openapi()
    assert "/api/v1/health" in schema["paths"]
    assert "/health" in schema["paths"]
    endpoint = schema["paths"]["/api/v1/health"]
    assert "get" in endpoint
    assert endpoint["get"]["tags"] == ["health"]
    assert "HealthResponse" in schema["components"]["schemas"]
