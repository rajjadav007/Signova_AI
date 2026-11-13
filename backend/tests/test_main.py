import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.main import app, root


def test_root_endpoint():
    """Verify root endpoint returns service status."""
    data = root()
    assert data["status"] == "online"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"


def test_app_instance():
    """Verify that app is an instance of FastAPI."""
    assert isinstance(app, FastAPI)


def test_app_metadata():
    """Verify application metadata (title, version, description)."""
    assert app.title == "Signova AI API"
    assert app.version == "0.1.0"
    assert app.description
    assert "Signova AI" in app.description


def test_openapi_schema():
    """Verify OpenAPI schema generation."""
    schema = app.openapi()
    assert schema is not None
    assert schema["info"]["title"] == "Signova AI API"
    assert schema["info"]["version"] == "0.1.0"
    assert "paths" in schema


def test_cors_middleware():
    """Verify CORS middleware is properly configured for frontend origin."""
    cors_middleware = None
    for middleware in app.user_middleware:
        if middleware.cls == CORSMiddleware:
            cors_middleware = middleware
            break

    assert cors_middleware is not None
    allowed_origins = cors_middleware.kwargs.get("allow_origins", [])
    assert "http://localhost:5173" in allowed_origins
    assert "*" not in allowed_origins
    assert cors_middleware.kwargs.get("allow_credentials") is True


def test_router_integration():
    """Verify central API router is configured."""
    assert settings.API_V1_STR == "/api/v1"
    # Ensure api prefix is configured
    assert settings.API_V1_STR.startswith("/")


def test_no_ml_modules_imported():
    """Verify that importing app does not load heavy ML/AI libraries."""
    forbidden_modules = ["torch.cuda", "mediapipe", "transformers"]
    loaded = [m for m in forbidden_modules if m in sys.modules]
    assert len(loaded) == 0, f"Forbidden ML modules found loaded: {loaded}"
