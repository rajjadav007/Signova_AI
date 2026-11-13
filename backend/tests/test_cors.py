import sys
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings, _parse_cors_origins
from app.main import app


def test_cors_middleware_single_registration():
    """Verify CORSMiddleware is registered exactly once on the application."""
    cors_middlewares = [m for m in app.user_middleware if m.cls == CORSMiddleware]
    assert len(cors_middlewares) == 1, f"Expected exactly 1 CORS middleware, found {len(cors_middlewares)}"


def test_cors_allowed_origins():
    """Verify both standard frontend origins are allowed."""
    cors = [m for m in app.user_middleware if m.cls == CORSMiddleware][0]
    allowed_origins = cors.kwargs.get("allow_origins", [])
    assert "http://localhost:5173" in allowed_origins
    assert "http://127.0.0.1:5173" in allowed_origins


def test_cors_credentials_and_no_wildcard():
    """Verify allow_credentials is True and wildcard origin '*' is strictly prohibited."""
    cors = [m for m in app.user_middleware if m.cls == CORSMiddleware][0]
    allowed_origins = cors.kwargs.get("allow_origins", [])
    assert cors.kwargs.get("allow_credentials") is True
    assert "*" not in allowed_origins


def test_cors_methods_and_headers():
    """Verify standard methods and required headers are supported."""
    cors = [m for m in app.user_middleware if m.cls == CORSMiddleware][0]
    methods = cors.kwargs.get("allow_methods", [])
    headers = cors.kwargs.get("allow_headers", [])
    # Either "*" or explicit methods/headers including Content-Type & Authorization
    assert "*" in methods or all(m in methods for m in ["GET", "POST", "OPTIONS"])
    assert "*" in headers or all(h in headers for h in ["Content-Type", "Authorization"])


def test_cors_parser_strips_whitespace_and_wildcards(monkeypatch):
    """Verify origin parser handles whitespace, empty entries, and excludes wildcard."""
    monkeypatch.setenv("CORS_ORIGINS", " http://localhost:5173 , , http://example.com , * ")
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:5173")
    origins = _parse_cors_origins()
    assert "http://localhost:5173" in origins
    assert "http://127.0.0.1:5173" in origins
    assert "http://example.com" in origins
    assert "" not in origins
    assert "*" not in origins
