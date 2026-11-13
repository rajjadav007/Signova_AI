import json
import sys
from pathlib import Path
from starlette.requests import Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.exceptions import (
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
)
from app.main import app
from app.schemas.error import ErrorResponse


def _dummy_request(method="GET", path="/api/v1/test"):
    scope = {"type": "http", "method": method, "path": path, "headers": []}
    return Request(scope)


def test_http_exception_404():
    """Verify 404 Not Found produces predictable error structure."""
    req = _dummy_request()
    exc = StarletteHTTPException(status_code=404)
    response = http_exception_handler(req, exc)

    assert response.status_code == 404
    data = json.loads(response.body.decode("utf-8"))
    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["message"] == "The requested resource was not found."

    # Validate against Pydantic schema
    ErrorResponse.model_validate(data)


def test_http_exception_custom_status_and_message():
    """Verify other HTTP status codes (e.g. 400, 403, 405) are preserved."""
    req = _dummy_request(method="POST")

    for status_code, expected_code in [(400, "BAD_REQUEST"), (403, "FORBIDDEN"), (405, "METHOD_NOT_ALLOWED")]:
        exc = StarletteHTTPException(status_code=status_code, detail=f"Custom error {status_code}")
        response = http_exception_handler(req, exc)

        assert response.status_code == status_code
        data = json.loads(response.body.decode("utf-8"))
        assert data["success"] is False
        assert data["error"]["code"] == expected_code
        assert data["error"]["message"] == f"Custom error {status_code}"
        ErrorResponse.model_validate(data)


def test_validation_exception_422():
    """Verify RequestValidationError produces 422 with validation details."""
    req = _dummy_request()
    raw_errors = [
        {"loc": ("body", "video"), "msg": "Field required", "type": "missing"},
        {"loc": ("query", "limit"), "msg": "Input should be a valid integer", "type": "int_type"},
    ]
    exc = RequestValidationError(raw_errors)
    response = validation_exception_handler(req, exc)

    assert response.status_code == 422
    data = json.loads(response.body.decode("utf-8"))
    assert data["success"] is False
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert data["error"]["message"] == "Request validation failed."
    assert isinstance(data["error"]["details"], list)
    assert len(data["error"]["details"]) == 2
    assert data["error"]["details"][0]["field"] == "video"
    ErrorResponse.model_validate(data)


def test_unhandled_exception_500_safe_response():
    """Verify unexpected exceptions return 500 and do not expose stack traces or secrets."""
    req = _dummy_request()
    sensitive_info = "postgresql://user:super_secret_pw@localhost:5432/db"
    exc = RuntimeError(f"Database connection failure: {sensitive_info}")

    response = unhandled_exception_handler(req, exc)

    assert response.status_code == 500
    data = json.loads(response.body.decode("utf-8"))
    assert data["success"] is False
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["message"] == "An unexpected internal server error occurred."
    assert sensitive_info not in response.body.decode("utf-8")
    assert "RuntimeError" not in response.body.decode("utf-8")
    ErrorResponse.model_validate(data)


def test_exception_handlers_registered_on_app():
    """Verify handlers are properly attached to the FastAPI application instance."""
    handlers = app.exception_handlers
    assert StarletteHTTPException in handlers
    assert RequestValidationError in handlers
    assert Exception in handlers
