import logging
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("signova.exceptions")

HTTP_STATUS_CODES: Dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    402: "PAYMENT_REQUIRED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    408: "REQUEST_TIMEOUT",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_SERVER_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}


def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle standard Starlette / FastAPI HTTPExceptions with predictable format."""
    code = HTTP_STATUS_CODES.get(exc.status_code, f"HTTP_{exc.status_code}")
    message = str(exc.detail) if exc.detail else "An HTTP error occurred."

    # Standardize default 404 message to match specification
    if exc.status_code == 404 and message.lower() in ("not found", ""):
        message = "The requested resource was not found."

    error_body: Dict[str, Any] = {
        "success": False,
        "detail": message,
        "error": {
            "code": code,
            "message": message,
        },
    }
    return JSONResponse(status_code=exc.status_code, content=error_body)


def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle RequestValidationError with 422 status and structured error details."""
    cleaned_errors = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if x != "body"]
        cleaned_errors.append({
            "field": ".".join(loc) if loc else "body",
            "message": err.get("msg", "Invalid value"),
            "type": err.get("type", "value_error"),
        })

    error_body: Dict[str, Any] = {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed.",
            "details": cleaned_errors,
        },
    }
    return JSONResponse(status_code=422, content=error_body)


def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions returning safe 500 without leaking internals."""
    req_method = request.method if request else "UNKNOWN"
    req_path = request.url.path if request and hasattr(request, "url") else "UNKNOWN"
    logger.error(
        "Unhandled exception during %s %s: %s",
        req_method,
        req_path,
        exc,
        exc_info=True,
    )
    error_body: Dict[str, Any] = {
        "success": False,
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected internal server error occurred.",
        },
    }
    return JSONResponse(status_code=500, content=error_body)


def register_exception_handlers(app: FastAPI) -> None:
    """Register all centralized exception handlers with the FastAPI application."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
