import sys
from pathlib import Path
from fastapi import APIRouter

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.api.router import api_router as central_router
from app.api import api_router as package_router
from app.main import app


def test_central_router_instance():
    """Verify central router is an APIRouter instance."""
    assert isinstance(central_router, APIRouter)
    assert central_router is package_router


def test_router_contains_health_route():
    """Verify health route is registered within central router."""
    paths = []
    for r in central_router.routes:
        if hasattr(r, "path"):
            paths.append(r.path)
        elif hasattr(r, "original_router"):
            paths.extend(
                [sub_r.path for sub_r in r.original_router.routes if hasattr(sub_r, "path")]
            )
    assert "/health" in paths


def test_app_openapi_paths_and_no_duplicate_prefix():
    """Verify OpenAPI schema contains /api/v1/health and no duplicated prefix."""
    schema = app.openapi()
    paths = list(schema["paths"].keys())
    assert "/api/v1/health" in paths
    for p in paths:
        assert not p.startswith("/api/v1/api/v1"), f"Duplicate prefix found: {p}"


def test_router_independence():
    """Verify router imports without loading ML modules."""
    forbidden = ["torch.cuda", "mediapipe", "transformers"]
    loaded = [m for m in forbidden if m in sys.modules]
    assert len(loaded) == 0, f"Unexpected ML modules loaded: {loaded}"
