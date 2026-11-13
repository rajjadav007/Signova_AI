import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("signova.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lightweight application startup and shutdown lifecycle."""
    logger.info("Starting %s (%s)...", settings.API_TITLE, settings.APP_ENV)
    yield
    logger.info("Shutting down %s...", settings.API_TITLE)
    try:
        from app.services.inference_service import inference_service
        inference_service.unload_model()
    except Exception:
        pass


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware (registered exactly once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Centralized Error Handlers
register_exception_handlers(app)

# Central API Router Integration (prefix applied exactly once)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["root"])
def root():
    """Root endpoint with service info and links."""
    return {
        "status": "online",
        "app": settings.API_TITLE,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }


@app.get("/health", tags=["health"])
def health_alias():
    """Root-level health check endpoint for proxy and monitor compatibility."""
    return {
        "status": "healthy",
        "application": settings.API_TITLE,
        "version": settings.API_VERSION,
        "environment": settings.APP_ENV,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
