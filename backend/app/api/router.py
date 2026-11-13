from fastapi import APIRouter

from app.api.routes import health, isolated_recognition, predict, speech, translation

api_router = APIRouter()

# Register modular routes
api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(isolated_recognition.router)
api_router.include_router(translation.router)
api_router.include_router(speech.router)

__all__ = ["api_router"]
