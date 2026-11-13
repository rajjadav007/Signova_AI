from fastapi import APIRouter

try:
    from app.api.routes import (
        health,
        isolated_recognition,
        predict,
        sentence_recognition,
        speech,
        translation,
    )
except ModuleNotFoundError:
    from backend.app.api.routes import (
        health,
        isolated_recognition,
        predict,
        sentence_recognition,
        speech,
        translation,
    )

api_router = APIRouter()

# Register modular routes
api_router.include_router(health.router)
api_router.include_router(predict.router)
api_router.include_router(isolated_recognition.router)
api_router.include_router(sentence_recognition.router)
api_router.include_router(translation.router)
api_router.include_router(speech.router)

__all__ = ["api_router"]
