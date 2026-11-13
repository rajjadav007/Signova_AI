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

__all__ = [
    "health",
    "isolated_recognition",
    "predict",
    "sentence_recognition",
    "speech",
    "translation",
]
