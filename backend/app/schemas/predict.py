from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class LandmarkSequencePredictionRequest(BaseModel):
    landmarks: List[List[float]] = Field(
        ...,
        description="Sequence of 75 landmarks (300 coordinates) per frame: shape (T, 300) or (T, 75, 4)",
    )
    decoding: Literal["beam", "greedy"] = Field("beam", description="T5 text decoding strategy (beam or greedy)")


class PredictionResponse(BaseModel):
    success: bool
    predicted_text: str
    decoding: str
    inference_time_seconds: float
    video_name: str
    device: str
    error: Optional[str] = None
