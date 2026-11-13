"""
Pydantic schemas for the Isolated ISL Recognition endpoints.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    label: str = Field(..., description="ISL gloss label (e.g. 'hello', 'food')")
    display_name: Optional[str] = Field(None, description="User-facing display name")
    confidence: float = Field(..., description="Softmax confidence score [0.0, 1.0]")
    category: Optional[str] = Field("general", description="Category: 'alphabet', 'number', 'daily', 'public'")
    sign: Optional[str] = Field(None, description="Gloss label alias")


class LandmarkSequenceRequest(BaseModel):
    landmarks: List[Any] = Field(
        default_factory=list,
        description="Temporal sequence of MediaPipe keypoints: shape (T, 300) or (T, 75, 4)",
    )
    category: Optional[str] = Field(
        None,
        description="Optional filter category: 'alphabet', 'number', 'daily', 'public'",
    )
    top_k: int = Field(3, ge=1, le=10, description="Number of top candidates to return")


class IsolatedRecognitionResponse(BaseModel):
    success: bool = Field(True, description="Whether inference completed successfully")
    predicted_label: str = Field(..., description="Confirmed ISL gloss label, or 'uncertain'/'unknown'")
    predicted_sign: str = Field(..., description="Highest-scoring raw predicted ISL sign label")
    display_name: str = Field(..., description="Human-readable sign name")
    category: str = Field("general", description="Class category ('alphabet', 'number', 'daily', 'public')")
    confidence: float = Field(..., description="Prediction confidence [0.0, 1.0]")
    status: str = Field(
        ...,
        description="Recognition status: 'recognized', 'uncertain'",
    )
    is_valid: bool = Field(..., description="True if status == 'recognized'")
    processing_time_ms: float = Field(..., description="End-to-end processing time in milliseconds")
    model_used: str = Field(..., description="Active model architecture name")
    rejection_threshold: float = Field(0.80, description="Confidence threshold used for recognition")
    top_3_predictions: List[TopPrediction] = Field(default_factory=list, description="Top 3 candidate predictions")
    top_predictions: List[TopPrediction] = Field(default_factory=list, description="Top candidate predictions alias")
    reason: Optional[str] = Field(None, description="Detailed explanation if prediction rejected")
