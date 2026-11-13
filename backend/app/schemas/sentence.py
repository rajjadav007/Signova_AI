"""
Pydantic schemas for the Continuous / Sentence-Level ISL Recognition endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SentenceLandmarkRequest(BaseModel):
    """
    Request model for sentence recognition from raw landmark sequence arrays.
    Supports:
    - (T, 75, 4) nested coordinates (x, y, z, visibility)
    - (T, 300) flattened spatial frames
    - (T, 678) pre-extracted temporal features
    """

    landmarks: List[Any] = Field(
        ...,
        description="Temporal sequence of MediaPipe keypoints: shape (T, 75, 4), (T, 300), or (T, 678)",
    )
    timestamps_ms: Optional[List[int]] = Field(
        None,
        description="Optional list of integer timestamps per frame in milliseconds",
    )
    fps: float = Field(
        25.0,
        gt=0.0,
        le=120.0,
        description="Video frame rate in frames per second",
    )
    top_k: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of top candidates to return",
    )


class TopSentencePrediction(BaseModel):
    """Candidate sentence prediction item."""

    sentence_id: int = Field(..., description="Sentence class ID (0 to 100)")
    sentence: str = Field(..., description="Predicted English sentence")
    gloss: Optional[str] = Field(None, description="ISL gloss representation")
    confidence: float = Field(..., description="Softmax confidence score [0.0, 1.0]")


class SentenceRecognitionResponse(BaseModel):
    """Response model for continuous sentence recognition."""

    success: bool = Field(True, description="Whether inference completed successfully")
    predicted_sentence_id: int = Field(..., description="Predicted sentence ID (0 to 100)")
    sentence: str = Field(..., description="Predicted English sentence")
    gloss: Optional[str] = Field(None, description="ISL gloss translation")
    confidence: float = Field(..., description="Prediction confidence score [0.0, 1.0]")
    top_predictions: List[TopSentencePrediction] = Field(
        default_factory=list,
        description="Top-k sentence predictions ordered by confidence",
    )
    processing_time_ms: float = Field(..., description="End-to-end processing time in milliseconds")
    model_used: str = Field(..., description="Active model architecture name")
    status: str = Field(
        "success",
        description="Recognition status: 'success', 'uncertain', 'failed'",
    )
    reason: Optional[str] = Field(None, description="Explanation if prediction is uncertain or failed")


class SentenceVocabularyItem(BaseModel):
    """Supported sentence class vocabulary item."""

    sentence_id: int = Field(..., description="Sentence ID (0 to 100)")
    sentence: str = Field(..., description="English sentence")
    gloss: Optional[str] = Field(None, description="ISL gloss representation")
    tokens: Optional[List[str]] = Field(default=None, description="Token sequence")


class SentenceVocabularyResponse(BaseModel):
    """Vocabulary response containing all supported sentence classes."""

    total: int = Field(..., description="Total number of supported sentence classes")
    sentences: List[SentenceVocabularyItem] = Field(..., description="List of sentence classes")


class DemoClassItem(BaseModel):
    """5-Class Hackathon demo class item."""

    demo_class_id: int = Field(..., description="Demo class index (0 to 4)")
    original_sentence_id: int = Field(..., description="Original ISL-CSLTR sentence ID")
    sentence: str = Field(..., description="English sentence")
    isl_gloss: str = Field(..., description="ISL gloss string")


class DemoClassesResponse(BaseModel):
    """List of supported classes in 5-class demo mode."""

    total: int = Field(..., description="Total number of demo classes")
    classes: List[DemoClassItem] = Field(..., description="Supported demo classes")


class DemoRankingItem(BaseModel):
    """Ranked sentence candidate from DTW sequence alignment."""

    class_id: int = Field(..., description="Demo class ID (0 to 4)")
    original_sentence_id: int = Field(..., description="Original ISL-CSLTR sentence ID")
    sentence: str = Field(..., description="English sentence")
    isl_gloss: str = Field(..., description="ISL gloss representation")
    dtw_distance: float = Field(..., description="Length-normalized DTW cosine distance")
    similarity: float = Field(..., description="Alignment similarity score [0.0..1.0]")


class DemoSentenceRecognitionResponse(BaseModel):
    """Inference response from the 5-class DTW-KNN demo model."""

    success: bool = Field(True, description="Whether inference completed successfully")
    is_recognized: bool = Field(..., description="False if input was rejected by threshold")
    status: str = Field(..., description="'Recognized' or 'Sentence not recognized.'")
    predicted_sentence: Optional[str] = Field(None, description="Predicted sentence or null if rejected")
    predicted_class_id: Optional[int] = Field(None, description="Predicted demo class ID or null if rejected")
    original_sentence_id: Optional[int] = Field(None, description="Original ISL-CSLTR sentence ID")
    top_similarity: float = Field(..., description="DTW alignment similarity of top candidate")
    top_distance: float = Field(..., description="DTW distance of top candidate")
    runner_up_distance: Optional[float] = Field(None, description="DTW distance of 2nd place class")
    margin_to_second: float = Field(..., description="Distance margin between rank 1 and rank 2")
    matching_latency_ms: float = Field(..., description="DTW sequence matching latency in ms")
    processing_time_ms: float = Field(..., description="End-to-end video processing latency in ms")
    rankings: List[DemoRankingItem] = Field(default_factory=list, description="Ranked list of all 5 classes")
    model_used: str = Field("DTWKNNClassifier-5Class", description="Model identifier")
    note: str = Field(
        "Similarity score is uncalibrated DTW alignment similarity [0.0..1.0], not a class probability.",
        description="Important calibration disclosure",
    )


class FiveClassSentenceRecognitionResponse(BaseModel):
    """Inference response from the 5-class BiGRU Attention sequence model or DTW-KNN."""

    success: bool = Field(True, description="Whether inference completed successfully")
    is_recognized: bool = Field(..., description="False if rejected or below threshold")
    status: str = Field(..., description="'SUCCESS', 'Recognized', 'Sentence not recognized.', etc.")
    predicted_sentence: Optional[str] = Field(None, description="Predicted sentence")
    predicted_class_id: Optional[int] = Field(None, description="Predicted class ID (0 to 4)")
    original_sentence_id: Optional[int] = Field(None, description="Original ISL-CSLTR sentence ID")
    confidence: float = Field(0.0, description="Highest softmax class probability or alignment similarity")
    class_probabilities: Dict[str, float] = Field(default_factory=dict, description="Softmax probabilities for each of the 5 classes")
    frame_count: int = Field(0, description="Number of video frames processed")
    processing_time_ms: float = Field(..., description="End-to-end processing latency in ms")
    model_inference_latency_ms: Optional[float] = Field(None, description="Neural network forward pass latency in ms")
    model_used: str = Field("FiveClassBiGRUAttentionModel", description="Model architecture used")
    reason: Optional[str] = Field(None, description="Rejection reason if not recognized")
    top_distance: Optional[float] = Field(None, description="DTW distance of top candidate")
    runner_up_distance: Optional[float] = Field(None, description="DTW distance of 2nd place candidate")
    margin_to_second: Optional[float] = Field(None, description="Distance margin between rank 1 and rank 2")
    matching_latency_ms: Optional[float] = Field(None, description="DTW sequence matching latency in ms")
    rankings: Optional[List[DemoRankingItem]] = Field(default=None, description="Ranked list of all 5 classes")




