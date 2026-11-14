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


class TwentyFiveClassCandidateItem(BaseModel):
    """Top candidate item for 25-class sentence recognition."""

    class_id: int = Field(..., description="Sentence class ID (0 to 24)")
    sentence: str = Field(..., description="Predicted English sentence")
    probability: float = Field(..., description="Softmax class probability [0.0..1.0]")


class TwentyFiveClassSentenceRecognitionResponse(BaseModel):
    """Inference response from the 25-class BiGRU Attention sequence model."""

    success: bool = Field(True, description="Whether inference completed successfully")
    is_recognized: bool = Field(True, description="Whether a valid prediction was obtained")
    is_confident: bool = Field(True, description="Whether confidence exceeds threshold (e.g. 0.25)")
    status: str = Field("SUCCESS", description="'SUCCESS', 'LOW_CONFIDENCE', or error status")
    predicted_sentence: Optional[str] = Field(None, description="Predicted sentence (always populated if valid inference)")
    sentence: Optional[str] = Field(None, description="Alias for predicted_sentence for client compatibility")
    predicted_class_id: Optional[int] = Field(None, description="Predicted class ID (0 to 24)")
    predicted_sentence_id: Optional[int] = Field(None, description="Alias for predicted_class_id for client compatibility")
    confidence: float = Field(0.0, description="Highest softmax class probability [0.0..1.0]")
    class_probabilities: Dict[str, float] = Field(default_factory=dict, description="Softmax probabilities for all 25 classes")
    top5_candidates: List[TwentyFiveClassCandidateItem] = Field(default_factory=list, description="Top-5 candidate sentences and their probabilities")
    top_predictions: Optional[List[Dict[str, Any]]] = Field(default=None, description="Candidate predictions for client compatibility")
    uncertainty_warning: Optional[str] = Field(None, description="Uncertainty warning when confidence is low")
    frame_count: int = Field(0, description="Number of video frames processed")
    processing_time_ms: float = Field(..., description="End-to-end processing latency in ms")
    model_inference_latency_ms: Optional[float] = Field(None, description="Neural network forward pass latency in ms")
    model_used: str = Field("TwentyFiveClassBiGRUAttentionModel", description="Model architecture used")
    model_identifier: str = Field("bigru_25class", description="Model identifier")
    reason: Optional[str] = Field(None, description="Failure or rejection reason if any")


class TwentyFiveClassItem(BaseModel):
    """25-class sentence class mapping item."""

    class_id: int = Field(..., description="Class ID (0 to 24)")
    sentence: str = Field(..., description="Sentence text")


class TwentyFiveClassesResponse(BaseModel):
    """Supported 25 continuous sentence classes."""

    total: int = Field(25, description="Total number of classes")
    classes: List[TwentyFiveClassItem] = Field(..., description="List of supported 25 sentence classes")





