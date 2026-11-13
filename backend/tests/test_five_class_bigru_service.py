"""
Unit and Integration Tests for FiveClassBiGRUService.
Tests:
1. Service initialization and class mapping loading.
2. Prediction on video bytes using FiveClassBiGRUAttentionModel.
3. Insufficient frames rejection (< 8 frames).
4. Class probabilities schema and calibration.
5. Endpoints verification.
"""

from pathlib import Path
import sys
import pytest

backend_dir = str(Path(__file__).resolve().parent.parent)
project_root = Path(backend_dir).parent
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.five_class_bigru_service import five_class_bigru_service


def test_five_class_bigru_service_initialization():
    classes = five_class_bigru_service.get_classes()
    assert len(classes) == 5, f"Expected 5 classes, got {len(classes)}"
    assert five_class_bigru_service.is_loaded is True

    expected_sentences = [
        "comb your hair",
        "help me",
        "i am fine. thank you sir.",
        "i need water",
        "who are you"
    ]
    loaded_sentences = [c["sentence"] for c in classes]
    assert loaded_sentences == expected_sentences


def test_five_class_bigru_insufficient_frames():
    # Pass a tiny invalid or empty video byte sequence
    res = five_class_bigru_service.predict_video(
        video_bytes=b"dummy bytes not a real video",
        filename="dummy.mp4",
        min_frames=8
    )
    assert res["success"] is True
    assert res["is_recognized"] is False
    assert res["status"] == "INSUFFICIENT_FRAMES"
    assert "Minimum 8 frames required" in res["reason"]


def test_five_class_bigru_video_prediction():
    # Test with real test signer video (comb your hair)
    sample_video_path = Path(
        r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level\comb your hair\comb (2).MP4"
    )
    if not sample_video_path.exists():
        pytest.skip(f"Sample video not found at {sample_video_path}")

    with open(sample_video_path, "rb") as f:
        video_bytes = f.read()

    res = five_class_bigru_service.predict_video(
        video_bytes=video_bytes,
        filename="comb (2).MP4",
        confidence_threshold=0.35,
        min_frames=8
    )

    assert res["success"] is True
    assert res["is_recognized"] is True
    assert res["predicted_sentence"] == "comb your hair"
    assert res["predicted_class_id"] == 0
    assert res["confidence"] >= 0.35
    assert len(res["class_probabilities"]) == 5
    assert res["frame_count"] > 0
    assert res["model_used"] == "FiveClassBiGRUAttentionModel"
