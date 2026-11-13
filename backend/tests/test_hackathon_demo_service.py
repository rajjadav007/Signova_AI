"""
Unit tests for HackathonDemoService and demo endpoints.
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

from app.services.hackathon_demo_service import hackathon_demo_service


def test_hackathon_demo_service_initialization():
    classes = hackathon_demo_service.get_classes()
    assert len(classes) == 5, f"Expected 5 classes, got {len(classes)}"
    assert hackathon_demo_service.is_loaded is True
    assert len(hackathon_demo_service.ref_seqs) == 10
    assert len(hackathon_demo_service.ref_labels) == 10

    expected_sentences = {
        "comb your hair",
        "help me",
        "i am fine. thank you sir",
        "i need water",
        "who are you",
    }
    loaded_sentences = {c["sentence"] for c in classes}
    assert loaded_sentences == expected_sentences


def test_hackathon_demo_video_prediction():
    # Test with a real reference video bytes
    sample_video_path = Path(
        r"D:\Datasets\ISL-CSLTR\isl-csltr-indian-sign-language-dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level\comb your hair\comb (4).MP4"
    )
    if not sample_video_path.exists():
        pytest.skip(f"Sample video not found at {sample_video_path}")

    with open(sample_video_path, "rb") as f:
        video_bytes = f.read()

    res = hackathon_demo_service.predict_video(
        video_bytes=video_bytes,
        filename="comb (4).MP4",
        threshold=0.40,
        min_margin=0.015,
    )

    assert res["success"] is True
    assert res["is_recognized"] is True
    assert res["predicted_sentence"] == "comb your hair"
    assert res["predicted_class_id"] == 0
    assert res["original_sentence_id"] == 5
    assert len(res["rankings"]) == 5
    assert res["top_distance"] < 0.35
    assert res["top_similarity"] > 0.65
    assert res["matching_latency_ms"] > 0
    assert res["processing_time_ms"] > 0


if __name__ == "__main__":
    test_hackathon_demo_service_initialization()
    print("test_hackathon_demo_service_initialization PASSED!")
    test_hackathon_demo_video_prediction()
    print("test_hackathon_demo_video_prediction PASSED!")
