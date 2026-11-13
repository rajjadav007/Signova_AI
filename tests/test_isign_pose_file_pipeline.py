"""
Unit and Integration Tests for F-016: Official iSign .pose File-Based Video Inference.
Validates:
1. Video-to-pose landmark serialization conforming to official iSign topology and ordering.
2. Official pose-format binary parsing and roundtrip fidelity.
3. Official pose_utils loader compatibility.
4. Missing-landmark representation and preservation.
5. End-to-end PoseTranslator inference with .pose files.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest
import torch

from ai.inference.pose_translator import PoseTranslator, translate_pose_file
from ai.pose_extraction.pose_generator import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    POSE_CONNECTIONS,
    POSE_LANDMARK_NAMES,
    build_isign_pose_header,
    landmarks_to_pose,
    save_pose_file,
    validate_pose_file,
    video_to_pose_file,
)
from ai.preprocessing.pose_preprocessor import load_pose_file, preprocess_pose

# Add official repo to sys.path for direct official loader comparison
OFFICIAL_INFERENCE_DIR = Path("D:/iSign-t5-pose-to-text/inference")
if str(OFFICIAL_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_INFERENCE_DIR))


def test_isign_pose_header_specification():
    """Verify PoseHeader adheres to official iSign components and topology."""
    header = build_isign_pose_header(width=1920, height=1080, depth=1000)

    assert header.dimensions.width == 1920
    assert header.dimensions.height == 1080
    assert header.dimensions.depth == 1000
    assert len(header.components) == 3

    c_pose, c_lh, c_rh = header.components
    assert c_pose.name == "POSE_LANDMARKS"
    assert len(c_pose.points) == 33
    assert len(c_pose.limbs) == 35
    assert c_pose.format == "XYZC"

    assert c_lh.name == "LEFT_HAND_LANDMARKS"
    assert len(c_lh.points) == 21
    assert len(c_lh.limbs) == 21
    assert c_lh.format == "XYZC"

    assert c_rh.name == "RIGHT_HAND_LANDMARKS"
    assert len(c_rh.points) == 21
    assert len(c_rh.limbs) == 21
    assert c_rh.format == "XYZC"


def test_landmarks_to_pose_roundtrip_numerical_identity():
    """Verify exact numerical preservation (0.0 error) across save/read roundtrip."""
    from pose_format import Pose

    np.random.seed(42)
    frames = 25
    original = np.random.rand(frames, 300).astype(np.float32)

    pose = landmarks_to_pose(original, width=1280, height=720, fps=29.97)

    with tempfile.NamedTemporaryFile(suffix=".pose", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        saved_path = save_pose_file(pose, tmp_path)
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

        # Validate with comprehensive validator
        report = validate_pose_file(saved_path, expected_landmarks=original)
        assert report["is_valid"] is True
        assert report["topology_valid"] is True
        assert report["points_valid"] is True
        assert report["shape_valid"] is True
        assert report["confidence_valid"] is True
        assert report["exact_roundtrip_match"] is True
        assert report["max_roundtrip_difference"] == 0.0

        # Load with Signova load_pose_file
        loaded_signova = load_pose_file(saved_path)
        np.testing.assert_array_equal(original, loaded_signova)

        # Load with Official iSign pose_utils.load_pose_file
        import pose_utils as official_pose_utils
        loaded_official = official_pose_utils.load_pose_file(str(saved_path))
        np.testing.assert_array_equal(original, loaded_official)

    finally:
        tmp_path.unlink(missing_ok=True)


def test_missing_landmark_representation():
    """Verify missing landmarks (confidence=0.0, coords=0.0) are accurately preserved."""
    frames = 15
    original = np.random.rand(frames, 300).astype(np.float32)

    # Frame 0: Missing left hand
    original[0, 33 * 4 : 54 * 4] = 0.0
    # Frame 1: Missing right hand
    original[1, 54 * 4 : 75 * 4] = 0.0
    # Frame 2: Missing both hands
    original[2, 33 * 4 : 75 * 4] = 0.0
    # Frame 3: Missing body landmarks
    original[3, 0 : 33 * 4] = 0.0

    pose = landmarks_to_pose(original, width=640, height=480, fps=30.0)

    with tempfile.NamedTemporaryFile(suffix=".pose", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_pose_file(pose, tmp_path)
        report = validate_pose_file(tmp_path, expected_landmarks=original)

        assert report["is_valid"] is True
        assert report["exact_roundtrip_match"] is True
        assert report["max_roundtrip_difference"] == 0.0
        assert report["zero_confidence_count"] > 0

        # Check official preprocess_pose works seamlessly with missing landmarks
        import pose_utils as official_pose_utils
        official_preprocessed, plen = official_pose_utils.preprocess_pose(
            str(tmp_path), max_length=500, pose_dim=600, use_motion=True
        )
        assert official_preprocessed.shape == (500, 600)
        assert plen == frames
        assert not np.isnan(official_preprocessed).any()

    finally:
        tmp_path.unlink(missing_ok=True)


def test_official_and_generated_pose_preprocessing_parity():
    """Compare tensor shapes, pose length, and motion feature values."""
    frames = 30
    synthetic = np.random.rand(frames, 300).astype(np.float32)
    pose = landmarks_to_pose(synthetic, width=1920, height=1080, fps=30.0)

    with tempfile.NamedTemporaryFile(suffix=".pose", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_pose_file(pose, tmp_path)

        # Official preprocessing
        import pose_utils as official_pose_utils
        official_feat, official_len = official_pose_utils.preprocess_pose(
            str(tmp_path), max_length=500, pose_dim=600, use_motion=True
        )

        # Signova preprocessing from the .pose file
        signova_prep = preprocess_pose(
            tmp_path, max_length=500, use_motion=True, device="cpu"
        )

        assert official_len == signova_prep.actual_length == frames
        assert official_feat.shape == (500, 600)
        assert tuple(signova_prep.pose_tensor.shape) == (1, 500, 600)

        # Compare features
        signova_feat = signova_prep.features
        np.testing.assert_allclose(official_feat, signova_feat, atol=1e-5)

    finally:
        tmp_path.unlink(missing_ok=True)


def test_pose_translator_inference_on_pose_file():
    """Verify PoseTranslator.translate_pose_file and predict on .pose path."""
    frames = 10
    synthetic = np.random.rand(frames, 300).astype(np.float32)
    pose = landmarks_to_pose(synthetic, width=1280, height=720, fps=25.0)

    with tempfile.NamedTemporaryFile(suffix=".pose", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_pose_file(pose, tmp_path)

        translator = PoseTranslator(auto_load=True, device="cpu")
        text = translator.translate_pose_file(tmp_path, max_length=32, num_beams=2)
        assert isinstance(text, str)
        assert len(text) > 0

        # Also test predict_with_metadata on .pose path
        res = translator.predict_with_metadata(tmp_path, max_length=32, num_beams=2)
        assert res.text == text
        assert res.input_frames == frames
        assert res.input_shape == (1, 500, 300)
        assert res.latency_ms > 0

    finally:
        tmp_path.unlink(missing_ok=True)
