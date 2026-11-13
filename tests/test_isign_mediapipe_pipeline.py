"""
Unit & Integration Tests for F-015.1 MediaPipe-to-iSign Pipeline Fix.
Tests:
- 75-landmark ordering (Pose 0..32, Left Hand 33..53, Right Hand 54..74).
- 300-dimensional spatial features.
- 600-dimensional motion features.
- Missing-hand handling (zeros in all 4 channels).
- Zero and NaN inputs.
- Incorrect MediaPipe outputs handling.
- Sequence padding and truncation.
- Full real-video extraction and inference integration.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from ai.config.settings import (
    DEFAULT_MEDIAPIPE_TASK_PATH,
    ISIGN_CHANNELS,
    ISIGN_HAND_COUNT,
    ISIGN_MOTION_DIM,
    ISIGN_POSE_COUNT,
    ISIGN_SPATIAL_DIM,
    ISIGN_TOTAL_LANDMARKS,
)
from ai.inference.model_loader import load_model, unload_model
from ai.inference.pose_translator import PoseTranslator
from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from ai.preprocessing.pose_preprocessor import (
    add_motion_features,
    assemble_frame_landmarks,
    normalize_pose,
    pad_or_truncate_pose,
    preprocess_pose,
    to_spatial_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_VIDEO_PATH = PROJECT_ROOT / "datasets" / "videos" / "10 granfather sick  (2).mov"


@pytest.fixture(scope="module")
def extractor():
    """Create extractor instance once for module tests."""
    ext = MediaPipeLandmarkExtractor()
    yield ext
    ext.close()


@pytest.fixture(scope="module")
def translator():
    """Load model once for inference tests."""
    load_model(device="cpu")
    t = PoseTranslator(auto_load=True, device="cpu")
    yield t
    unload_model()


def test_1_isign_75_landmark_ordering(extractor):
    """Verify exact 75 landmark order: 33 Pose -> 21 Left Hand -> 21 Right Hand."""
    # Dummy synthetic RGB frame
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    kps, quality = extractor.extract_isign_frame(frame)

    assert kps.shape == (ISIGN_TOTAL_LANDMARKS, ISIGN_CHANNELS)
    assert kps.shape == (75, 4)
    assert kps.dtype == np.float32

    # Verify assemble_frame_landmarks follows identical indexing
    pose_part = np.full((33, 4), 1.0, dtype=np.float32)
    lh_part = np.full((21, 4), 2.0, dtype=np.float32)
    rh_part = np.full((21, 4), 3.0, dtype=np.float32)

    spatial_vec = assemble_frame_landmarks(pose=pose_part, left_hand=lh_part, right_hand=rh_part)
    assert spatial_vec.shape == (300,)
    np.testing.assert_array_equal(spatial_vec[0:132], 1.0)       # 33 * 4 = 132
    np.testing.assert_array_equal(spatial_vec[132:216], 2.0)     # 21 * 4 = 84
    np.testing.assert_array_equal(spatial_vec[216:300], 3.0)     # 21 * 4 = 84


def test_2_spatial_300_and_motion_600_dimensions():
    """Verify 300 spatial features expand to 600 spatial+motion features."""
    frames = 40
    raw_spatial = np.random.randn(frames, ISIGN_SPATIAL_DIM).astype(np.float32)

    # Spatial check
    spatial_mat = to_spatial_matrix(raw_spatial)
    assert spatial_mat.shape == (frames, 300)

    # Motion features check
    normalized = normalize_pose(spatial_mat, method="zscore")
    with_motion = add_motion_features(normalized)
    assert with_motion.shape == (frames, 600)

    # First frame velocity must be zero
    np.testing.assert_array_equal(with_motion[0, 300:], 0.0)

    # Preprocessed tensor check (simple model without motion)
    prep = preprocess_pose(raw_spatial, max_length=500, use_motion=False, device="cpu")
    assert prep.pose_tensor.shape == (1, 500, 300)
    assert prep.actual_length == frames
    assert prep.pose_length.item() == frames


def test_3_missing_hand_handling_zeros(extractor):
    """Verify missing hands produce exact zeros across all 4 channels [x, y, z, conf]."""
    # Pure blank black frame where no hands exist
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    kps, quality = extractor.extract_isign_frame(blank_frame)

    assert kps.shape == (75, 4)
    assert not quality["pose"]
    assert not quality["left_hand"]
    assert not quality["right_hand"]
    assert quality["missing_landmarks_count"] == 75
    assert quality["missing_percentage"] == 100.0

    # Left hand slice (33..53) and Right hand slice (54..74) must be exactly 0.0
    np.testing.assert_array_equal(kps[33:54], 0.0)
    np.testing.assert_array_equal(kps[54:75], 0.0)


def test_4_zero_and_nan_inputs_resilience():
    """Verify preprocessor handles zero and NaN inputs without NaN propagation."""
    frames = 50
    # Array filled with NaNs and Infs
    corrupt_array = np.full((frames, 300), np.nan, dtype=np.float32)
    corrupt_array[10, 50] = np.inf
    corrupt_array[20, 80] = -np.inf

    prep = preprocess_pose(corrupt_array, max_length=500, use_motion=True, device="cpu")
    assert not torch.isnan(prep.pose_tensor).any()
    assert not torch.isinf(prep.pose_tensor).any()
    assert prep.actual_length == frames

    # All zeros array
    zeros_array = np.zeros((frames, 300), dtype=np.float32)
    prep_zeros = preprocess_pose(zeros_array, max_length=500, use_motion=True, device="cpu")
    assert not torch.isnan(prep_zeros.pose_tensor).any()
    assert prep_zeros.actual_length == frames


def test_5_incorrect_mediapipe_outputs_handling():
    """Verify that malformed or unexpected shapes raise clean ValueErrors."""
    # Invalid channel count (e.g. 75 landmarks with 5 channels)
    bad_channels = np.zeros((10, 75, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported pose array shape"):
        to_spatial_matrix(bad_channels)

    # Empty array
    empty_arr = np.array([])
    with pytest.raises(ValueError):
        to_spatial_matrix(empty_arr)

    # Unsupported landmark count (e.g. 94 landmarks without conversion)
    bad_lms = np.zeros((10, 94, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported pose array shape"):
        to_spatial_matrix(bad_lms)


def test_6_padding_and_truncation():
    """Verify sequences < 500 are padded, and sequences > 500 are truncated."""
    # Sequence shorter than 500 (e.g. 120 frames of 300 spatial dims)
    short_seq = np.ones((120, 300), dtype=np.float32)
    padded, length = pad_or_truncate_pose(short_seq, max_length=500)
    assert padded.shape == (500, 300)
    assert length == 120
    np.testing.assert_array_equal(padded[120:], 0.0)

    # Sequence longer than 500 (e.g. 620 frames)
    long_seq = np.ones((620, 300), dtype=np.float32)
    truncated, t_length = pad_or_truncate_pose(long_seq, max_length=500)
    assert truncated.shape == (500, 300)
    assert t_length == 500


def test_7_full_real_video_extraction_and_inference(extractor, translator):
    """Verify full end-to-end extraction from real video and inference execution."""
    if not REAL_VIDEO_PATH.exists():
        pytest.skip(f"Real test video not found at {REAL_VIDEO_PATH}")

    # Extract 20 frames from real video
    raw_kps, meta = extractor.extract_isign_video(REAL_VIDEO_PATH, max_frames=20)
    assert raw_kps.shape == (20, 300)
    assert meta["read_frames"] == 20
    assert meta["pose_detection_rate"] > 0.0

    # Run inference through PoseTranslator
    result = translator.predict_with_metadata(raw_kps, num_beams=2, max_length=32)
    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert result.input_frames == 20
    assert result.input_shape == (1, 500, 300)
    assert result.latency_ms > 0


def test_8_modern_tasks_anatomical_hand_assignment():
    """Verify anatomical matching: wrist proximity to pose landmark 15 (L) and 16 (R)."""
    from ai.pose_extraction.mediapipe_extractor import _assign_hands_anatomically

    class MockLandmark:
        def __init__(self, x, y, z=0.0, visibility=1.0):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility

    class MockCategory:
        def __init__(self, name, score=0.99):
            self.category_name = name
            self.score = score

    # Pose with visible left wrist at (0.3, 0.7) and right wrist at (0.7, 0.7)
    pose_landmarks = [MockLandmark(0.5, 0.5) for _ in range(33)]
    pose_landmarks[15] = MockLandmark(0.3, 0.7, visibility=0.95)  # Left wrist
    pose_landmarks[16] = MockLandmark(0.7, 0.7, visibility=0.95)  # Right wrist

    # Case 1: Hand detected near right wrist
    hand_r = [MockLandmark(0.68, 0.71) for _ in range(21)]
    # In mirror mode, MediaPipe labels right hand as "Left"
    lh, rh = _assign_hands_anatomically([hand_r], [[MockCategory("Left")]], pose_landmarks)
    assert lh is None
    assert rh is not None
    assert rh == hand_r

    # Case 2: Hand detected near left wrist
    hand_l = [MockLandmark(0.31, 0.69) for _ in range(21)]
    lh, rh = _assign_hands_anatomically([hand_l], [[MockCategory("Right")]], pose_landmarks)
    assert lh is not None
    assert rh is None
    assert lh == hand_l

    # Case 3: Two hands detected
    lh, rh = _assign_hands_anatomically([hand_r, hand_l], [[MockCategory("Left")], [MockCategory("Right")]], pose_landmarks)
    assert lh == hand_l
    assert rh == hand_r

    # Case 4: Pose missing -> fallback to inverted handedness
    lh, rh = _assign_hands_anatomically([hand_r], [[MockCategory("Left")]], None)
    assert lh is None
    assert rh == hand_r


def test_9_modern_tasks_timestamp_monotonicity_and_metadata(extractor):
    """Verify video extraction records monotonically increasing timestamps and full metadata."""
    if not REAL_VIDEO_PATH.exists():
        pytest.skip(f"Real test video not found at {REAL_VIDEO_PATH}")

    raw_kps, meta = extractor.extract_isign_video(REAL_VIDEO_PATH, max_frames=10)
    assert raw_kps.shape == (10, 300)
    assert meta["read_frames"] == 10
    assert meta["engine"] == "modern_tasks"
    assert "extraction_fps" in meta
    assert meta["extraction_fps"] > 0
    assert "missing_hand_percentage" in meta
    assert 0.0 <= meta["missing_hand_percentage"] <= 100.0


def test_10_invalid_and_corrupted_frames_handling(extractor):
    """Verify handling of invalid or non-existent inputs without crashing."""
    # 1. Non-existent video file
    with pytest.raises(FileNotFoundError):
        extractor.extract_isign_video("non_existent_video_path.mp4")

    # 2. Blank frame
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    kps, quality = extractor.extract_isign_frame(blank_frame)
    assert kps.shape == (75, 4)
    assert quality["missing_landmarks_count"] == 75


def test_11_engine_fallback_and_preprocessing_compatibility():
    """Verify modern and legacy engines both produce compatible 75-landmark representations."""
    ext_modern = MediaPipeLandmarkExtractor(engine="modern")
    ext_legacy = MediaPipeLandmarkExtractor(engine="legacy")

    dummy_frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    kps_mod, _ = ext_modern.extract_isign_frame(dummy_frame)
    kps_leg, _ = ext_legacy.extract_isign_frame(dummy_frame)

    assert kps_mod.shape == (75, 4)
    assert kps_leg.shape == (75, 4)

    # Preprocessing pipeline works identically for both
    prep_mod = preprocess_pose(kps_mod, max_length=500, device="cpu")
    prep_leg = preprocess_pose(kps_leg, max_length=500, device="cpu")

    assert prep_mod.pose_tensor.shape == (1, 500, 300)
    assert prep_leg.pose_tensor.shape == (1, 500, 300)

    ext_modern.close()
    ext_legacy.close()

