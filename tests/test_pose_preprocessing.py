"""
Unit tests for F-014: Input Preparation for iSign-T5.
Tests synthetic landmark sequences, component ordering, temporal normalization,
velocity computation, padding/truncation, error handling, and model compatibility.

NOTE: All tests use synthetic landmarks to verify mathematical and architectural logic.
No fabricated ISL recognition or translation is performed.
"""

import numpy as np
import pytest
import torch

from ai.inference.model_loader import load_model, unload_model
from ai.preprocessing import (
    BASE_SPATIAL_DIM,
    CHANNELS_PER_LANDMARK,
    DEFAULT_MAX_POSE_LENGTH,
    HAND_LANDMARK_COUNT,
    MOTION_DIM,
    POSE_LANDMARK_COUNT,
    TOTAL_FEATURE_DIM,
    TOTAL_LANDMARKS,
    PreprocessedPose,
    add_motion_features,
    assemble_frame_landmarks,
    normalize_pose,
    pad_or_truncate_pose,
    preprocess_pose,
    to_spatial_matrix,
)


@pytest.fixture(scope="module")
def shared_model():
    """Load model once for compatibility tests and unload at end of module."""
    model = load_model(device="cpu")
    yield model
    unload_model()


def test_1_valid_pose_input_formats():
    """1. Accepts multiple valid input formats: (F, 300), (F, 75, 4), and structured dict."""
    frames = 25

    # 1a. Array of shape (frames, 300) with default simple model (use_motion=False)
    arr_300 = np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32)
    prep_300 = preprocess_pose(arr_300, max_length=500, use_motion=False)
    assert isinstance(prep_300, PreprocessedPose)
    assert prep_300.pose_tensor.shape == (1, 500, 300)
    assert prep_300.actual_length == frames

    # 1b. Array of shape (frames, 75, 4)
    arr_75_4 = np.random.randn(frames, TOTAL_LANDMARKS, CHANNELS_PER_LANDMARK).astype(np.float32)
    prep_75_4 = preprocess_pose(arr_75_4, max_length=500, use_motion=False)
    assert prep_75_4.pose_tensor.shape == (1, 500, 300)
    assert prep_75_4.actual_length == frames

    # 1c. Sequence dictionary
    dict_input = {
        "pose": np.random.randn(frames, POSE_LANDMARK_COUNT, 4).astype(np.float32),
        "left_hand": np.random.randn(frames, HAND_LANDMARK_COUNT, 4).astype(np.float32),
        "right_hand": np.random.randn(frames, HAND_LANDMARK_COUNT, 4).astype(np.float32),
    }
    prep_dict = preprocess_pose(dict_input, max_length=500, use_motion=False)
    assert prep_dict.pose_tensor.shape == (1, 500, 300)
    assert prep_dict.actual_length == frames


def test_2_landmark_ordering():
    """2. Verifies strict landmark ordering: Pose (0..32), Left hand (33..53), Right hand (54..74)."""
    # Create distinguishable landmark coordinates
    pose_pts = np.full((POSE_LANDMARK_COUNT, 4), 1.0, dtype=np.float32)
    lh_pts = np.full((HAND_LANDMARK_COUNT, 4), 2.0, dtype=np.float32)
    rh_pts = np.full((HAND_LANDMARK_COUNT, 4), 3.0, dtype=np.float32)

    vec = assemble_frame_landmarks(pose=pose_pts, left_hand=lh_pts, right_hand=rh_pts)
    assert vec.shape == (300,)

    # Pose landmarks: 0..32 -> first 33 * 4 = 132 elements should be 1.0
    np.testing.assert_array_equal(vec[:132], 1.0)
    # Left hand landmarks: 33..53 -> next 21 * 4 = 84 elements (132..216) should be 2.0
    np.testing.assert_array_equal(vec[132:216], 2.0)
    # Right hand landmarks: 54..74 -> final 21 * 4 = 84 elements (216..300) should be 3.0
    np.testing.assert_array_equal(vec[216:300], 3.0)


def test_3_spatial_feature_extraction_and_missing_components():
    """3. Spatial feature extraction handles missing components by zero-filling."""
    # When left hand is None (e.g. hand out of frame), its 84 features should be zero
    pose_pts = np.ones((POSE_LANDMARK_COUNT, 4), dtype=np.float32)
    rh_pts = np.ones((HAND_LANDMARK_COUNT, 4), dtype=np.float32)

    vec = assemble_frame_landmarks(pose=pose_pts, left_hand=None, right_hand=rh_pts)
    assert vec.shape == (300,)
    # Pose: non-zero
    np.testing.assert_array_equal(vec[:132], 1.0)
    # Left hand: all zeros
    np.testing.assert_array_equal(vec[132:216], 0.0)
    # Right hand: non-zero
    np.testing.assert_array_equal(vec[216:300], 1.0)


def test_4_zscore_normalization():
    """4. Temporal Z-score normalization produces zero mean and unit variance per channel."""
    frames = 50
    dims = BASE_SPATIAL_DIM
    # Generate data with known mean 5.0 and std 2.0
    raw_data = np.random.randn(frames, dims).astype(np.float32) * 2.0 + 5.0

    normalized = normalize_pose(raw_data, method="zscore")
    assert normalized.shape == (frames, dims)

    # Check temporal mean is close to 0 and std is close to 1
    mean = np.mean(normalized, axis=0)
    std = np.std(normalized, axis=0)
    np.testing.assert_allclose(mean, 0.0, atol=1e-5)
    np.testing.assert_allclose(std, 1.0, atol=1e-5)

    # Constant channel edge case: std == 0 should not raise division by zero
    constant_data = np.full((frames, dims), 42.0, dtype=np.float32)
    norm_constant = normalize_pose(constant_data, method="zscore")
    np.testing.assert_array_equal(norm_constant, 0.0)


def test_5_motion_feature_calculation():
    """5. Motion feature calculation appends temporal velocity to spatial features."""
    frames = 10
    dims = 300
    # Sequential values: frame t has value t * 10
    raw = np.tile(np.arange(frames, dtype=np.float32)[:, None], (1, dims)) * 10.0

    with_motion = add_motion_features(raw)
    assert with_motion.shape == (frames, 600)

    # First half is spatial
    np.testing.assert_array_equal(with_motion[:, :300], raw)
    # Velocity for frame 1..9 is 10.0
    np.testing.assert_allclose(with_motion[1:, 300:], 10.0)


def test_6_first_frame_velocity_behavior():
    """6. First frame velocity is strictly zero."""
    frames = 15
    dims = 300
    raw = np.random.randn(frames, dims).astype(np.float32)

    with_motion = add_motion_features(raw)
    assert with_motion.shape == (frames, 600)
    # Velocity at index 0 must be 0 for all channels
    np.testing.assert_array_equal(with_motion[0, 300:], 0.0)


def test_7_padding_behavior():
    """7. Sequences shorter than 500 are padded with zeros."""
    frames = 30
    raw = np.ones((frames, 600), dtype=np.float32)

    padded, length = pad_or_truncate_pose(raw, max_length=500)
    assert padded.shape == (500, 600)
    assert length == 30
    # Active region is 1.0
    np.testing.assert_array_equal(padded[:30], 1.0)
    # Padded region is 0.0
    np.testing.assert_array_equal(padded[30:], 0.0)


def test_8_truncation_behavior():
    """8. Sequences longer than 500 are truncated to 500 frames."""
    frames = 550
    raw = np.ones((frames, 600), dtype=np.float32)

    padded, length = pad_or_truncate_pose(raw, max_length=500)
    assert padded.shape == (500, 600)
    assert length == 500
    np.testing.assert_array_equal(padded, 1.0)


def test_9_output_dimensions_and_types():
    """9. Verifies exact output shapes and PyTorch dtypes for simple model."""
    frames = 40
    raw = np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32)
    prep = preprocess_pose(raw, max_length=500, use_motion=False)

    # Tensor shape and dtype (simple model: 300 dimensions)
    assert prep.pose_tensor.shape == (1, 500, 300)
    assert prep.pose_tensor.dtype == torch.float32
    assert prep.pose_length.shape == (1,)
    assert prep.pose_length.dtype == torch.long
    assert prep.pose_length.item() == 40

    # NumPy array shape and dtype
    assert prep.features.shape == (500, 300)
    assert prep.features.dtype == np.float32
    assert prep.actual_length == 40


def test_10_invalid_and_empty_input_handling():
    """10. Empty sequences or invalid shapes raise clear ValueError."""
    # Empty array
    with pytest.raises(ValueError, match="empty"):
        preprocess_pose(np.empty((0, 300), dtype=np.float32))

    # Empty list
    with pytest.raises(ValueError, match="empty"):
        preprocess_pose([])

    # Incompatible shape (e.g. 150 dims instead of 300)
    with pytest.raises(ValueError, match="Unsupported pose array shape"):
        preprocess_pose(np.random.randn(10, 150).astype(np.float32))


def test_11_nan_and_infinite_values_cleaned():
    """11. Input with NaNs and Infs is sanitized with no NaNs/Infs remaining."""
    frames = 20
    raw = np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32)
    # Inject NaNs and Infs
    raw[2, 10:20] = np.nan
    raw[5, 40:50] = np.inf
    raw[8, 70:80] = -np.inf

    prep = preprocess_pose(raw, max_length=500, use_motion=False)

    assert not torch.isnan(prep.pose_tensor).any(), "NaN found in output tensor"
    assert not torch.isinf(prep.pose_tensor).any(), "Inf found in output tensor"
    assert not np.isnan(prep.features).any(), "NaN found in output features"
    assert not np.isinf(prep.features).any(), "Inf found in output features"


def test_12_compatibility_with_pose_encoder(shared_model):
    """12. Verified tensor compatibility with t5-base simple pose encoder (300 in, 768 out)."""
    frames = 35
    raw = np.random.randn(frames, BASE_SPATIAL_DIM).astype(np.float32)
    prep = preprocess_pose(raw, max_length=500, use_motion=False, device="cpu")

    # Pose encoder forward pass
    with torch.no_grad():
        encoded = shared_model.pose_encoder(prep.pose_tensor)

    # Input: (1, 500, 300) -> Output: (1, 500, 768)
    assert encoded.shape == (1, 500, 768), f"Expected (1, 500, 768), got {encoded.shape}"
    assert encoded.dtype == torch.float32
    assert not torch.isnan(encoded).any()
