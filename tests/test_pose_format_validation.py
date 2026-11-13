"""
Validation tests for F-012.1: Pose Format Dependency Setup and Preprocessing Verification.
"""

import importlib.metadata
from pathlib import Path
import sys
import numpy as np
import pytest

# Add official inference directory to path for pose_utils testing
OFFICIAL_INFERENCE_DIR = Path("D:/iSign-t5-pose-to-text/inference")
if str(OFFICIAL_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_INFERENCE_DIR))


def test_pose_format_installed_and_imports():
    """1. pose-format dependency imports successfully and version is recorded."""
    import pose_format
    from pose_format import Pose

    assert Pose is not None
    version = importlib.metadata.version("pose-format")
    assert version == "0.15.0", f"Expected pose-format 0.15.0, got {version}"


@pytest.mark.skip(
    reason="BLOCKED: No official .pose sample files are included in the downloaded model repository D:\\iSign-t5-pose-to-text"
)
def test_official_pose_sample_parsing():
    """2. Test parsing an official .pose sample (blocked because no sample exists in model repo)."""
    model_dir = Path("D:/iSign-t5-pose-to-text")
    pose_files = list(model_dir.rglob("*.pose"))
    assert len(pose_files) > 0, "No official .pose file available in model repository"

    from pose_format import Pose

    sample_pose = pose_files[0]
    with open(sample_pose, "rb") as f:
        pose = Pose.read(f.read())
    assert pose is not None
    assert pose.body is not None


def test_official_preprocessing_zscore_normalization():
    """4a. Verify official normalize_pose (zscore) implementation."""
    from pose_utils import normalize_pose

    # Synthetic 10 frames of 300 features
    dummy_data = np.random.randn(10, 300).astype(np.float32) * 5.0 + 10.0
    normalized = normalize_pose(dummy_data, method="zscore")

    assert normalized.shape == (10, 300)
    # Check mean across temporal dimension is close to 0
    np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=1e-5)
    # Check std across temporal dimension is close to 1
    np.testing.assert_allclose(np.std(normalized, axis=0), 1.0, atol=1e-5)


def test_official_preprocessing_motion_features():
    """4b. Verify official add_motion_features calculation for t5-small-motion."""
    from pose_utils import add_motion_features

    # 5 frames, 300 spatial dims
    frames = 5
    dims = 300
    dummy_pose = np.arange(frames * dims, dtype=np.float32).reshape(frames, dims)

    with_motion = add_motion_features(dummy_pose)
    assert with_motion.shape == (frames, dims * 2)  # (5, 600)

    # First half is spatial
    np.testing.assert_array_equal(with_motion[:, :dims], dummy_pose)
    # Frame 0 velocity is zeros
    np.testing.assert_array_equal(with_motion[0, dims:], np.zeros(dims, dtype=np.float32))
    # Frame 1 velocity is frame 1 - frame 0
    np.testing.assert_allclose(with_motion[1, dims:], dummy_pose[1] - dummy_pose[0])


def test_official_preprocessing_padding_and_truncation():
    """4c. Verify pad_or_truncate_pose sequence bounding to 500 frames."""
    from pose_utils import pad_or_truncate_pose

    # Short sequence (20 frames)
    short_seq = np.ones((20, 600), dtype=np.float32)
    padded, length = pad_or_truncate_pose(short_seq, max_length=500)
    assert padded.shape == (500, 600)
    assert length == 20
    assert np.all(padded[20:] == 0.0)

    # Long sequence (600 frames)
    long_seq = np.ones((600, 600), dtype=np.float32)
    truncated, length_trunc = pad_or_truncate_pose(long_seq, max_length=500)
    assert truncated.shape == (500, 600)
    assert length_trunc == 500


def test_official_json_pose_format_parsing():
    """4d. Verify official load_pose_file JSON parser branch."""
    import json
    import tempfile
    from pose_utils import load_pose_file, preprocess_pose

    # The official load_pose_file supports JSON format: {"data": [{"keypoints": [...]}]}
    frames = 15
    dims = 300
    sample_payload = {
        "data": [
            {"keypoints": [float(i) for i in range(dims)]}
            for _ in range(frames)
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(sample_payload, tmp)
        tmp_path = tmp.name

    try:
        raw_loaded = load_pose_file(tmp_path)
        assert raw_loaded.shape == (frames, dims)

        # Run complete official preprocess_pose for t5-small-motion (pose_dim=600, use_motion=True)
        processed, actual_len = preprocess_pose(
            tmp_path,
            max_length=500,
            pose_dim=600,
            use_motion=True,
            normalization="zscore",
        )
        assert processed.shape == (500, 600)
        assert actual_len == frames
    finally:
        Path(tmp_path).unlink(missing_ok=True)
