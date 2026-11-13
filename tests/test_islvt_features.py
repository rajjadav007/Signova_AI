"""
Validation suite for Phase 8.2 ISLVT Feature Extraction Pipeline.

Validates:
1. Video-to-annotation mapping.
2. Landmark tensor dimensions (T, 75, 4) and (T, 300).
3. Missing landmark rates and zero-filling (no fabricated landmarks, no NaNs/Infs).
4. Sequence validity and non-emptiness (T >= 50).
5. Duplicate recordings detection (ID 4 side-take duplication).
6. Camera-pair consistency and leakage prevention (same split for paired views).
7. Feature file readability and metadata schema integrity.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "islvt_continuous"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
SPLITS_PATH = OUTPUT_DIR / "splits.json"
SUMMARY_PATH = OUTPUT_DIR / "feature_summary.json"


@pytest.fixture(scope="module")
def manifest_data():
    """Load manifest records."""
    assert MANIFEST_PATH.exists(), f"Manifest file missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) > 0, "Manifest contains 0 records."
    return records


@pytest.fixture(scope="module")
def splits_data():
    """Load splits definition."""
    assert SPLITS_PATH.exists(), f"Splits file missing: {SPLITS_PATH}"
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_and_summary_files_exist():
    """Verify all required Phase 8.2 output manifest files exist."""
    assert MANIFEST_PATH.exists()
    assert MANIFEST_CSV.exists()
    assert SPLITS_PATH.exists()
    assert SUMMARY_PATH.exists()

    df = pd.read_csv(MANIFEST_CSV)
    assert len(df) == 32, f"Expected 32 extracted videos, got {len(df)}"


def test_video_to_annotation_mapping(manifest_data):
    """Verify video-to-annotation mapping for the 16 target sentences."""
    sentence_ids = set(r["sentence_id"] for r in manifest_data)
    expected_ids = {4, 11, 12, 13, 17, 18, 20, 21, 26, 31, 34, 44, 50, 63, 67, 72}
    assert sentence_ids == expected_ids, f"Mismatch in sentence IDs: {sentence_ids ^ expected_ids}"

    # Verify each sentence has valid tokens and at least one target gloss
    for r in manifest_data:
        assert len(r["tokens"]) > 0
        assert len(r["target_glosses"]) > 0
        assert any(t in {"i", "go", "eat", "food"} for t in r["target_glosses"])


def test_landmark_tensor_dimensions(manifest_data):
    """Verify (T, 75, 4) and (T, 300) landmark tensor dimensions and dtypes."""
    for r in manifest_data:
        rel_path = r["npz_relative_path"]
        npz_file = OUTPUT_DIR / rel_path
        assert npz_file.exists(), f"NPZ file not found: {npz_file}"

        with np.load(npz_file) as data:
            assert "landmarks" in data
            assert "spatial_features" in data
            assert "timestamps_ms" in data
            assert "frame_indices" in data
            assert "detection_mask" in data
            assert "metadata_json" in data

            landmarks = data["landmarks"]
            spatial = data["spatial_features"]
            timestamps = data["timestamps_ms"]
            frames = data["frame_indices"]
            mask = data["detection_mask"]

            t_frames = landmarks.shape[0]
            assert t_frames == r["frame_count"]
            assert landmarks.shape == (t_frames, 75, 4)
            assert landmarks.dtype == np.float32

            assert spatial.shape == (t_frames, 300)
            assert spatial.dtype == np.float32

            assert timestamps.shape == (t_frames,)
            assert np.all(np.diff(timestamps) > 0), "Timestamps must be strictly monotonic"

            assert mask.shape == (t_frames, 3)
            assert mask.dtype == bool


def test_missing_landmark_handling_and_no_nans(manifest_data):
    """Verify landmark missing handling: exact zeros for unobserved, no NaNs or Infs."""
    for r in manifest_data:
        npz_file = OUTPUT_DIR / r["npz_relative_path"]
        with np.load(npz_file) as data:
            landmarks = data["landmarks"]

            # No NaNs or Infs anywhere
            assert not np.isnan(landmarks).any(), f"NaN found in {r['video_filename']}"
            assert not np.isinf(landmarks).any(), f"Inf found in {r['video_filename']}"

            # Pose detection rate should be high (> 90%)
            assert r["pose_detection_rate"] >= 0.90, f"Low pose rate in {r['video_filename']}: {r['pose_detection_rate']}"

            # Coordinates are bounded normalized [0, 1] with margin for natural overshoot
            # Unobserved landmarks must be strictly 0.0
            unobserved_mask = (landmarks[:, :, 3] == 0.0)
            if np.any(unobserved_mask):
                unobserved_coords = landmarks[unobserved_mask][:, :3]
                assert np.all(unobserved_coords == 0.0), "Unobserved landmarks must have 0.0 coordinates (no fabrication)"


def test_sequence_length_and_validity(manifest_data):
    """Verify all sequences are valid and non-empty (T >= 50 frames, duration >= 2.0s)."""
    for r in manifest_data:
        assert r["frame_count"] >= 50, f"Sequence too short: {r['frame_count']} in {r['video_filename']}"
        assert r["duration_seconds"] >= 2.0
        assert r["fps"] == 25.0
        assert r["resolution"] == "1920x1080"


def test_duplicate_recordings_and_single_views(manifest_data):
    """Verify duplicate takes and missing angle for ID 4 are properly handled."""
    id4_records = [r for r in manifest_data if r["sentence_id"] == 4]
    assert len(id4_records) == 2
    for r in id4_records:
        assert r["camera_view"] == "side"
        assert r["pair_status"] == "duplicate_side_take_no_front"


def test_camera_pair_consistency_and_splits(manifest_data, splits_data):
    """Verify front and side views are strictly kept in the same split (leakage prevention)."""
    by_sid = {}
    for r in manifest_data:
        by_sid.setdefault(r["sentence_id"], []).append(r)

    train_count = 0
    test_count = 0

    for sid, recs in by_sid.items():
        assert len(recs) == 2, f"Sentence {sid} should have 2 videos"
        splits = set(r["split"] for r in recs)
        assert len(splits) == 1, f"Sentence {sid} has videos in conflicting splits: {splits}"
        split = recs[0]["split"]
        if split == "train":
            train_count += len(recs)
        else:
            test_count += len(recs)

    assert train_count == 24  # 12 sentences x 2
    assert test_count == 8    # 4 sentences x 2

    # Verify all 4 target words are in both train and test
    train_targets = set()
    test_targets = set()
    for r in manifest_data:
        if r["split"] == "train":
            train_targets.update(r["target_glosses"])
        else:
            test_targets.update(r["target_glosses"])

    expected = {"i", "go", "eat", "food"}
    assert train_targets == expected, f"Train missing target words: {expected - train_targets}"
    assert test_targets == expected, f"Test missing target words: {expected - test_targets}"


def test_metadata_schema_integrity(manifest_data):
    """Verify required metadata fields are stored within each NPZ archive."""
    required_keys = [
        "video_id", "video_filename", "sentence_id", "split", "camera_view",
        "english", "marathi", "isl_gloss", "tokens", "target_glosses",
        "frame_count", "fps", "duration_seconds", "resolution", "feature_dims",
        "pose_detection_rate", "left_hand_detection_rate", "right_hand_detection_rate",
        "average_pose_visibility", "missing_landmark_percentage"
    ]
    for r in manifest_data:
        npz_file = OUTPUT_DIR / r["npz_relative_path"]
        with np.load(npz_file) as data:
            meta = json.loads(str(data["metadata_json"]))
            for k in required_keys:
                assert k in meta, f"Key '{k}' missing from NPZ metadata in {r['video_filename']}"
