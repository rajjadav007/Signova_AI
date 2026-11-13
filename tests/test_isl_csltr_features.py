"""
Unit and Integration Tests for Phase 8.3.5 ISL-CSLTR Landmark Extraction and Dataset Preparation.

Validates:
1. Annotation-to-video mapping (101 sentences, 170 unique glosses, exact tokenization).
2. Signer split isolation (zero signer leakage across Train, Validation, Test).
3. Duplicate leakage prevention (zero duplicate video pairs cross splits; all duplicates reside in Train).
4. Feature tensor shape validation (T, 75, 4) and flattened (T, 300).
5. Missing landmark handling (exact zero imputation, no fabricated values, no NaNs or Infs).
6. Monotonic timestamps and frame indices.
7. Outlier video capping (ensures 180s outlier is capped at <= 200 frames).
8. Manifest and metadata consistency (vocabulary.json, splits.json, duplicates.json, manifest.csv).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"
MANIFEST_DIR = DATASET_DIR / "manifests"
METADATA_DIR = DATASET_DIR / "metadata"
FEATURES_DIR = DATASET_DIR / "features"


@pytest.fixture(scope="module")
def vocabulary_data():
    vocab_path = METADATA_DIR / "vocabulary.json"
    assert vocab_path.exists(), f"Vocabulary file missing: {vocab_path}"
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def splits_data():
    splits_path = MANIFEST_DIR / "splits.json"
    assert splits_path.exists(), f"Splits file missing: {splits_path}"
    with open(splits_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def manifest_df():
    manifest_csv = MANIFEST_DIR / "manifest.csv"
    assert manifest_csv.exists(), f"Manifest CSV missing: {manifest_csv}"
    return pd.read_csv(manifest_csv)


@pytest.fixture(scope="module")
def duplicates_data():
    dup_path = MANIFEST_DIR / "duplicates.json"
    assert dup_path.exists(), f"Duplicates file missing: {dup_path}"
    with open(dup_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Test 1: Annotation-to-Video Mapping and Vocabulary
# ---------------------------------------------------------------------------
def test_annotation_mapping_and_vocabulary(vocabulary_data, manifest_df):
    """Verify 101 sentences and 170 unique glosses + 1 CTC blank class."""
    assert vocabulary_data["blank_index"] == 0
    assert vocabulary_data["num_classes"] == 171
    assert len(vocabulary_data["gloss_to_id"]) == 171
    assert vocabulary_data["gloss_to_id"]["<blank>"] == 0

    # Verify sentence IDs span 0..100
    sentence_ids = set(manifest_df["sentence_id"].unique())
    assert len(sentence_ids) == 101
    assert min(sentence_ids) == 0
    assert max(sentence_ids) == 100

    # Total videos in manifest must be exactly 687
    assert len(manifest_df) == 687, f"Expected 687 videos, got {len(manifest_df)}"


# ---------------------------------------------------------------------------
# Test 2: Signer Split Isolation (Zero Leakage)
# ---------------------------------------------------------------------------
def test_signer_split_isolation(manifest_df, splits_data):
    """Verify strict signer independence across Train, Validation, and Test."""
    train_signers = set(manifest_df[manifest_df["split"] == "train"]["signer_id"].unique())
    val_signers = set(manifest_df[manifest_df["split"] == "validation"]["signer_id"].unique())
    test_signers = set(manifest_df[manifest_df["split"] == "test"]["signer_id"].unique())

    # Verified signer groupings:
    # Train: 1, 3, 4, 5, 6
    # Validation: 7
    # Test: 2
    assert val_signers == {7}, f"Validation must only contain Signer 7, got: {val_signers}"
    assert test_signers == {2}, f"Test must only contain Signer 2, got: {test_signers}"
    assert train_signers.issubset({1, 3, 4, 5, 6}), f"Train signers unexpected: {train_signers}"

    # Zero signer overlap
    assert len(train_signers & val_signers) == 0, "Signer leakage between Train and Validation!"
    assert len(train_signers & test_signers) == 0, "Signer leakage between Train and Test!"
    assert len(val_signers & test_signers) == 0, "Signer leakage between Validation and Test!"

    assert splits_data["zero_signer_leakage"] is True


# ---------------------------------------------------------------------------
# Test 3: Duplicate Leakage Prevention
# ---------------------------------------------------------------------------
def test_duplicate_leakage_prevention(manifest_df, duplicates_data):
    """Verify all 5 duplicate pairs are contained within the same split (Train)."""
    assert len(duplicates_data) == 10, f"Expected 10 duplicate records (5 pairs), got {len(duplicates_data)}"

    # Check that duplicate groups never cross splits
    group_splits = manifest_df[manifest_df["is_duplicate"] == True].groupby("duplicate_group_id")["split"].unique()
    for gid, splits in group_splits.items():
        assert len(splits) == 1, f"Duplicate group {gid} spans multiple splits: {splits}!"
        assert splits[0] == "train", f"Duplicate group {gid} expected in train, found in {splits[0]}"


# ---------------------------------------------------------------------------
# Test 4: Extracted Feature Tensor Dimensions and Dtypes
# ---------------------------------------------------------------------------
def test_feature_tensor_dimensions_and_dtypes():
    """Verify all existing extracted .npz files have (T, 75, 4) and (T, 300) float32 shapes."""
    npz_files = list(FEATURES_DIR.glob("*/*.npz"))
    assert len(npz_files) > 0, "No extracted feature files found to validate."

    for npz_path in npz_files:
        with np.load(npz_path) as data:
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

            t = landmarks.shape[0]
            assert t > 0, f"Empty sequence in {npz_path.name}"
            assert landmarks.shape == (t, 75, 4), f"Bad landmarks shape in {npz_path.name}: {landmarks.shape}"
            assert spatial.shape == (t, 300), f"Bad spatial shape in {npz_path.name}: {spatial.shape}"
            assert landmarks.dtype == np.float32
            assert spatial.dtype == np.float32

            assert timestamps.shape == (t,)
            assert frames.shape == (t,)
            assert mask.shape == (t, 3)


# ---------------------------------------------------------------------------
# Test 5: Missing Landmark Handling (Zero-fill, No NaNs or Infs)
# ---------------------------------------------------------------------------
def test_missing_landmark_handling_and_no_nans():
    """Verify missing landmarks are zero-filled, and no NaN or Inf values exist."""
    npz_files = list(FEATURES_DIR.glob("*/*.npz"))

    for npz_path in npz_files:
        with np.load(npz_path) as data:
            landmarks = data["landmarks"]

            # Absence of NaNs and Infs
            assert not np.isnan(landmarks).any(), f"NaN found in {npz_path.name}"
            assert not np.isinf(landmarks).any(), f"Inf found in {npz_path.name}"

            # Valid confidence range [0.0, 1.0]
            conf = landmarks[:, :, 3]
            assert np.all((conf >= 0.0) & (conf <= 1.0)), f"Invalid confidence range in {npz_path.name}"

            # When confidence is 0.0, coordinates should be exactly 0.0 (hand_missing_strategy='zero')
            zero_conf_mask = (conf == 0.0)
            if np.any(zero_conf_mask):
                coords_at_zero = landmarks[:, :, :3][zero_conf_mask]
                assert np.all(coords_at_zero == 0.0), f"Non-zero coordinates with 0 confidence in {npz_path.name}"


# ---------------------------------------------------------------------------
# Test 6: Monotonic Timestamps and Frame Ordering
# ---------------------------------------------------------------------------
def test_monotonic_timestamps_and_frame_indices():
    """Verify timestamps are strictly monotonic and frame indices are 0..T-1."""
    npz_files = list(FEATURES_DIR.glob("*/*.npz"))

    for npz_path in npz_files:
        with np.load(npz_path) as data:
            timestamps = data["timestamps_ms"]
            frames = data["frame_indices"]
            t = len(timestamps)

            assert np.array_equal(frames, np.arange(t, dtype=np.int32))
            if t > 1:
                assert np.all(np.diff(timestamps) > 0), f"Non-monotonic timestamps in {npz_path.name}"


# ---------------------------------------------------------------------------
# Test 7: Outlier Duration Capping
# ---------------------------------------------------------------------------
def test_outlier_duration_capping(manifest_df):
    """Verify the 180s outlier video is properly recorded and capped."""
    outlier_rows = manifest_df[manifest_df["video_filename"].str.contains("i do not like it (2)", case=False, regex=False)]
    assert len(outlier_rows) == 1, "Expected exactly 1 outlier video row."
    outlier = outlier_rows.iloc[0]
    assert bool(outlier["is_outlier"]) is True

    # Check extracted file if it exists
    outlier_npz = FEATURES_DIR / outlier["split"] / f"{outlier['video_id']}.npz"
    if outlier_npz.exists():
        with np.load(outlier_npz) as d:
            t = d["landmarks"].shape[0]
            assert t <= 200, f"Outlier video was not capped: extracted {t} frames (expected <= 200)"
