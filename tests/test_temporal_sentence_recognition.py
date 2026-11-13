"""
Tests for Temporal Sentence Recognition in Signova AI.

Validates:
1. Temporal feature engineering: 678 dimensions, zero-safe missing landmark handling, timestamp awareness.
2. Sentence dataset and collate module: variable-length sequences, attention masks, class targets.
3. TemporalSentenceClassifier architecture: forward pass, masked attention pooling, 101-class output.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_temporal_features_dimensions_and_masks():
    """Verify temporal feature extraction produces (T, 678) features with correct channels."""
    from ai.continuous.temporal_features import extract_temporal_features

    T = 30
    # 75 landmarks x 4 coordinates (x, y, z, visibility)
    landmarks = np.zeros((T, 75, 4), dtype=np.float32)
    detection_mask = np.ones((T, 3), dtype=bool)
    timestamps_ms = np.arange(0, T * 40, 40, dtype=np.int64)

    # Set shoulders for normalization (indices 11 and 12)
    landmarks[:, 11, :2] = np.array([-0.2, 0.5])
    landmarks[:, 12, :2] = np.array([0.2, 0.5])

    # Add dummy wrist movement (index 15)
    for t in range(T):
        landmarks[t, 15, :3] = np.array([0.1 * t, 0.05 * t, 0.0])

    features = extract_temporal_features(landmarks, detection_mask, timestamps_ms)

    assert features.shape == (T, 678), f"Expected shape ({T}, 678), got {features.shape}"
    assert features.dtype == np.float32

    # Spatial coords: 225, Velocity: 225, Acceleration: 225, Masks: 3
    # Masks should be 1.0
    masks = features[:, 675:678]
    assert np.allclose(masks, 1.0)


def test_temporal_features_missing_landmark_safety():
    """Verify missing landmarks do not create artificial velocity/acceleration spikes."""
    from ai.continuous.temporal_features import extract_temporal_features

    T = 10
    landmarks = np.zeros((T, 75, 4), dtype=np.float32)
    detection_mask = np.zeros((T, 3), dtype=bool)
    detection_mask[:, 0] = True  # Pose present
    detection_mask[5:, 1] = True  # Left hand only appears at frame 5

    # Shoulders
    landmarks[:, 11, :2] = np.array([-0.2, 0.5])
    landmarks[:, 12, :2] = np.array([0.2, 0.5])

    # Left hand landmarks (33..53) suddenly appear at frame 5 with nonzero coords
    landmarks[5:, 33:54, 0] = 0.5
    landmarks[5:, 33:54, 1] = 0.8

    features = extract_temporal_features(landmarks, detection_mask)

    # Left hand velocities are in range 225 + 33*3 to 225 + 54*3
    lh_vel_slice = slice(225 + 33 * 3, 225 + 54 * 3)

    # At frame 5, left hand was missing in frame 4. Velocity at frame 5 MUST BE 0.0!
    # No fake jump from 0.0 -> 0.5!
    assert np.all(features[5, lh_vel_slice] == 0.0), "Velocity at frame 5 must be 0 when previous frame was missing"

    # At frame 4, hand was missing, velocity must be 0.0
    assert np.all(features[4, lh_vel_slice] == 0.0)


def test_temporal_sentence_classifier_forward_and_attention():
    """Verify TemporalSentenceClassifier forward pass and masked attention pooling."""
    from ai.continuous.sentence_classifier import TemporalSentenceClassifier

    B, T, D = 4, 45, 678
    model = TemporalSentenceClassifier(input_dim=678, num_classes=101, hidden_dim=256)
    model.eval()

    x = torch.randn(B, T, D)
    lengths = torch.tensor([45, 30, 20, 10], dtype=torch.long)
    attention_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = True

    with torch.no_grad():
        logits = model(x, lengths=lengths, attention_mask=attention_mask)

    assert logits.shape == (B, 101), f"Expected logits shape ({B}, 101), got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN values"

    # Test predict method
    pred_res = model.predict(x[0:1, :30], lengths=torch.tensor([30]))
    assert "predicted_class" in pred_res
    assert "probabilities" in pred_res
    assert "confidence" in pred_res
    assert pred_res["probabilities"].shape == (1, 101)
    assert 0 <= pred_res["predicted_class"] < 101
    assert 0.0 <= pred_res["confidence"] <= 1.0


def test_sentence_dataset_collation():
    """Verify collation of variable-length sequences with attention mask."""
    from ai.continuous.sentence_dataset import collate_sentence_batch

    sample1 = {
        "features": torch.randn(20, 678),
        "length": 20,
        "label": 5,
        "sentence": "hello how are you",
        "video_id": "v1",
        "signer_id": 1,
    }
    sample2 = {
        "features": torch.randn(35, 678),
        "length": 35,
        "label": 12,
        "sentence": "thank you very much",
        "video_id": "v2",
        "signer_id": 3,
    }

    batch = collate_sentence_batch([sample1, sample2])
    assert batch["features"].shape == (2, 35, 678)
    assert batch["lengths"].tolist() == [20, 35]
    assert batch["labels"].tolist() == [5, 12]
    assert batch["attention_mask"].shape == (2, 35)
    assert batch["attention_mask"][0, 19].item() is True
    assert batch["attention_mask"][0, 20].item() is False
    assert batch["attention_mask"][1, 34].item() is True


def test_temporal_data_augmentation():
    """Verify temporal augmentation preserves shape and finite numbers."""
    from ai.continuous.sentence_dataset import augment_temporal_sequence

    T = 40
    feat = torch.randn(T, 678)
    aug = augment_temporal_sequence(
        feat,
        speed_range=(0.85, 1.15),
        time_mask_max_frames=4,
        num_time_masks=2,
        hand_dropout_prob=0.2,
    )
    assert aug.ndim == 2
    assert aug.shape[-1] == 678
    assert torch.isfinite(aug).all()
    # Speed jitter may scale T slightly
    assert 25 <= aug.shape[0] <= 60


def test_recognize_sentence_inference_function():
    """Verify end-to-end inference function on sample NPZ file."""
    from scripts.recognize_sentence import recognize_isl_sentence

    sample_npz = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "features" / "test" / "s000_signer2_are_you_free_today__2_.npz"
    if sample_npz.exists():
        res = recognize_isl_sentence(source=sample_npz, top_k=3, device_name="cpu")
        assert "predicted_sentence" in res
        assert "predicted_class_id" in res
        assert "confidence" in res
        assert len(res["top_candidates"]) == 3
        assert 0 <= res["predicted_class_id"] < 101
        assert 0.0 <= res["confidence"] <= 1.0
