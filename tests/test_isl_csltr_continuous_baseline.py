"""
Unit tests for ISL-CSLTR Continuous Baseline Recognition (Phase 8.3.6).

Tests:
1. Vocabulary mapping (170 glosses + blank index 0 = 171 classes).
2. CTC blank handling (blank removal).
3. Repeated-token decoding (collapse consecutive identical tokens, retain tokens across blank).
4. Variable-length batching and padding masks.
5. Subsampled input length calculation (Conv1d stride-2).
6. CTC target length verification.
7. Invalid sample detection (subsampled_length < target_length).
8. Model forward pass output shape and CTCLoss compatibility.
9. Checkpoint saving, loading, and deterministic inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from ai.continuous.isl_csltr_model import (
    ISLCSLTRContinuousModel,
    compute_subsampled_lengths,
    decode_greedy_ctc,
    compute_levenshtein,
    compute_sequence_metrics,
)
from ai.continuous.isl_csltr_dataset import ISLCSLTRDataset, collate_isl_csltr_batch


@pytest.fixture
def dataset_paths():
    project_root = Path(__file__).resolve().parent.parent
    root = project_root / "datasets" / "isl_csltr_continuous"
    return {
        "root": root,
        "train_manifest": root / "manifests" / "train_manifest.json",
        "val_manifest": root / "manifests" / "validation_manifest.json",
        "test_manifest": root / "manifests" / "test_manifest.json",
        "vocab": root / "metadata" / "vocabulary.json",
        "features": root / "features",
    }


def test_vocabulary_mapping(dataset_paths):
    """Test Section 3 & 11: 170 glosses + blank 0 = 171 classes, no duplicates."""
    vocab_p = dataset_paths["vocab"]
    assert vocab_p.exists(), f"Vocab file missing at {vocab_p}"

    with open(vocab_p, "r", encoding="utf-8") as f:
        vdata = json.load(f)

    gloss_to_id = vdata["gloss_to_id"]
    blank_idx = vdata["blank_index"]
    num_classes = vdata["num_classes"]

    assert blank_idx == 0, f"Expected blank_idx 0, got {blank_idx}"
    assert num_classes == 171, f"Expected 171 classes, got {num_classes}"
    assert "<blank>" in gloss_to_id, "<blank> not in gloss_to_id"
    assert gloss_to_id["<blank>"] == 0

    # Ensure exactly 170 distinct glosses mapped to 1..170
    gloss_ids = [idx for k, idx in gloss_to_id.items() if k != "<blank>"]
    assert len(gloss_ids) == 170, f"Expected 170 gloss tokens, got {len(gloss_ids)}"
    assert len(set(gloss_ids)) == 170, "Duplicate IDs in vocabulary"
    assert min(gloss_ids) == 1
    assert max(gloss_ids) == 170


def test_ctc_blank_handling_and_repeated_tokens():
    """Test Section 6 & 11: Greedy CTC decoding collapses consecutive tokens and drops blanks."""
    id_to_gloss = {0: "<blank>", 1: "I", 2: "GO", 3: "EAT", 4: "FOOD"}

    # Case 1: [0, 1, 1, 0, 2, 0, 0, 3] -> [1, 2, 3] ("I", "GO", "EAT")
    # T=8, B=1, C=5
    raw_seq = [0, 1, 1, 0, 2, 0, 0, 3]
    probs = torch.zeros(8, 1, 5)
    for t_step, cls in enumerate(raw_seq):
        probs[t_step, 0, cls] = 10.0  # high logit

    sub_lens = torch.tensor([8])
    decoded = decode_greedy_ctc(probs, sub_lens, blank_idx=0, id_to_gloss=id_to_gloss)
    assert len(decoded) == 1
    assert decoded[0]["token_ids"] == [1, 2, 3]
    assert decoded[0]["tokens"] == ["I", "GO", "EAT"]

    # Case 2: Consecutive identical tokens with blank between -> [1, 1] ("I", "I")
    # [1, 1, 0, 1] -> [1, 1]
    raw_seq2 = [1, 1, 0, 1, 1]
    probs2 = torch.zeros(5, 1, 5)
    for t_step, cls in enumerate(raw_seq2):
        probs2[t_step, 0, cls] = 10.0

    decoded2 = decode_greedy_ctc(probs2, torch.tensor([5]), blank_idx=0, id_to_gloss=id_to_gloss)
    assert decoded2[0]["token_ids"] == [1, 1]
    assert decoded2[0]["tokens"] == ["I", "I"]

    # Case 3: All blanks -> []
    probs3 = torch.zeros(4, 1, 5)
    probs3[:, 0, 0] = 10.0
    decoded3 = decode_greedy_ctc(probs3, torch.tensor([4]), blank_idx=0, id_to_gloss=id_to_gloss)
    assert decoded3[0]["token_ids"] == []
    assert decoded3[0]["tokens"] == []
    assert decoded3[0]["blank_percentage"] == 100.0


def test_subsampled_input_lengths_and_invalid_detection():
    """Test Section 3 & 11: Length calculation and detection of invalid CTC sequences."""
    # With Conv1d(kernel=3, stride=2, padding=1): length -> (T + 1) // 2
    raw_lengths = torch.tensor([1, 2, 3, 4, 10, 25, 95, 200])
    expected = torch.tensor([1, 1, 2, 2, 5, 13, 48, 100])
    computed = compute_subsampled_lengths(raw_lengths)
    assert torch.equal(computed, expected)

    # CTC validity rule: subsampled_T >= target_length (plus repeats)
    # If target has 6 tokens, minimum subsampled T must be >= 6 (raw frames >= 11)
    target_len = 6
    short_raw_t = 8  # subsampled = 4 < 6 -> INVALID
    valid_raw_t = 15  # subsampled = 8 >= 6 -> VALID

    sub_short = compute_subsampled_lengths(torch.tensor([short_raw_t]))
    assert sub_short.item() < target_len  # detected as invalid

    sub_valid = compute_subsampled_lengths(torch.tensor([valid_raw_t]))
    assert sub_valid.item() >= target_len


def test_variable_length_collation():
    """Test Section 5 & 11: Collation correctly zero-pads and retains original lengths."""
    mock_batch = [
        {
            "video_id": "v1",
            "features": torch.ones(10, 300),
            "length": 10,
            "token_ids": torch.tensor([1, 2], dtype=torch.long),
            "tokens": ["I", "GO"],
            "signer_id": 1,
            "sentence": "i go",
            "resolution": "320x240",
        },
        {
            "video_id": "v2",
            "features": torch.full((25, 300), 2.0),
            "length": 25,
            "token_ids": torch.tensor([3, 4, 5], dtype=torch.long),
            "tokens": ["EAT", "FOOD", "NOW"],
            "signer_id": 3,
            "sentence": "eat food now",
            "resolution": "1920x1080",
        },
    ]

    collated = collate_isl_csltr_batch(mock_batch)
    features = collated["features"]
    input_lengths = collated["input_lengths"]
    targets = collated["targets"]
    target_lengths = collated["target_lengths"]

    assert features.shape == (2, 25, 300)
    assert torch.equal(input_lengths, torch.tensor([10, 25]))
    # Verify padding
    assert torch.all(features[0, 10:, :] == 0.0)
    assert torch.all(features[0, :10, :] == 1.0)
    assert torch.all(features[1, :25, :] == 2.0)

    # Verify targets concatenated 1D
    assert torch.equal(targets, torch.tensor([1, 2, 3, 4, 5]))
    assert torch.equal(target_lengths, torch.tensor([2, 3]))


def test_model_forward_and_ctc_loss():
    """Test Section 4 & 11: Forward pass produces correct tensor shapes for torch.nn.CTCLoss."""
    model = ISLCSLTRContinuousModel(
        input_dim=300,
        proj_dim=64,
        hidden_dim=64,
        num_layers=2,
        num_classes=171,
        dropout=0.1,
    )
    model.eval()

    # Batch of 2 samples with lengths 20 and 30
    x = torch.randn(2, 30, 300)
    lengths = torch.tensor([20, 30])
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    target_lengths = torch.tensor([2, 3], dtype=torch.long)

    log_probs, sub_lens = model(x, lengths)
    # Expected subsampled lengths: (20+1)//2 = 10, (30+1)//2 = 15 -> T'=15
    assert log_probs.shape == (15, 2, 171)
    assert torch.equal(sub_lens, torch.tensor([10, 15]))

    # Test CTCLoss computation
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss_fn(log_probs, targets, sub_lens, target_lengths)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0


def test_checkpoint_saving_and_reproducible_inference():
    """Test Section 10 & 11: Checkpoint save/load reproduces bit-exact inference."""
    torch.manual_seed(42)
    model = ISLCSLTRContinuousModel(
        input_dim=300,
        proj_dim=32,
        hidden_dim=32,
        num_layers=2,
        num_classes=10,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(1, 20, 300)
    lens = torch.tensor([20])
    with torch.no_grad():
        out1, _ = model(x, lens)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 300,
                "proj_dim": 32,
                "hidden_dim": 32,
                "num_layers": 2,
                "num_classes": 10,
                "dropout": 0.0,
            }
        }, ckpt_path)

        # Load back
        ckpt = torch.load(ckpt_path)
        loaded_model = ISLCSLTRContinuousModel(**ckpt["config"])
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.eval()

        with torch.no_grad():
            out2, _ = loaded_model(x, lens)

        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model inference differs"


def test_metrics_calculation():
    """Test Section 7: Levenshtein distance and WER/TER/Exact Match calculations."""
    refs = [
        ["YOU", "FREE", "TODAY"],
        ["I", "GO", "HOME"],
        ["HOW", "ARE", "YOU"],
    ]
    # Hypotheses: 1 exact match, 1 substitution, 1 deletion
    hyps = [
        ["YOU", "FREE", "TODAY"],      # exact match (dist=0)
        ["I", "WALK", "HOME"],         # 1 substitution (dist=1)
        ["HOW", "YOU"],                # 1 deletion (dist=1)
    ]
    # Total ref tokens = 3 + 3 + 3 = 9
    # Total distance = 0 + 1 + 1 = 2
    # WER = 2 / 9 = 0.2222
    # Exact match = 1 / 3 = 0.3333

    metrics = compute_sequence_metrics(refs, hyps)
    assert metrics["exact_matches"] == 1
    assert abs(metrics["exact_match_rate"] - (1/3)) < 1e-3
    assert abs(metrics["wer"] - (2/9)) < 1e-3
    assert metrics["substitutions"] == 1
    assert metrics["deletions"] == 1
    assert metrics["insertions"] == 0
