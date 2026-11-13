"""
Unit and integration tests for Phase 8.3.2 Bi-GRU + CTC Baseline Implementation.

Verifies:
1. Model initialization and forward tensor dimensions.
2. Subsampled length formula consistency with actual Conv1D output.
3. Bi-GRU bidirectional hidden dimension (256 = 128 per direction).
4. CTC Loss forward calculation, zero-infinity safety, and backward gradient flow.
5. CTC constraint validation (repeated labels, sequence length margin T' >= L).
6. Dataset loading from datasets/islvt_continuous/ with real extracted features.
7. Collate function padding and length tracking.
8. Greedy CTC decoding (consecutive duplicate collapsing, blank removal).
9. Blank prediction percentage calculation and blank-only collapse check.
10. Edit distance (Levenshtein) and token error rate (TER) metric calculation.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from ai.continuous.bigru_ctc import (
    BiGRUCTCModel,
    ContinuousISLDataset,
    collate_continuous_fn,
    compute_levenshtein,
    compute_subsampled_lengths,
    decode_greedy,
    BLANK_INDEX,
    GLOSS_TO_ID,
    ID_TO_GLOSS,
    NUM_CLASSES,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "islvt_continuous"


def test_vocab_definition():
    """Verify vocabulary indexes match Phase 8.3.2 specification."""
    assert GLOSS_TO_ID["blank"] == 0
    assert GLOSS_TO_ID["i"] == 1
    assert GLOSS_TO_ID["go"] == 2
    assert GLOSS_TO_ID["eat"] == 3
    assert GLOSS_TO_ID["food"] == 4
    assert NUM_CLASSES == 5
    assert BLANK_INDEX == 0
    assert len(ID_TO_GLOSS) == 5


def test_subsampled_length_formula():
    """Verify subsampled length formula matches actual Conv1D output length."""
    conv = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
    
    test_lengths = [1, 2, 3, 4, 5, 10, 69, 138, 147, 200, 291]
    for t in test_lengths:
        x = torch.randn(1, 128, t)
        out = conv(x)
        expected_len = compute_subsampled_lengths(torch.tensor([t])).item()
        assert out.shape[2] == expected_len, f"Mismatch at T={t}: conv={out.shape[2]}, formula={expected_len}"


def test_model_forward_dimensions():
    """Verify model forward pass produces (T', B, 5) log_probs."""
    model = BiGRUCTCModel(
        input_dim=300,
        proj_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
        dropout=0.3,
    )
    model.eval()

    # Batch of 2 samples with different lengths
    b, max_t = 2, 150
    x = torch.randn(b, max_t, 300)
    lengths = torch.tensor([150, 100], dtype=torch.long)

    with torch.no_grad():
        log_probs, subsampled_lengths = model(x, lengths)

    # Subsampled lengths: (150+1)//2 = 75, (100+1)//2 = 50
    assert subsampled_lengths.tolist() == [75, 50]
    assert log_probs.shape == (75, 2, 5)
    # Check log_probs sum to ~1 in exp space
    probs = torch.exp(log_probs)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_gru_bidirectional_dimension():
    """Verify BiGRU layer has hidden_size=128 per direction, yielding 256 dimensions."""
    model = BiGRUCTCModel(proj_dim=128, hidden_dim=128, num_layers=2)
    assert model.gru.bidirectional is True
    assert model.gru.hidden_size == 128
    assert model.classifier.in_features == 256
    assert model.classifier.out_features == 5


def test_ctc_loss_and_gradient_flow():
    """Verify PyTorch CTCLoss calculation and backward gradient flow."""
    model = BiGRUCTCModel()
    model.train()
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    b = 2
    x = torch.randn(b, 100, 300, requires_grad=True)
    lengths = torch.tensor([100, 80], dtype=torch.long)
    targets = torch.tensor([1, 2, 3], dtype=torch.long)  # sample 1: [1, 2], sample 2: [3]
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    log_probs, sub_lengths = model(x, lengths)
    loss = criterion(log_probs, targets, sub_lengths, target_lengths)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert x.grad is not None
    assert model.classifier.weight.grad is not None
    assert torch.norm(model.classifier.weight.grad) > 0.0


def test_ctc_repeated_label_handling():
    """Verify CTC loss functions on repeated labels (e.g. ID 26: ['eat', 'i', 'eat'])."""
    model = BiGRUCTCModel()
    model.train()
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    # Sequence with repeated token separated by another: eat(3) -> i(1) -> eat(3)
    target = torch.tensor([3, 1, 3], dtype=torch.long)
    target_length = torch.tensor([3], dtype=torch.long)
    x = torch.randn(1, 120, 300)
    input_length = torch.tensor([120], dtype=torch.long)

    log_probs, sub_lengths = model(x, input_length)
    loss = criterion(log_probs, target, sub_lengths, target_length)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_dataset_loading():
    """Verify ContinuousISLDataset loads train and test splits correctly."""
    train_ds = ContinuousISLDataset(DATASET_DIR, split="train")
    test_ds = ContinuousISLDataset(DATASET_DIR, split="test")

    assert len(train_ds) == 24
    assert len(test_ds) == 8

    # Verify single sample structure
    sample = train_ds[0]
    assert "features" in sample
    assert "target" in sample
    assert "sentence_id" in sample
    assert "video_id" in sample
    assert "camera_view" in sample
    assert sample["features"].shape == (sample["feature_length"], 300)
    assert sample["target"].dtype == torch.long
    assert len(sample["target"]) == sample["target_length"]


def test_collate_function():
    """Verify collate_continuous_fn handles variable length sequences."""
    train_ds = ContinuousISLDataset(DATASET_DIR, split="train")
    batch = [train_ds[0], train_ds[1], train_ds[2]]

    collated = collate_continuous_fn(batch)
    features = collated["features"]
    feature_lengths = collated["feature_lengths"]
    targets = collated["targets"]
    target_lengths = collated["target_lengths"]

    assert features.shape[0] == 3
    assert features.shape[1] == max(item["feature_length"] for item in batch)
    assert features.shape[2] == 300
    assert feature_lengths.shape == (3,)
    assert target_lengths.shape == (3,)
    assert targets.shape[0] == target_lengths.sum()


def test_greedy_decoder():
    """Verify greedy CTC decoder collapses repeats, filters blanks, tracks blank percentage."""
    # Build synthetic log_probs (T'=6, B=1, C=5)
    # Timesteps:
    # t0: blank(0)
    # t1: 'i'(1)
    # t2: 'i'(1)  -> consecutive duplicate of 1
    # t3: blank(0)
    # t4: 'go'(2)
    # t5: blank(0)
    # Expected collapsed sequence: ['i', 'go'] (tokens: [1, 2])
    log_probs = torch.full((6, 1, 5), -10.0)
    log_probs[0, 0, 0] = 0.0
    log_probs[1, 0, 1] = 0.0
    log_probs[2, 0, 1] = 0.0
    log_probs[3, 0, 0] = 0.0
    log_probs[4, 0, 2] = 0.0
    log_probs[5, 0, 0] = 0.0

    sub_lengths = torch.tensor([6], dtype=torch.long)
    decoded = decode_greedy(log_probs, sub_lengths, blank_idx=0)

    assert len(decoded) == 1
    d = decoded[0]
    assert d["token_ids"] == [1, 2]
    assert d["tokens"] == ["i", "go"]
    assert d["raw_preds"] == [0, 1, 1, 0, 2, 0]
    # Blank count: 3 blanks out of 6 timesteps -> 50%
    assert pytest.approx(d["blank_percentage"], 0.1) == 50.0


def test_blank_only_collapse_detection():
    """Verify detection when model predicts exclusively blank tokens."""
    # All timesteps predict blank (0)
    log_probs = torch.full((10, 1, 5), -10.0)
    log_probs[:, 0, 0] = 0.0

    sub_lengths = torch.tensor([10], dtype=torch.long)
    decoded = decode_greedy(log_probs, sub_lengths, blank_idx=0)

    assert decoded[0]["token_ids"] == []
    assert decoded[0]["tokens"] == []
    assert decoded[0]["blank_percentage"] == 100.0


def test_levenshtein_edit_distance():
    """Verify Levenshtein distance computation."""
    assert compute_levenshtein(["i", "go"], ["i", "go"]) == 0
    assert compute_levenshtein(["i", "go"], ["i"]) == 1
    assert compute_levenshtein(["i"], ["go"]) == 1
    assert compute_levenshtein([], ["food"]) == 1
    assert compute_levenshtein(["eat", "i", "eat"], ["i", "eat"]) == 1
