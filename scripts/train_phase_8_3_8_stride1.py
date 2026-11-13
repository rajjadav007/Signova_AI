"""
Phase 8.3.8: Step C & D - Controlled Alternative Experiment with Stride-1 Conv1D.

Model:
- Input: MediaPipe spatial features (T, 300)
- Projection: Linear(300, 128) + LayerNorm + Dropout(0.3) + ReLU
- Temporal Conv1D: kernel=3, stride=1, padding=1, LayerNorm, ReLU (Preserves full T' = T temporal resolution)
- Temporal Recurrence: 2-layer Bidirectional GRU (hidden=128, output=256, dropout=0.3)
- Classifier Head: Dropout(0.3) + Linear(256, 171)
- Loss: CTCLoss(blank=0, zero_infinity=True)

Ponytail Design: Minimal diff, PyTorch-native, exact tracking, no unnecessary dependencies.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.isl_csltr_dataset import ISLCSLTRDataset, collate_isl_csltr_batch
from ai.continuous.isl_csltr_model import compute_sequence_metrics, decode_greedy_ctc

OUT_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "phase_8_3_8"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ISLCSLTRStride1Model(nn.Module):
    """Temporal Conv1D (stride=1) + 2-layer BiGRU + CTC Continuous ISL Model."""

    def __init__(
        self,
        input_dim: int = 300,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 171,
        dropout: float = 0.3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.stride = stride

        # 1. Feature Projection
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_ln = nn.LayerNorm(proj_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.proj_relu = nn.ReLU()

        # 2. Temporal Conv1D with stride=1 (preserves T' = T)
        self.conv = nn.Conv1d(
            in_channels=proj_dim,
            out_channels=proj_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv_ln = nn.LayerNorm(proj_dim)
        self.conv_relu = nn.ReLU()

        # 3. Two-layer Bidirectional GRU
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Classification Head
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Tensor of shape (B, T, input_dim)
            lengths: Optional 1D Tensor of shape (B,) with original lengths T_i
            
        Returns:
            log_probs: Tensor of shape (T', B, num_classes) ready for torch.nn.CTCLoss
            sub_lengths: 1D Tensor of shape (B,) with output sequence lengths T'_i
        """
        b, t, _ = x.shape
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=x.device)

        # 1. Projection
        h = self.proj(x)
        h = self.proj_ln(h)
        h = self.proj_dropout(h)
        h = self.proj_relu(h)

        # 2. Temporal Conv1d (B, C, T)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = self.conv_ln(h)
        h = self.conv_relu(h)

        # Output length calculation: floor((T + 2*1 - 1*(3-1) - 1) / stride) + 1
        if self.stride == 1:
            sub_lengths = lengths.clone().to(x.device)
        else:
            sub_lengths = ((lengths + 1) // self.stride).to(x.device)

        # 3. Packed sequence for BiGRU
        if sub_lengths.min() != sub_lengths.max():
            packed = nn.utils.rnn.pack_padded_sequence(
                h, sub_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(h)

        # 4. Classifier
        out = self.head_dropout(out)
        logits = self.classifier(out)

        # CTC expects (T', B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, sub_lengths


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    id_to_gloss: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_references: List[List[str]] = []
    all_hypotheses: List[List[str]] = []
    sample_records: List[Dict[str, Any]] = []

    all_blank_posteriors: List[float] = []
    total_output_frames = 0
    blank_argmax_frames = 0
    ref_lengths: List[int] = []
    pred_lengths: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            tokens_list = batch["tokens_list"]
            video_ids = batch["video_ids"]
            signers = batch["signers"]
            resolutions = batch["resolutions"]

            log_probs, sub_lens = model(features, input_lengths)
            loss = ctc_loss_fn(log_probs, targets, sub_lens, target_lengths)
            bs = features.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            probs = torch.exp(log_probs).transpose(0, 1)  # (B, T', C)
            decoded = decode_greedy_ctc(log_probs, sub_lens, blank_idx=0, id_to_gloss=id_to_gloss)

            for i in range(bs):
                slen = int(sub_lens[i].item())
                s_probs = probs[i, :slen]
                blank_p = s_probs[:, 0].cpu().numpy()
                argmax_tokens = torch.argmax(s_probs, dim=-1).cpu().numpy()

                all_blank_posteriors.extend(blank_p.tolist())
                total_output_frames += slen
                blank_argmax_frames += int(np.sum(argmax_tokens == 0))

                ref = tokens_list[i]
                hyp = decoded[i]["tokens"]
                all_references.append(ref)
                all_hypotheses.append(hyp)
                ref_lengths.append(len(ref))
                pred_lengths.append(len(hyp))

                if len(sample_records) < 15:
                    sample_records.append({
                        "video_id": video_ids[i],
                        "signer_id": signers[i],
                        "resolution": resolutions[i],
                        "reference": ref,
                        "hypothesis": hyp,
                        "exact_match": ref == hyp,
                    })

    mean_loss = total_loss / max(total_samples, 1)
    seq_metrics = compute_sequence_metrics(all_references, all_hypotheses)

    blank_argmax_rate = (blank_argmax_frames / max(total_output_frames, 1)) * 100.0
    mean_blank_posterior = float(np.mean(all_blank_posteriors)) if all_blank_posteriors else 0.0

    hyp_gloss_counts = Counter([g for hyp in all_hypotheses for g in hyp])
    distinct_glosses = len(hyp_gloss_counts)

    return {
        "loss": round(mean_loss, 4),
        "total_samples": total_samples,
        "wer": round(seq_metrics["wer"], 4),
        "ter": round(seq_metrics["ter"], 4),
        "exact_match_rate": round(seq_metrics["exact_match_rate"], 4),
        "macro_f1": round(seq_metrics["macro_f1"], 4),
        "substitutions": seq_metrics["substitutions"],
        "deletions": seq_metrics["deletions"],
        "insertions": seq_metrics["insertions"],
        "total_reference_tokens": seq_metrics.get("total_ref_tokens", 0),
        "blank_frame_rate_pct": round(blank_argmax_rate, 2),
        "mean_blank_posterior": round(mean_blank_posterior, 4),
        "distinct_glosses_emitted": distinct_glosses,
        "mean_predicted_length": round(float(np.mean(pred_lengths)), 2) if pred_lengths else 0.0,
        "mean_reference_length": round(float(np.mean(ref_lengths)), 2) if ref_lengths else 0.0,
        "emitted_gloss_counts": dict(hyp_gloss_counts),
        "per_gloss_metrics": seq_metrics.get("per_gloss", {}),
        "sample_predictions": sample_records,
    }


def run_smoke_test(device: torch.device) -> bool:
    print("--- Running Smoke Test ---")
    model = ISLCSLTRStride1Model(
        input_dim=300,
        proj_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=171,
        dropout=0.3,
        stride=1,
    ).to(device)

    # Fake batch: B=2, T=30, input_dim=300
    B, T, C = 2, 30, 300
    x = torch.randn(B, T, C, device=device)
    lengths = torch.tensor([30, 25], dtype=torch.long, device=device)
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device=device)
    target_lengths = torch.tensor([3, 2], dtype=torch.long, device=device)

    log_probs, sub_lens = model(x, lengths)
    assert log_probs.shape[0] == T, f"Expected T'={T}, got {log_probs.shape[0]}"
    assert log_probs.shape[1] == B, f"Expected B={B}, got {log_probs.shape[1]}"
    assert log_probs.shape[2] == 171, f"Expected C=171, got {log_probs.shape[2]}"
    assert torch.equal(sub_lens, lengths), f"Sub lengths mismatch: {sub_lens} vs {lengths}"

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = ctc_loss_fn(log_probs, targets, sub_lens, target_lengths)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    assert torch.isfinite(grad_norm), f"Gradient norm is not finite: {grad_norm}"

    # Checkpoint save/load test
    test_ckpt = OUT_DIR / "smoke_test_ckpt.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, test_ckpt)
    loaded = torch.load(test_ckpt, map_location=device)
    model.load_state_dict(loaded["model_state_dict"])
    if test_ckpt.exists():
        test_ckpt.unlink()

    print(f"Smoke test PASSED successfully. Loss: {loss.item():.4f}, Grad norm: {grad_norm:.4f}")
    return True


def run_experiment(max_epochs: int = 50, batch_size: int = 16, patience: int = 15) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"Executing Phase 8.3.8 Stride-1 Experiment on Device: {device}")

    # 1. Smoke test
    run_smoke_test(device)

    # 2. Load manifests & vocab
    train_manifest = DATASET_ROOT / "manifests" / "train_manifest.json"
    val_manifest = DATASET_ROOT / "manifests" / "validation_manifest.json"
    test_manifest = DATASET_ROOT / "manifests" / "test_manifest.json"
    vocab_path = DATASET_ROOT / "metadata" / "vocabulary.json"

    train_features = DATASET_ROOT / "features" / "train"
    val_features = DATASET_ROOT / "features" / "validation"
    test_features = DATASET_ROOT / "features" / "test"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vdata = json.load(f)
    gloss_to_id = vdata["gloss_to_id"]
    id_to_gloss = {int(v): k for k, v in gloss_to_id.items()}
    num_classes = vdata.get("num_classes", len(gloss_to_id))

    train_dataset = ISLCSLTRDataset(
        manifest_path=train_manifest,
        features_root=train_features,
        vocab_path=vocab_path,
        canonical_only=True,
    )
    val_dataset = ISLCSLTRDataset(
        manifest_path=val_manifest,
        features_root=val_features,
        vocab_path=vocab_path,
        canonical_only=True,
    )

    print(f"Train dataset size: {len(train_dataset)} canonical videos")
    print(f"Validation dataset size: {len(val_dataset)} canonical videos")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_isl_csltr_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_isl_csltr_batch,
    )

    # 3. Model setup
    set_seed(42)
    model = ISLCSLTRStride1Model(
        input_dim=300,
        proj_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        stride=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    # Save experiment config
    exp_config = {
        "experiment_name": "Phase 8.3.8 Stride-1 Temporal Learning",
        "architecture": {
            "model_class": "ISLCSLTRStride1Model",
            "input_dim": 300,
            "proj_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_classes": num_classes,
            "dropout": 0.3,
            "temporal_conv": "Conv1D(kernel=3, stride=1, padding=1)",
            "effective_temporal_resolution": "Full (T' = T, no downsampling)",
        },
        "training_hyperparameters": {
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR(T_max=50, eta_min=1e-5)",
            "gradient_clip": 5.0,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "loss": "CTCLoss(blank=0, zero_infinity=True)",
            "random_seed": 42,
            "hardware": "CPU",
        },
    }
    with open(OUT_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)

    # 4. Training loop
    best_val_wer = float("inf")
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    training_history: List[Dict[str, Any]] = []

    start_training_time = time.time()
    print("\n--- Starting Main Training ---")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            log_probs, sub_lens = model(features, input_lengths)
            loss = ctc_loss_fn(log_probs, targets, sub_lens, target_lengths)

            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch {epoch}: {loss.item()}. Skipping.")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        scheduler.step()
        epoch_train_loss = train_loss / max(train_batches, 1)

        # Validation evaluation
        val_eval = evaluate_split(model, val_loader, ctc_loss_fn, id_to_gloss, device)
        val_loss = val_eval["loss"]
        val_wer = val_eval["wer"]
        val_f1 = val_eval["macro_f1"]
        val_em = val_eval["exact_match_rate"]
        val_blank_pct = val_eval["blank_frame_rate_pct"]
        epoch_duration = round(time.time() - t0, 2)

        history_entry = {
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_ter": val_eval["ter"],
            "val_exact_match_rate": val_em,
            "val_macro_f1": val_f1,
            "val_blank_percentage": val_blank_pct,
            "lr": round(optimizer.param_groups[0]["lr"], 6),
            "epoch_time_seconds": epoch_duration,
        }
        training_history.append(history_entry)

        is_best = False
        # Save best checkpoint strictly monitored on validation WER (secondary on Macro F1)
        if val_wer < best_val_wer or (val_wer == best_val_wer and val_f1 > best_val_f1):
            best_val_wer = val_wer
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_wer": val_wer,
                    "val_macro_f1": val_f1,
                    "val_loss": val_loss,
                    "config": exp_config,
                },
                OUT_DIR / "best_checkpoint.pt",
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val WER: {val_wer:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Blank %: {val_blank_pct:.1f}% | "
            f"Time: {epoch_duration}s"
            f"{' [BEST]' if is_best else ''}"
        )

        # Check early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_training_duration = round(time.time() - start_training_time, 2)
    print(f"Training finished in {total_training_duration} seconds. Best Epoch: {best_epoch} with Val WER: {best_val_wer:.4f}")

    # Save training history
    with open(OUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    # 5. Final validation evaluation on best checkpoint
    print("\n--- Evaluating Best Checkpoint on Validation Partition ---")
    best_ckpt = torch.load(OUT_DIR / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_final_eval = evaluate_split(model, val_loader, ctc_loss_fn, id_to_gloss, device)
    val_final_eval["best_epoch"] = best_epoch
    val_final_eval["total_training_duration_seconds"] = total_training_duration

    with open(OUT_DIR / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(val_final_eval, f, indent=2)

    # 6. Model selection gate & untouched test set evaluation
    # Baseline comparison: Phase 8.3.6 Val WER = 0.8257
    baseline_val_wer = 0.8257
    print(f"\nModel Selection Verdict:")
    print(f"Baseline Phase 8.3.6 Val WER: {baseline_val_wer * 100:.2f}%")
    print(f"Phase 8.3.8 Stride-1 Val WER:  {val_final_eval['wer'] * 100:.2f}%")

    if val_final_eval["wer"] < baseline_val_wer:
        print(">> OUTCOME A: Stride-1 model improved over baseline!")
        selected_model_is_new = True
    else:
        print(">> OUTCOME B: Stride-1 model did NOT improve over baseline.")
        selected_model_is_new = False

    # Run final test evaluation per Section 10 ("Only after checkpoint selection, run one final evaluation on the untouched test set")
    print("\n--- Running Final Evaluation on Test Partition (Signer 2) ---")
    test_dataset = ISLCSLTRDataset(
        manifest_path=test_manifest,
        features_root=test_features,
        vocab_path=vocab_path,
        canonical_only=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_isl_csltr_batch,
    )
    test_eval = evaluate_split(model, test_loader, ctc_loss_fn, id_to_gloss, device)
    test_eval["evaluated_checkpoint"] = str(OUT_DIR / "best_checkpoint.pt")
    test_eval["selected_model_is_new"] = selected_model_is_new

    with open(OUT_DIR / "final_test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_eval, f, indent=2)

    print(f"Test evaluation completed. Test WER: {test_eval['wer'] * 100:.2f}%, Test Macro F1: {test_eval['macro_f1']:.4f}")
    print(f"All artifacts stored in {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run only smoke test")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(torch.device("cpu"))
    else:
        run_experiment(max_epochs=args.max_epochs, batch_size=args.batch_size, patience=args.patience)
