"""
Phase 8.3.6: Train Accuracy-First Continuous ISL Baseline Model.

Architecture:
- Input: MediaPipe spatial features (T, 300)
- Projection: Linear(300, 128) + LayerNorm + Dropout(0.3) + ReLU
- Temporal Subsampling: Conv1D(128, 128, kernel=3, stride=2, padding=1) + LayerNorm + ReLU
- Temporal Recurrence: 2-layer Bidirectional GRU (hidden=128, output=256, dropout=0.3)
- Classifier Head: Dropout(0.3) + Linear(256, 171)
- Loss: CTCLoss(blank=0, zero_infinity=True)

Training Strategy:
- Split: Train (Signers 1, 3, 4, 5, 6; 499 canonical samples)
- Validation: Signer 7 (85 samples)
- Test: Signer 2 (98 samples) - evaluated once after model selection
- Batch Size: 16
- Optimizer: AdamW(lr=3e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR(T_max=80, eta_min=1e-5)
- Early Stopping: Patience=15 on Validation WER
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.isl_csltr_dataset import ISLCSLTRDataset, collate_isl_csltr_batch
from ai.continuous.isl_csltr_model import (
    ISLCSLTRContinuousModel,
    compute_sequence_metrics,
    decode_greedy_ctc,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    id_to_gloss: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a dataset split and return metrics and sample predictions."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_references: List[List[str]] = []
    all_hypotheses: List[List[str]] = []
    sample_records: List[Dict[str, Any]] = []

    blank_percentages: List[float] = []

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

            decoded = decode_greedy_ctc(log_probs, sub_lens, blank_idx=0, id_to_gloss=id_to_gloss)

            for i, dec in enumerate(decoded):
                ref = tokens_list[i]
                hyp = dec["tokens"]
                all_references.append(ref)
                all_hypotheses.append(hyp)
                blank_percentages.append(dec["blank_percentage"])

                sample_records.append({
                    "video_id": video_ids[i],
                    "signer_id": signers[i],
                    "resolution": resolutions[i],
                    "reference": ref,
                    "hypothesis": hyp,
                    "exact_match": (ref == hyp),
                    "blank_percentage": dec["blank_percentage"],
                    "total_frames": dec["total_frames"],
                })

    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    metrics = compute_sequence_metrics(all_references, all_hypotheses)
    metrics["ctc_loss"] = round(avg_loss, 4)
    metrics["mean_blank_percentage"] = round(float(np.mean(blank_percentages)), 2) if blank_percentages else 0.0
    metrics["sample_records"] = sample_records

    # Signer-specific metrics
    signer_groups: Dict[int, Dict[str, Any]] = {}
    for rec in sample_records:
        s_id = rec["signer_id"]
        if s_id not in signer_groups:
            signer_groups[s_id] = {"refs": [], "hyps": []}
        signer_groups[s_id]["refs"].append(rec["reference"])
        signer_groups[s_id]["hyps"].append(rec["hypothesis"])

    signer_breakdown = {}
    for s_id, data in signer_groups.items():
        s_metrics = compute_sequence_metrics(data["refs"], data["hyps"])
        signer_breakdown[f"signer_{s_id}"] = {
            "samples": len(data["refs"]),
            "wer": s_metrics["wer"],
            "exact_match_rate": s_metrics["exact_match_rate"],
        }
    metrics["signer_breakdown"] = signer_breakdown

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ISL-CSLTR Continuous Baseline")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=128, help="BiGRU hidden dimension")
    parser.add_argument("--proj_dim", type=int, default=128, help="Projection dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "isl_csltr_continuous"),
        help="Checkpoint output directory",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (threads={torch.get_num_threads()})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset paths
    dataset_root = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"
    train_manifest = dataset_root / "manifests" / "train_manifest.json"
    val_manifest = dataset_root / "manifests" / "validation_manifest.json"
    test_manifest = dataset_root / "manifests" / "test_manifest.json"
    vocab_path = dataset_root / "metadata" / "vocabulary.json"

    print("Loading datasets...")
    train_dataset = ISLCSLTRDataset(
        manifest_path=train_manifest,
        features_root=dataset_root / "features" / "train",
        vocab_path=vocab_path,
        canonical_only=True,
    )
    val_dataset = ISLCSLTRDataset(
        manifest_path=val_manifest,
        features_root=dataset_root / "features" / "validation",
        vocab_path=vocab_path,
        canonical_only=True,
    )
    test_dataset = ISLCSLTRDataset(
        manifest_path=test_manifest,
        features_root=dataset_root / "features" / "test",
        vocab_path=vocab_path,
        canonical_only=True,
    )

    id_to_gloss = train_dataset.id_to_gloss
    num_classes = train_dataset.num_classes

    print(f"Dataset loaded: Train={len(train_dataset)} (canonical), Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Vocabulary classes: {num_classes} (blank=0)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_isl_csltr_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_isl_csltr_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_isl_csltr_batch,
    )

    # Initialize model
    model = ISLCSLTRContinuousModel(
        input_dim=300,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Save configurations
    model_config = {
        "input_dim": 300,
        "proj_dim": args.proj_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": 2,
        "num_classes": num_classes,
        "dropout": args.dropout,
        "total_parameters": total_params,
    }
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "loss_fn": "CTCLoss(blank=0, zero_infinity=True)",
        "train_samples_effective": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }
    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    with open(output_dir / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump({
            "gloss_to_id": train_dataset.gloss_to_id,
            "id_to_gloss": {str(k): v for k, v in id_to_gloss.items()},
            "num_classes": num_classes,
            "blank_index": 0,
        }, f, indent=2)

    # Training Loop
    best_val_wer = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_history: List[Dict[str, Any]] = []

    start_train_time = time.time()
    print("\nStarting Training Loop...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val WER':>8} | {'Exact M':>8} | {'Blank %':>8} | {'LR':>9} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            log_probs, sub_lens = model(features, input_lengths)
            loss = ctc_loss_fn(log_probs, targets, sub_lens, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = features.size(0)
            train_loss += loss.item() * bs
            train_samples += bs

        scheduler.step()
        epoch_time = time.time() - t0

        avg_train_loss = train_loss / train_samples
        current_lr = scheduler.get_last_lr()[0]

        # Validation evaluation
        val_metrics = evaluate_split(model, val_loader, ctc_loss_fn, id_to_gloss, device)
        val_loss = val_metrics["ctc_loss"]
        val_wer = val_metrics["wer"]
        val_em = val_metrics["exact_match_rate"]
        val_blank_pct = val_metrics["mean_blank_percentage"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_ter": val_metrics["ter"],
            "val_exact_match_rate": val_em,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_blank_percentage": val_blank_pct,
            "lr": round(current_lr, 6),
            "epoch_time_seconds": round(epoch_time, 2),
        }
        epoch_history.append(epoch_record)

        is_best = False
        # Save best checkpoint: prioritize WER, but track loss if WER still 1.0 (warmup)
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            is_best = True

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_wer": val_wer,
                "val_loss": val_loss,
                "config": model_config,
            }, output_dir / "best_checkpoint.pt")
        elif val_wer == 1.0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Still in representation learning phase
            # Save checkpoint as current best loss
            if best_val_wer == float("inf") or best_val_wer == 1.0:
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_wer": val_wer,
                    "val_loss": val_loss,
                    "config": model_config,
                }, output_dir / "best_checkpoint.pt")
        else:
            patience_counter += 1

        best_flag = " *" if is_best else ""
        print(f"{epoch:5d} | {avg_train_loss:10.4f} | {val_loss:9.4f} | {val_wer:7.2%} | {val_em:7.2%} | {val_blank_pct:7.1f}% | {current_lr:9.6f} | {epoch_time:5.1f}s{best_flag}")

        # Check Early Stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement in validation WER for {args.patience} epochs. Stopping at epoch {epoch}.")
            break

    total_training_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes).")
    print(f"Best Validation WER: {best_val_wer:.2%} at Epoch {best_epoch}")

    # Save epoch history
    with open(output_dir / "epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)

    # Final Evaluation using Best Checkpoint
    print("\nLoading best checkpoint for final evaluation...")
    best_ckpt = torch.load(output_dir / "best_checkpoint.pt")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    print("\n--- Running Final Validation Split Evaluation (Signer 7) ---")
    final_val_eval = evaluate_split(model, val_loader, ctc_loss_fn, id_to_gloss, device)
    print(f"Final Validation Loss: {final_val_eval['ctc_loss']}")
    print(f"Final Validation WER:  {final_val_eval['wer']:.2%}")
    print(f"Final Validation TER:  {final_val_eval['ter']:.2%}")
    print(f"Final Validation Exact Match: {final_val_eval['exact_match_rate']:.2%}")
    print(f"Final Validation Macro F1:    {final_val_eval['macro_f1']:.4f}")

    print("\n--- Running Final Test Split Evaluation (Signer 2, Unseen Native Deaf Signer, QVGA) ---")
    final_test_eval = evaluate_split(model, test_loader, ctc_loss_fn, id_to_gloss, device)
    print(f"Final Test Loss: {final_test_eval['ctc_loss']}")
    print(f"Final Test WER:  {final_test_eval['wer']:.2%}")
    print(f"Final Test TER:  {final_test_eval['ter']:.2%}")
    print(f"Final Test Exact Match: {final_test_eval['exact_match_rate']:.2%}")
    print(f"Final Test Macro F1:    {final_test_eval['macro_f1']:.4f}")

    # Save complete evaluation results
    eval_results = {
        "best_epoch": best_epoch,
        "best_val_wer": best_val_wer,
        "total_training_time_seconds": round(total_training_time, 2),
        "validation_evaluation": final_val_eval,
        "test_evaluation": final_test_eval,
    }

    with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nAll artifacts and evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
