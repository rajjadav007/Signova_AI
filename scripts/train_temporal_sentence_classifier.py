"""
Training Pipeline for 101-Class Indian Sign Language Continuous Sentence Classification.

Script: scripts/train_temporal_sentence_classifier.py
Features:
- Configurable CLI with argparse
- Loads train, validation, test splits from ISL-CSLTR continuous dataset
- 678-dimensional temporal feature extraction with in-memory caching
- CrossEntropyLoss with smoothed inverse-frequency weights & label smoothing (0.05)
- AdamW optimizer & ReduceLROnPlateau learning rate scheduler
- Early stopping on validation loss (patience=12)
- Saves best checkpoint to models/isl_csltr_continuous/best_temporal_sentence_classifier.pt
- Evaluates held-out test split (Signer 2): top-1 accuracy, top-3 accuracy, macro-F1, per-signer metrics
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.sentence_dataset import (
    ISLCSLTRSentenceDataset,
    collate_sentence_batch,
    compute_smoothed_class_weights,
)
from ai.continuous.sentence_classifier import TemporalSentenceClassifier


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Compute top-k accuracy given logits (N, C) and targets (N,)."""
    if logits.size(0) == 0:
        return 0.0
    k = min(k, logits.size(1))
    top_k_indices = torch.topk(logits, k=k, dim=-1).indices  # (N, k)
    correct = (top_k_indices == targets.unsqueeze(1)).any(dim=1).float().sum().item()
    return correct / float(logits.size(0))


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 101,
) -> Dict[str, Any]:
    """
    Evaluate model on a DataLoader split and compute metrics.

    Returns:
        Dict containing loss, top1_accuracy, top3_accuracy, macro_f1,
        per_signer metrics, predictions list, prediction distribution,
        and per-class performance metrics.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_logits_list: List[torch.Tensor] = []
    all_signers: List[int] = []
    all_sentences: List[str] = []
    all_video_ids: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(features, lengths=lengths, attention_mask=attention_mask)
            loss = criterion(logits, targets)

            bs = features.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().tolist())
            all_logits_list.append(logits.cpu())
            all_signers.extend(batch["signer_ids"])
            all_sentences.extend(batch["sentences"])
            all_video_ids.extend(batch["video_ids"])

    if total_samples == 0:
        return {
            "loss": 0.0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "macro_f1": 0.0,
            "per_signer": {},
            "predictions": [],
            "prediction_distribution": {},
            "unique_predicted_classes": 0,
        }

    all_logits = torch.cat(all_logits_list, dim=0)  # (N, num_classes)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    avg_loss = total_loss / total_samples
    top1_acc = accuracy_score(all_targets, all_preds)
    top3_acc = compute_top_k_accuracy(all_logits, targets_tensor, k=3)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # Per-signer metrics
    per_signer: Dict[str, Dict[str, Any]] = {}
    for sid in sorted(set(all_signers)):
        s_mask = [s == sid for s in all_signers]
        s_targets = [t for t, m in zip(all_targets, s_mask) if m]
        s_preds = [p for p, m in zip(all_preds, s_mask) if m]
        s_mask_tensor = torch.tensor(s_mask, dtype=torch.bool)
        s_logits = all_logits[s_mask_tensor]
        s_targets_tensor = torch.tensor(s_targets, dtype=torch.long)

        s_top1 = accuracy_score(s_targets, s_preds)
        s_top3 = compute_top_k_accuracy(s_logits, s_targets_tensor, k=3)
        s_f1 = f1_score(s_targets, s_preds, average="macro", zero_division=0)

        per_signer[f"signer_{sid}"] = {
            "signer_id": sid,
            "num_samples": len(s_targets),
            "top1_accuracy": round(float(s_top1), 4),
            "top3_accuracy": round(float(s_top3), 4),
            "macro_f1": round(float(s_f1), 4),
        }

    # Prediction distribution & collapse inspection
    pred_counter = Counter(all_preds)
    unique_preds = len(pred_counter)
    pred_dist = {int(k): int(v) for k, v in pred_counter.most_common()}

    # Detailed per-sample predictions
    top3_all = torch.topk(all_logits, k=min(3, all_logits.size(1)), dim=-1).indices.tolist()
    probs_all = F.softmax(all_logits, dim=-1)
    confs_all = [float(probs_all[i, p].item()) for i, p in enumerate(all_preds)]

    predictions = []
    correct_examples = []
    incorrect_examples = []
    for i in range(total_samples):
        is_corr = bool(all_targets[i] == all_preds[i])
        item_pred = {
            "video_id": all_video_ids[i],
            "signer_id": all_signers[i],
            "sentence": all_sentences[i],
            "true_label": all_targets[i],
            "predicted_label": all_preds[i],
            "top3_labels": top3_all[i],
            "confidence": round(confs_all[i], 4),
            "is_correct": is_corr,
            "is_top3_correct": bool(all_targets[i] in top3_all[i]),
        }
        predictions.append(item_pred)
        if is_corr and len(correct_examples) < 10:
            correct_examples.append(item_pred)
        elif not is_corr and len(incorrect_examples) < 10:
            incorrect_examples.append(item_pred)

    # Per-class precision, recall, f1, support
    prec, rec, fscore, supp = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {}
    for c in range(num_classes):
        if supp[c] > 0 or pred_counter[c] > 0:
            per_class[str(c)] = {
                "class_id": c,
                "precision": round(float(prec[c]), 4),
                "recall": round(float(rec[c]), 4),
                "f1": round(float(fscore[c]), 4),
                "support": int(supp[c]),
                "predicted_count": int(pred_counter.get(c, 0)),
            }

    # Confusion matrix (for classes present in targets or predictions)
    present_labels = sorted(set(all_targets + all_preds))
    cm = confusion_matrix(all_targets, all_preds, labels=present_labels).tolist()

    return {
        "loss": round(float(avg_loss), 4),
        "top1_accuracy": round(float(top1_acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "total_samples": total_samples,
        "unique_predicted_classes": unique_preds,
        "prediction_distribution": pred_dist,
        "per_signer": per_signer,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present_labels,
            "matrix": cm,
        },
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples,
        "predictions": predictions,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train 101-Class Temporal ISL Continuous Sentence Classifier"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(PROJECT_ROOT / "datasets" / "isl_csltr_continuous"),
        help="Path to ISL-CSLTR continuous dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "isl_csltr_continuous"),
        help="Path to save models and evaluation reports",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for Conv1D and GRU",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout probability",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor for CrossEntropyLoss",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.5,
        help="Class weight exponential smoothing factor",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without validation score improvement)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Apply dynamic temporal data augmentation during training",
    )
    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Disable dynamic temporal data augmentation during training",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_temporal_sentence_classifier_v2.pt",
        help="Filename for the best checkpoint (prevents overwriting older models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=101,
        help="Number of target sentence classes",
    )
    return parser.parse_args()


def train_pipeline(args: argparse.Namespace) -> None:
    """Execute complete training and evaluation pipeline."""
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ISL Continuous Sentence Classifier Training")
    print(f"Device: {device}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Hyperparameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, "
          f"hidden_dim={args.hidden_dim}, dropout={args.dropout}, label_smoothing={args.label_smoothing}, "
          f"patience={args.patience}, seed={args.seed}")

    # Manifest file paths
    manifests_dir = dataset_dir / "manifests"
    train_manifest = manifests_dir / "train_manifest.json"
    val_manifest = manifests_dir / "validation_manifest.json"
    test_manifest = manifests_dir / "test_manifest.json"

    # Fallback to manifest.csv if split json manifests are not found
    if not train_manifest.exists():
        train_manifest = manifests_dir / "manifest.csv"
        val_manifest = manifests_dir / "manifest.csv"
        test_manifest = manifests_dir / "manifest.csv"

    print("\nLoading datasets with 678 temporal feature engineering...")
    t0_data = time.time()
    train_dataset = ISLCSLTRSentenceDataset(
        manifest_path=train_manifest,
        canonical_only=True,
        cache_features=True,
        augment=args.augment,
    )
    val_dataset = ISLCSLTRSentenceDataset(
        manifest_path=val_manifest,
        canonical_only=True,
        cache_features=True,
        augment=False,
    )
    test_dataset = ISLCSLTRSentenceDataset(
        manifest_path=test_manifest,
        canonical_only=True,
        cache_features=True,
        augment=False,
    )
    print(f"Datasets loaded in {time.time() - t0_data:.2f}s:")
    print(f"  Train:      {len(train_dataset)} samples (augment={args.augment})")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test:       {len(test_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sentence_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sentence_batch,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sentence_batch,
        num_workers=0,
    )

    # Compute smoothed class weights
    print(f"Computing smoothed class weights (num_classes={args.num_classes}, smoothing={args.smoothing})...")
    class_weights = compute_smoothed_class_weights(
        manifest_path=train_manifest,
        num_classes=args.num_classes,
        smoothing=args.smoothing,
    ).to(device)

    # Loss function with label smoothing and smoothed class weights
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # Initialize model
    model = TemporalSentenceClassifier(
        input_dim=678,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture initialized:")
    print(f"  Total trainable parameters: {total_params:,}")

    # Optimizer & Scheduler (Cosine Annealing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    # Label mapping dictionary
    label_map: Dict[int, str] = {}
    for r in train_dataset.records:
        sid = int(r["sentence_id"])
        if sid not in label_map and "sentence" in r:
            label_map[sid] = str(r["sentence"])
    for r in val_dataset.records:
        sid = int(r["sentence_id"])
        if sid not in label_map and "sentence" in r:
            label_map[sid] = str(r["sentence"])
    for r in test_dataset.records:
        sid = int(r["sentence_id"])
        if sid not in label_map and "sentence" in r:
            label_map[sid] = str(r["sentence"])

    stem = Path(args.checkpoint_name).stem
    best_ckpt_path = output_dir / args.checkpoint_name
    eval_results_path = output_dir / f"{stem}_evaluation.json"
    history_path = output_dir / f"{stem}_training_history.json"
    preds_path = output_dir / f"test_predictions_{stem}.json"
    best_score = -1.0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    epoch_history: List[Dict[str, Any]] = []

    print("\nStarting Training Loop...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val Top-1':>10} | {'Val Top-3':>10} | {'Val Macro-F1':>12} | {'LR':>9} | {'Time':>6}")
    print("-" * 88)

    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(features, lengths=lengths, attention_mask=attention_mask)
            loss = criterion(logits, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = features.size(0)
            train_loss += loss.item() * bs
            train_samples += bs

        avg_train_loss = train_loss / max(1, train_samples)

        # Validation evaluation
        val_eval = evaluate_split(model, val_loader, criterion, device, num_classes=args.num_classes)
        val_loss = val_eval["loss"]
        val_top1 = val_eval["top1_accuracy"]
        val_top3 = val_eval["top3_accuracy"]
        val_f1 = val_eval["macro_f1"]

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(avg_train_loss), 4),
            "val_loss": val_loss,
            "val_top1_accuracy": val_top1,
            "val_top3_accuracy": val_top3,
            "val_macro_f1": val_f1,
            "lr": round(float(current_lr), 6),
            "epoch_time_seconds": round(float(epoch_time), 2),
        }
        epoch_history.append(epoch_record)

        # Checkpoint criterion: prioritize Macro-F1 and Top-1 accuracy, with val_loss as tiebreaker
        val_score = float(val_f1 * 0.5 + val_top1 * 0.5)
        is_best = False
        if val_score > best_score or (abs(val_score - best_score) < 1e-5 and val_loss < best_val_loss):
            best_score = val_score
            best_val_loss = val_loss
            best_val_acc = val_top1
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            is_best = True

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_top1,
                "val_top3_accuracy": val_top3,
                "val_macro_f1": val_f1,
                "model_config": {
                    "input_dim": 678,
                    "num_classes": args.num_classes,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": 2,
                    "dropout": args.dropout,
                    "total_parameters": total_params,
                },
                "training_config": vars(args),
                "label_map": {str(k): v for k, v in label_map.items()},
            }, best_ckpt_path)
        else:
            patience_counter += 1

        best_flag = " *" if is_best else ""
        print(f"{epoch:5d} | {avg_train_loss:10.4f} | {val_loss:9.4f} | {val_top1:9.2%} | {val_top3:9.2%} | {val_f1:12.4f} | {current_lr:9.6f} | {epoch_time:5.1f}s{best_flag}")

        # Check early stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No validation score improvement for {args.patience} epochs. Stopping at epoch {epoch}.")
            break

    total_time = time.time() - t_train_start
    print(f"\nTraining completed in {total_time:.2f}s ({total_time / 60:.2f} minutes).")
    print(f"Best Checkpoint: Epoch {best_epoch} (Val Top-1: {best_val_acc:.2%}, Macro-F1: {best_val_f1:.4f}, Loss: {best_val_loss:.4f})")

    # Load best checkpoint for final evaluation
    print(f"\nLoading best checkpoint from {best_ckpt_path} for evaluation...")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Final Validation Split Evaluation (Signer 7)
    print("\n" + "=" * 50)
    print("FINAL VALIDATION EVALUATION (Signer 7)")
    print("=" * 50)
    final_val_eval = evaluate_split(model, val_loader, criterion, device, num_classes=args.num_classes)
    print(f"Validation Loss:     {final_val_eval['loss']:.4f}")
    print(f"Validation Top-1 Acc: {final_val_eval['top1_accuracy']:.2%}")
    print(f"Validation Top-3 Acc: {final_val_eval['top3_accuracy']:.2%}")
    print(f"Validation Macro-F1:  {final_val_eval['macro_f1']:.4f}")

    # Final Held-Out Test Split Evaluation (Signer 2)
    print("\n" + "=" * 50)
    print("FINAL HELD-OUT TEST EVALUATION (Signer 2)")
    print("=" * 50)
    test_eval = evaluate_split(model, test_loader, criterion, device, num_classes=args.num_classes)
    print(f"Test Loss:           {test_eval['loss']:.4f}")
    print(f"Test Top-1 Accuracy: {test_eval['top1_accuracy']:.2%}")
    print(f"Test Top-3 Accuracy: {test_eval['top3_accuracy']:.2%}")
    print(f"Test Macro-F1:       {test_eval['macro_f1']:.4f}")

    print("\nPer-Signer Performance:")
    for signer_key, metrics in test_eval["per_signer"].items():
        print(f"  {signer_key}: N={metrics['num_samples']}, Top-1={metrics['top1_accuracy']:.2%}, "
              f"Top-3={metrics['top3_accuracy']:.2%}, Macro-F1={metrics['macro_f1']:.4f}")

    # Save summary and evaluation metadata
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_architecture": "TemporalSentenceClassifier (2x Conv1D residual + 2-layer BiGRU + Masked Temporal Attention)",
        "input_features": "678-dim Temporal Features (225 spatial + 225 velocity + 225 acceleration + 3 detection masks)",
        "num_classes": args.num_classes,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_macro_f1": best_val_f1,
        "training_time_seconds": round(float(total_time), 2),
        "validation_metrics": final_val_eval,
        "test_metrics": test_eval,
        "collapse_analysis": {
            "unique_predicted_test_classes": test_eval["unique_predicted_classes"],
            "test_prediction_distribution": test_eval["prediction_distribution"],
            "is_collapsed": bool(test_eval["unique_predicted_classes"] <= 3),
        },
        "hyperparameters": vars(args),
    }

    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nEvaluation summary saved to: {eval_results_path}")

    # Save training history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)
    print(f"Training history saved to:   {history_path}")

    # Save detailed predictions
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(test_eval["predictions"], f, indent=2)
    print(f"Sample test predictions saved to: {preds_path}")


if __name__ == "__main__":
    train_pipeline(parse_args())
