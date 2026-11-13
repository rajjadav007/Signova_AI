"""
Experiment D: Temporal Convolutional Network (Temporal CNN / Dilated 1D ResNet).

Evaluates:
- Pure 1D convolutional architecture replacing the BiGRU + Attention baseline.
- Variable-length ordered temporal sequences with padding attention masks.
- Dilated 1D residual blocks (dilations: 1, 2, 4) capturing multi-scale temporal context.
- Masked global temporal average pooling + Linear classifier (101 classes).
- Identical loss configuration as BiGRU baseline (smoothed class weights + label smoothing 0.05).
- Evaluates on validation split (Signer 7, 85 samples).
- Saves checkpoint to models/isl_csltr_continuous/best_temporal_cnn_checkpoint.pt and report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.sentence_dataset import (
    ISLCSLTRSentenceDataset,
    collate_sentence_batch,
    compute_smoothed_class_weights,
)
from scripts.eval_utils import compute_benchmark_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DilatedResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class DilatedTemporalCNN(nn.Module):
    """
    Dilated 1D Temporal Convolutional Network for Continuous Sign Language Sentence Classification.
    """
    def __init__(
        self,
        input_dim: int = 678,
        num_classes: int = 101,
        base_channels: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )

        self.block1 = DilatedResidualBlock1D(base_channels, base_channels, dilation=1, dropout=dropout)
        self.block2 = DilatedResidualBlock1D(base_channels, base_channels * 2, dilation=2, dropout=dropout)
        self.block3 = DilatedResidualBlock1D(base_channels * 2, base_channels * 2, dilation=4, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, 678)
        attention_mask: (B, T) boolean or float
        Returns: logits (B, num_classes)
        """
        # (B, T, D) -> (B, D, T)
        x_conv = x.transpose(1, 2)
        h = self.input_proj(x_conv)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)  # (B, 2*base_channels, T)

        # Masked temporal pooling
        if attention_mask is not None:
            # mask: (B, 1, T)
            mask = attention_mask.unsqueeze(1).float()
            h_masked = h * mask
            sum_mask = mask.sum(dim=-1).clamp(min=1.0)
            pooled = h_masked.sum(dim=-1) / sum_mask  # (B, 2*base_channels)
        else:
            pooled = h.mean(dim=-1)

        logits = self.classifier(pooled)
        return logits


def evaluate_temporal_cnn(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []

    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)  # (B, T, 678)
            targets = batch["labels"].to(device)     # (B,)
            mask = batch["attention_mask"].to(device) # (B, T)

            logits = model(features, attention_mask=mask)
            loss = criterion(logits, targets)

            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            total_loss += loss.item() * targets.size(0)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.append(probs)

    infer_time = time.time() - t0
    avg_loss = total_loss / max(1, len(all_targets))
    all_probs_arr = np.concatenate(all_probs, axis=0)

    metrics = compute_benchmark_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_preds),
        y_probs=all_probs_arr,
        num_classes=101,
        train_time_sec=0.0,
        inference_time_sec=infer_time,
    )
    metrics["loss"] = float(avg_loss)
    return metrics


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Temporal CNN on device: {device}")

    manifests_dir = PROJECT_ROOT / "datasets" / "isl_csltr_continuous" / "manifests"
    models_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load datasets
    train_manifest = manifests_dir / "train_manifest.json"
    val_manifest = manifests_dir / "validation_manifest.json"

    train_ds = ISLCSLTRSentenceDataset(
        manifest_path=train_manifest,
        canonical_only=True,
        cache_features=True,
        augment=False,
    )
    val_ds = ISLCSLTRSentenceDataset(
        manifest_path=val_manifest,
        canonical_only=True,
        cache_features=True,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_sentence_batch,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_sentence_batch,
        num_workers=0,
    )

    # Smoothed class weights
    class_weights = compute_smoothed_class_weights(
        manifest_path=train_manifest,
        num_classes=101,
        smoothing=0.5,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    # Model
    model = DilatedTemporalCNN(
        input_dim=678,
        num_classes=101,
        base_channels=128,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dilated Temporal CNN initialized ({total_params:,} parameters)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_top1 = 0.0
    best_metrics = None
    best_epoch = 0
    patience = 12
    patience_counter = 0

    t0_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits = model(features, attention_mask=mask)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * targets.size(0)

        scheduler.step()
        avg_train_loss = train_loss / len(train_ds)

        val_metrics = evaluate_temporal_cnn(model, val_loader, criterion, device)
        v_loss = val_metrics["loss"]
        v_top1 = val_metrics["top1_accuracy"]
        v_top3 = val_metrics["top3_accuracy"]
        v_f1 = val_metrics["macro_f1"]
        v_pred_cnt = val_metrics["unique_predicted_classes"]

        if epoch % 5 == 0 or epoch == 1 or v_loss < best_val_loss:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.3f} | "
                f"Val Loss: {v_loss:.3f} | Top-1: {v_top1*100:5.2f}% | Top-3: {v_top3*100:5.2f}% | "
                f"F1: {v_f1:6.4f} | Preds: {v_pred_cnt:2d}/85"
            )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_top1 = v_top1
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": v_loss,
                    "val_top1": v_top1,
                    "val_top3": v_top3,
                    "val_f1": v_f1,
                },
                models_dir / "best_temporal_cnn_checkpoint.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
                break

    total_train_time = time.time() - t0_train
    best_metrics["train_time_sec"] = total_train_time

    results = {
        "experiment": "Experiment D: Temporal CNN",
        "description": "Dilated 1D Residual TCN with Masked Temporal Pooling",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_top1": best_val_top1,
        "total_train_time_sec": total_train_time,
        "best_validation_metrics": best_metrics,
    }

    eval_file = models_dir / "eval_exp_d_temporal_cnn.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment D completed! Best Epoch {best_epoch} (Val Loss: {best_val_loss:.3f}, Top-1: {best_val_top1*100:.2f}%)")
    print(f"Results saved to {eval_file}")


if __name__ == "__main__":
    main()
