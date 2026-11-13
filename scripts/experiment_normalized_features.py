"""
Phase 8.3.6 Experiment 1: Controlled Accuracy Improvement via Spatial Feature Normalization.

Hypothesis:
Raw MediaPipe landmarks contain significant signer-position and resolution variance across
the 5 train signers, Signer 7 (Val, FHD), and Signer 2 (Test, QVGA).
Normalizing landmarks by centering on the shoulder midpoint and scaling by inter-shoulder distance
will reduce cross-signer distribution shift and alleviate CTC mode collapse.

Controlled Factor:
- Changed: Landmark spatial normalization (shoulder-centered, scale-normalized)
- Kept constant: Model architecture, BiGRU layers, learning rate (3e-4), optimizer, batch size (16), patience (15)
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.isl_csltr_dataset import collate_isl_csltr_batch
from ai.continuous.isl_csltr_model import (
    ISLCSLTRContinuousModel,
    compute_sequence_metrics,
    decode_greedy_ctc,
)


def normalize_landmarks_tensor(landmarks: np.ndarray) -> np.ndarray:
    """Vectorized normalization (T, 75, 4): center on shoulder midpoint and scale by shoulder distance."""
    T, num_pts, channels = landmarks.shape

    # Extract shoulder positions (T, 1, 2)
    left_sh = landmarks[:, 11:12, :2]
    right_sh = landmarks[:, 12:13, :2]
    sh_mid = (left_sh + right_sh) / 2.0  # (T, 1, 2)

    sh_dist = np.linalg.norm(left_sh - right_sh, axis=-1, keepdims=True)  # (T, 1, 1)
    valid_dist = np.where(sh_dist > 0.01, sh_dist, 0.2)  # (T, 1, 1)

    normed_xy = (landmarks[:, :, :2] - sh_mid) / valid_dist
    normed_z = landmarks[:, :, 2:3] / valid_dist
    conf = landmarks[:, :, 3:4]

    normed = np.concatenate([normed_xy, normed_z, conf], axis=-1)
    return normed.reshape(T, -1).astype(np.float32)


class NormalizedISLDataset(Dataset):
    """Dataset with online shoulder-normalization."""

    def __init__(self, manifest_path: Path, features_root: Path, canonical_only: bool = True):
        with open(manifest_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if canonical_only:
            self.records = [r for r in records if r.get("is_canonical", True)]
        else:
            self.records = records
        self.features_root = features_root

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        npz_name = Path(record["npz_relative_path"]).name
        npz_path = self.features_root / npz_name

        data = np.load(npz_path)
        lm = data["landmarks"]  # (T, 75, 4)
        norm_feat = normalize_landmarks_tensor(lm)  # (T, 300)

        return {
            "video_id": record["video_id"],
            "features": torch.from_numpy(norm_feat),
            "length": norm_feat.shape[0],
            "token_ids": torch.tensor(record["token_ids"], dtype=torch.long),
            "tokens": record["tokens"],
            "signer_id": record["signer_id"],
            "sentence": record["sentence"],
            "resolution": record.get("resolution", "unknown"),
        }


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cpu")
    dataset_root = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"
    vocab_path = dataset_root / "metadata" / "vocabulary.json"

    with open(vocab_path, "r", encoding="utf-8") as f:
        vdata = json.load(f)
    id_to_gloss = {int(v): k for k, v in vdata["gloss_to_id"].items()}
    num_classes = vdata["num_classes"]

    train_ds = NormalizedISLDataset(
        dataset_root / "manifests" / "train_manifest.json",
        dataset_root / "features" / "train",
        canonical_only=True,
    )
    val_ds = NormalizedISLDataset(
        dataset_root / "manifests" / "validation_manifest.json",
        dataset_root / "features" / "validation",
        canonical_only=True,
    )
    test_ds = NormalizedISLDataset(
        dataset_root / "manifests" / "test_manifest.json",
        dataset_root / "features" / "test",
        canonical_only=True,
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_isl_csltr_batch)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_isl_csltr_batch)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_isl_csltr_batch)

    model = ISLCSLTRContinuousModel(
        input_dim=300, proj_dim=128, hidden_dim=128, num_layers=2, num_classes=num_classes, dropout=0.3
    ).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    best_val_wer = float("inf")
    best_epoch = 0
    patience = 15
    patience_counter = 0
    history = []

    print("Running Controlled Experiment 1: Shoulder-Normalized Features (max 50 epochs)...")
    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            x, in_lens = batch["features"], batch["input_lengths"]
            targets, tgt_lens = batch["targets"], batch["target_lengths"]

            optimizer.zero_grad()
            log_probs, sub_lens = model(x, in_lens)
            loss = ctc_loss_fn(log_probs, targets, sub_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = x.size(0)
            train_loss += loss.item() * bs
            train_samples += bs

        scheduler.step()
        avg_train_loss = train_loss / train_samples

        # Validation
        model.eval()
        val_refs, val_hyps = [], []
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x, in_lens = batch["features"], batch["input_lengths"]
                targets, tgt_lens = batch["targets"], batch["target_lengths"]
                log_probs, sub_lens = model(x, in_lens)
                loss = ctc_loss_fn(log_probs, targets, sub_lens, tgt_lens)
                bs = x.size(0)
                val_loss += loss.item() * bs
                val_samples += bs

                decoded = decode_greedy_ctc(log_probs, sub_lens, blank_idx=0, id_to_gloss=id_to_gloss)
                for i, dec in enumerate(decoded):
                    val_refs.append(batch["tokens_list"][i])
                    val_hyps.append(dec["tokens"])

        val_metrics = compute_sequence_metrics(val_refs, val_hyps)
        avg_val_loss = val_loss / val_samples
        val_wer = val_metrics["wer"]

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_wer": val_wer,
            "val_exact_match": val_metrics["exact_match_rate"],
        })

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict().copy()
        elif val_wer == 1.0 and avg_val_loss < 4.8:
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or val_wer < best_val_wer:
            print(f"Epoch {epoch:2d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Val WER: {val_wer:.2%}", flush=True)

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val WER: {best_val_wer:.2%} at epoch {best_epoch}", flush=True)
            break

    # Final test evaluation for Experiment 1
    model.load_state_dict(best_state)
    model.eval()
    test_refs, test_hyps = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, in_lens = batch["features"], batch["input_lengths"]
            log_probs, sub_lens = model(x, in_lens)
            decoded = decode_greedy_ctc(log_probs, sub_lens, blank_idx=0, id_to_gloss=id_to_gloss)
            for i, dec in enumerate(decoded):
                test_refs.append(batch["tokens_list"][i])
                test_hyps.append(dec["tokens"])

    test_metrics = compute_sequence_metrics(test_refs, test_hyps)
    print(f"\nExperiment 1 Results:")
    print(f"Best Val WER: {best_val_wer:.2%} (vs Baseline 82.57%)")
    print(f"Test WER:     {test_metrics['wer']:.2%} (vs Baseline 83.09%)")

    exp_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    with open(exp_dir / "experiment_1_normalization_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Experiment 1: Shoulder-Centered Spatial Normalization",
            "best_epoch": best_epoch,
            "best_val_wer": best_val_wer,
            "test_wer": test_metrics["wer"],
            "test_exact_match": test_metrics["exact_match_rate"],
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
