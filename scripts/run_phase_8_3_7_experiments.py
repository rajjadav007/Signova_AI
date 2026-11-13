"""
Phase 8.3.7: CTC Training Stability and Class Imbalance Optimization.

Implements and evaluates:
- Experiment 1: Class-Balanced Sampling (WeightedRandomSampler with smoothed inverse frequency).
- Experiment 2: CTC Blank Penalty (Logit bias subtraction during training: beta in {0.5, 1.0, 2.0}).
- Experiment 3: Sequence-Level Class-Balanced CTC (reduction='none' with sample-level rare-gloss weighting).

Strict Safety:
- Preserves Phase 8.3.6 baseline checkpoint and results.
- Preserves splits, architecture, vocabulary, canonical filtering.
- Validation-based model selection; test set evaluated only once on selected model.
- All artifacts saved to models/isl_csltr_continuous/phase_8_3_7/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.isl_csltr_dataset import ISLCSLTRDataset, collate_isl_csltr_batch
from ai.continuous.isl_csltr_model import (
    ISLCSLTRContinuousModel,
    compute_sequence_metrics,
    compute_subsampled_lengths,
    decode_greedy_ctc,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ISLCSLTRModelWithBlankPenalty(ISLCSLTRContinuousModel):
    """Subclass adding configurable blank logit bias offset."""

    def __init__(self, blank_penalty: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.blank_penalty = float(blank_penalty)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=x.device)

        # 1. Feature Projection
        h = self.proj(x)
        h = self.proj_ln(h)
        h = self.proj_dropout(h)
        h = self.proj_relu(h)

        # 2. Temporal Subsampling
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)
        h = self.conv_ln(h)
        h = self.conv_relu(h)

        subsampled_lengths = compute_subsampled_lengths(lengths).to(x.device)

        # 3. Recurrent Encoding
        if subsampled_lengths.min() != subsampled_lengths.max():
            packed = nn.utils.rnn.pack_padded_sequence(
                h, subsampled_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(h)

        # 4. Classifier Head
        out = self.head_dropout(out)
        logits = self.classifier(out)  # (B, T', num_classes)

        # Apply blank penalty if non-zero
        if self.blank_penalty != 0.0:
            blank_col = logits[:, :, :1] - self.blank_penalty
            logits = torch.cat([blank_col, logits[:, :, 1:]], dim=-1)

        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, subsampled_lengths


def evaluate_split_extended(
    model: nn.Module,
    dataloader: DataLoader,
    ctc_loss_fn: nn.CTCLoss,
    id_to_gloss: Dict[int, str],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model with full sequence metrics, blank posteriors, and length analysis."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_references: List[List[str]] = []
    all_hypotheses: List[List[str]] = []
    sample_records: List[Dict[str, Any]] = []
    blank_percentages: List[float] = []
    all_blank_posteriors: List[float] = []

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

            probs = torch.exp(log_probs).transpose(0, 1)  # (B, T', num_classes)
            for i in range(bs):
                slen = int(sub_lens[i].item())
                sample_blank_probs = probs[i, :slen, 0].cpu().numpy()
                all_blank_posteriors.extend(sample_blank_probs.tolist())

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
    metrics["mean_blank_posterior"] = round(float(np.mean(all_blank_posteriors)), 4) if all_blank_posteriors else 0.0

    ref_lens = [len(r) for r in all_references]
    hyp_lens = [len(h) for h in all_hypotheses]
    metrics["mean_ref_length"] = round(float(np.mean(ref_lens)), 2) if ref_lens else 0.0
    metrics["mean_hyp_length"] = round(float(np.mean(hyp_lens)), 2) if hyp_lens else 0.0
    metrics["distinct_glosses_predicted"] = len(set(g for h in all_hypotheses for g in h))
    metrics["predicted_gloss_counts"] = dict(Counter(g for h in all_hypotheses for g in h).most_common(15))
    metrics["sample_records"] = sample_records

    return metrics


def compute_sample_weights(
    records: List[Dict[str, Any]],
    gloss_counts: Counter,
    max_multiplier: float = 5.0,
) -> Tuple[List[float], Dict[str, Any]]:
    """Compute smoothed inverse-frequency weights for WeightedRandomSampler."""
    weights = []
    for r in records:
        toks = r["tokens"]
        if not toks:
            weights.append(1.0)
            continue
        # Smoothed inverse frequency: sum(1 / sqrt(freq))
        w = sum(1.0 / np.sqrt(max(gloss_counts[g], 1)) for g in toks)
        weights.append(w)

    weights_np = np.array(weights, dtype=np.float64)
    mean_w = np.mean(weights_np)
    clamped_weights = np.clip(weights_np, a_min=None, a_max=max_multiplier * mean_w)

    # Normalize weights so mean is 1.0
    norm_weights = clamped_weights / np.mean(clamped_weights)

    # Calculate expected sampling counts per gloss
    exp_gloss_counts: Dict[str, float] = defaultdict(float)
    total_w = np.sum(norm_weights)
    for r, nw in zip(records, norm_weights):
        prob = nw / total_w
        for g in r["tokens"]:
            exp_gloss_counts[g] += prob * len(records)

    sampling_stats = {
        "raw_weight_min": float(np.min(weights_np)),
        "raw_weight_mean": float(np.mean(weights_np)),
        "raw_weight_max": float(np.max(weights_np)),
        "clamped_weight_max": float(np.max(clamped_weights)),
        "top_5_expected_glosses": sorted(exp_gloss_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        "bottom_5_expected_glosses": sorted(exp_gloss_counts.items(), key=lambda x: x[1])[:5],
    }

    return norm_weights.tolist(), sampling_stats


def train_single_experiment(
    experiment_name: str,
    exp_type: str,  # "balanced_sampling", "blank_penalty", "class_balanced_ctc"
    output_dir: Path,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    blank_penalty: float = 0.0,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Starting Experiment: {experiment_name} (type: {exp_type})")
    print(f"Params: blank_penalty={blank_penalty}, patience={patience}, seed={seed}")
    print(f"=======================================================\n")

    set_seed(seed)
    device = torch.device("cpu")
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Ponytail: Reuse cached completed experiment results
    cached_res = exp_dir / "validation_results.json"
    if cached_res.exists():
        print(f"Found completed results for {experiment_name}. Loading from disk.")
        with open(cached_res, "r", encoding="utf-8") as f:
            return json.load(f)

    dataset_root = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"
    train_manifest = dataset_root / "manifests" / "train_manifest.json"
    val_manifest = dataset_root / "manifests" / "validation_manifest.json"
    vocab_path = dataset_root / "metadata" / "vocabulary.json"

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

    id_to_gloss = train_dataset.id_to_gloss
    num_classes = train_dataset.num_classes

    # Compute training gloss counts
    train_gloss_counter = Counter()
    for r in train_dataset.records:
        for t in r["tokens"]:
            train_gloss_counter[t] += 1

    # Configure DataLoader according to experiment type
    sample_weights_list = None
    sampling_stats = {}
    if exp_type == "balanced_sampling":
        sample_weights, sampling_stats = compute_sample_weights(
            train_dataset.records, train_gloss_counter, max_multiplier=5.0
        )
        sample_weights_list = sample_weights
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_isl_csltr_batch,
        )
    else:
        # Precompute per-sample weights if needed for class_balanced_ctc
        if exp_type == "class_balanced_ctc":
            sample_weights, sampling_stats = compute_sample_weights(
                train_dataset.records, train_gloss_counter, max_multiplier=5.0
            )
            sample_weights_list = sample_weights
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

    # Initialize model
    model = ISLCSLTRModelWithBlankPenalty(
        blank_penalty=blank_penalty,
        input_dim=300,
        proj_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    # Loss setup
    if exp_type == "class_balanced_ctc":
        # Per-sample loss with sequence weighting
        ctc_loss_none = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
        # Map video_id to sample weight
        vid_to_weight = {
            r["video_id"]: w for r, w in zip(train_dataset.records, sample_weights_list)
        }
    else:
        ctc_loss_none = None

    eval_ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_wer = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    epoch_history: List[Dict[str, Any]] = []

    start_time = time.time()
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>9} | {'Val WER':>8} | {'Exact M':>8} | {'Blank %':>8} | {'Mean P(B)':>9} | {'Distinct':>8} | {'Time':>6}")
    print("-" * 92)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            video_ids = batch["video_ids"]

            optimizer.zero_grad()
            log_probs, sub_lens = model(features, input_lengths)

            if exp_type == "class_balanced_ctc":
                # Compute unreduced sample losses
                sample_losses = ctc_loss_none(log_probs, targets, sub_lens, target_lengths)
                batch_weights = torch.tensor(
                    [vid_to_weight[vid] for vid in video_ids],
                    dtype=torch.float32,
                    device=device,
                )
                loss = torch.sum(batch_weights * sample_losses) / torch.sum(batch_weights)
            else:
                loss = eval_ctc_loss_fn(log_probs, targets, sub_lens, target_lengths)

            # Detect NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss encountered at epoch {epoch}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = features.size(0)
            train_loss += loss.item() * bs
            train_samples += bs

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0

        # Evaluate on validation split
        val_metrics = evaluate_split_extended(model, val_loader, eval_ctc_loss_fn, id_to_gloss, device)
        val_loss = val_metrics["ctc_loss"]
        val_wer = val_metrics["wer"]
        val_em = val_metrics["exact_match_rate"]
        val_blank_pct = val_metrics["mean_blank_percentage"]
        val_blank_post = val_metrics["mean_blank_posterior"]
        val_distinct = val_metrics["distinct_glosses_predicted"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_ter": val_metrics["ter"],
            "val_exact_match_rate": val_em,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_blank_percentage": val_blank_pct,
            "val_blank_posterior": val_blank_post,
            "val_distinct_glosses": val_distinct,
            "lr": round(scheduler.get_last_lr()[0], 6),
            "epoch_time_seconds": round(epoch_time, 2),
        }
        epoch_history.append(epoch_record)

        is_best = False
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            is_best = True

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_wer": val_wer,
                "val_loss": val_loss,
                "blank_penalty": blank_penalty,
                "config": {
                    "experiment_name": experiment_name,
                    "exp_type": exp_type,
                    "blank_penalty": blank_penalty,
                    "num_classes": num_classes,
                },
            }, exp_dir / "best_checkpoint.pt")
        elif val_wer == 1.0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if best_val_wer == float("inf") or best_val_wer == 1.0:
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_wer": val_wer,
                    "val_loss": val_loss,
                    "blank_penalty": blank_penalty,
                    "config": {
                        "experiment_name": experiment_name,
                        "exp_type": exp_type,
                        "blank_penalty": blank_penalty,
                        "num_classes": num_classes,
                    },
                }, exp_dir / "best_checkpoint.pt")
        else:
            patience_counter += 1

        best_flag = " *" if is_best else ""
        print(f"{epoch:5d} | {avg_train_loss:10.4f} | {val_loss:9.4f} | {val_wer:7.2%} | {val_em:7.2%} | {val_blank_pct:7.1f}% | {val_blank_post:9.4f} | {val_distinct:8d} | {epoch_time:5.1f}s{best_flag}")

        if patience_counter >= patience:
            print(f"\n[Early Stopping] No improvement in validation WER for {patience} epochs. Stopped at epoch {epoch}.")
            break

    total_training_time = time.time() - start_time

    # Save epoch history
    with open(exp_dir / "epoch_history.json", "w", encoding="utf-8") as f:
        json.dump(epoch_history, f, indent=2)

    # Load best checkpoint and evaluate on validation
    best_ckpt = torch.load(exp_dir / "best_checkpoint.pt")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_final_eval = evaluate_split_extended(model, val_loader, eval_ctc_loss_fn, id_to_gloss, device)

    experiment_results = {
        "experiment_name": experiment_name,
        "exp_type": exp_type,
        "blank_penalty": blank_penalty,
        "best_epoch": best_epoch,
        "best_val_wer": best_val_wer,
        "total_training_time_seconds": round(total_training_time, 2),
        "sampling_stats": sampling_stats,
        "validation_evaluation": val_final_eval,
    }

    with open(exp_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\n--- Experiment {experiment_name} Summary ---")
    print(f"Best Epoch: {best_epoch} | Time: {total_training_time:.1f}s")
    print(f"Validation Loss: {val_final_eval['ctc_loss']}")
    print(f"Validation WER:  {val_final_eval['wer']:.2%}")
    print(f"Validation TER:  {val_final_eval['ter']:.2%}")
    print(f"Validation Exact Match: {val_final_eval['exact_match_rate']:.2%}")
    print(f"Validation Macro F1:    {val_final_eval['macro_f1']:.4f}")
    print(f"Mean Blank Pct:  {val_final_eval['mean_blank_percentage']:.1f}%")
    print(f"Mean Blank Post: {val_final_eval['mean_blank_posterior']:.4f}")
    print(f"Distinct Glosses: {val_final_eval['distinct_glosses_predicted']}")
    print(f"Top Predictions: {val_final_eval['predicted_gloss_counts']}")

    return experiment_results


def evaluate_test_set_on_selected_model(
    selected_checkpoint_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate untouched test set (Signer 2) once on selected best model."""
    print(f"\n=======================================================")
    print(f"Evaluating Untouched Test Set on Selected Checkpoint:")
    print(f"{selected_checkpoint_path}")
    print(f"=======================================================\n")

    device = torch.device("cpu")
    dataset_root = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"
    test_manifest = dataset_root / "manifests" / "test_manifest.json"
    vocab_path = dataset_root / "metadata" / "vocabulary.json"

    test_dataset = ISLCSLTRDataset(
        manifest_path=test_manifest,
        features_root=dataset_root / "features" / "test",
        vocab_path=vocab_path,
        canonical_only=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_isl_csltr_batch,
    )

    ckpt = torch.load(selected_checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    blank_penalty = ckpt.get("blank_penalty", cfg.get("blank_penalty", 0.0))
    num_classes = cfg.get("num_classes", 171)

    model = ISLCSLTRModelWithBlankPenalty(
        blank_penalty=blank_penalty,
        input_dim=300,
        proj_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    test_eval = evaluate_split_extended(
        model, test_loader, ctc_loss_fn, test_dataset.id_to_gloss, device
    )

    print("\n--- Final Test Split Evaluation Results (Signer 2, Unseen Native Deaf) ---")
    print(f"Test CTC Loss: {test_eval['ctc_loss']}")
    print(f"Test WER:      {test_eval['wer']:.2%}")
    print(f"Test TER:      {test_eval['ter']:.2%}")
    print(f"Test Exact Match: {test_eval['exact_match_rate']:.2%}")
    print(f"Test Macro F1:    {test_eval['macro_f1']:.4f}")
    print(f"Mean Blank Pct:   {test_eval['mean_blank_percentage']:.1f}%")
    print(f"Mean Blank Post:  {test_eval['mean_blank_posterior']:.4f}")
    print(f"Distinct Glosses: {test_eval['distinct_glosses_predicted']}")
    print(f"Top Predictions:  {test_eval['predicted_gloss_counts']}")

    test_results_path = output_dir / "final_selected_test_evaluation.json"
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump({
            "selected_checkpoint": str(selected_checkpoint_path),
            "test_evaluation": test_eval,
        }, f, indent=2)

    print(f"Test evaluation saved to {test_results_path}")
    return test_eval


def main():
    parser = argparse.ArgumentParser(description="Phase 8.3.7 Experiments Runner")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch smoke test across experiments")
    parser.add_argument("--run_all", action="store_true", help="Run all 5 experiments sequentially")
    parser.add_argument("--exp", type=str, default="", help="Run specific experiment name")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "phase_8_3_7"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print("Running smoke tests (1 epoch each)...")
        train_single_experiment("smoke_exp1", "balanced_sampling", out_dir, epochs=1, patience=1)
        train_single_experiment("smoke_exp2", "blank_penalty", out_dir, epochs=1, patience=1, blank_penalty=1.0)
        train_single_experiment("smoke_exp3", "class_balanced_ctc", out_dir, epochs=1, patience=1)
        print("Smoke tests passed!")
        return

    # List of planned experiments:
    experiments = [
        # Exp 1: Class-Balanced Sampling
        ("exp1_balanced_sampling", "balanced_sampling", 0.0),
        # Exp 2: CTC Blank Penalty across conservative strengths: 0.5, 1.0, 2.0
        ("exp2_blank_penalty_0_5", "blank_penalty", 0.5),
        ("exp2_blank_penalty_1_0", "blank_penalty", 1.0),
        ("exp2_blank_penalty_2_0", "blank_penalty", 2.0),
        # Exp 3: Sequence-Level Class-Balanced CTC
        ("exp3_sequence_class_balanced_ctc", "class_balanced_ctc", 0.0),
    ]

    if args.exp:
        exp_match = [e for e in experiments if e[0] == args.exp]
        if not exp_match:
            print(f"Unknown experiment: {args.exp}")
            return
        experiments = exp_match

    results_summary = {}
    for exp_name, exp_type, penalty in experiments:
        res = train_single_experiment(
            experiment_name=exp_name,
            exp_type=exp_type,
            output_dir=out_dir,
            epochs=args.epochs,
            patience=args.patience,
            blank_penalty=penalty,
        )
        results_summary[exp_name] = {
            "exp_type": exp_type,
            "blank_penalty": penalty,
            "best_epoch": res["best_epoch"],
            "val_loss": res["validation_evaluation"]["ctc_loss"],
            "val_wer": res["validation_evaluation"]["wer"],
            "val_ter": res["validation_evaluation"]["ter"],
            "val_exact_match": res["validation_evaluation"]["exact_match_rate"],
            "val_macro_f1": res["validation_evaluation"]["macro_f1"],
            "mean_blank_pct": res["validation_evaluation"]["mean_blank_percentage"],
            "mean_blank_post": res["validation_evaluation"]["mean_blank_posterior"],
            "distinct_glosses": res["validation_evaluation"]["distinct_glosses_predicted"],
            "top_predictions": res["validation_evaluation"]["predicted_gloss_counts"],
        }

    # Save validation comparison summary
    with open(out_dir / "all_experiments_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # Compare against Phase 8.3.6 baseline
    baseline_wer = 0.8257
    baseline_f1 = 0.0053
    print("\n=======================================================")
    print("ALL EXPERIMENTS VALIDATION COMPARISON")
    print("=======================================================")
    print(f"{'Experiment':<32} | {'Val WER':>8} | {'Val F1':>8} | {'Exact M':>8} | {'Blank %':>8} | {'Blank Post':>10} | {'Distinct':>8}")
    print("-" * 96)
    print(f"{'Phase 8.3.6 Baseline':<32} | {baseline_wer:7.2%} | {baseline_f1:8.4f} | {'0.00%':>8} | {'96.3%':>8} | {'0.9468':>10} | {2:>8}")

    best_exp_name = None
    best_wer = float("inf")
    best_f1 = -1.0

    for name, r in results_summary.items():
        print(f"{name:<32} | {r['val_wer']:7.2%} | {r['val_macro_f1']:8.4f} | {r['val_exact_match']:7.2%} | {r['mean_blank_pct']:7.1f}% | {r['mean_blank_post']:10.4f} | {r['distinct_glosses']:8d}")
        # Selection criterion: primary is Validation WER, secondary is Macro F1 and distinct glosses
        if r["val_wer"] < best_wer or (abs(r["val_wer"] - best_wer) < 1e-4 and r["val_macro_f1"] > best_f1):
            best_wer = r["val_wer"]
            best_f1 = r["val_macro_f1"]
            best_exp_name = name

    print(f"\nBest Performing Experiment on Validation Split: {best_exp_name} (Val WER: {best_wer:.2%}, F1: {best_f1:.4f})")

    # Evaluate untouched test set on best experiment checkpoint
    selected_ckpt = out_dir / best_exp_name / "best_checkpoint.pt"
    evaluate_test_set_on_selected_model(selected_ckpt, out_dir)


if __name__ == "__main__":
    main()
