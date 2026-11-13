"""
Training and Evaluation Pipeline for Phase 8.3.2: Bi-GRU + CTC Continuous ISL Baseline.

Executes:
1. Deterministic seeded training on extracted Phase 8.2 features.
2. AdamW optimizer + Cosine Annealing + Gradient Clipping.
3. Loss tracking and checkpoint persistence.
4. Comprehensive test evaluation:
   - Test CTC loss.
   - Exact sequence match accuracy (by recording and by sentence).
   - Token error rate (TER) and token-level accuracy.
   - Per-class precision, recall, and support.
   - Blank prediction percentage and blank-only collapse verification.
   - Decoded vs reference sequence comparison for every test recording.
   - Camera view breakdown (front vs side).
   - Front-only training ablation.

Safeguards strictly enforced:
- Zero test data used for early stopping or tuning.
- Documented absence of independent validation split.
- Sentence IDs tracked separately from Recording IDs.
- Paired camera views kept together in splits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai.continuous.bigru_ctc import (
    BiGRUCTCModel,
    ContinuousISLDataset,
    collate_continuous_fn,
    compute_levenshtein,
    decode_greedy,
    BLANK_INDEX,
    GLOSS_TO_ID,
    ID_TO_GLOSS,
    NUM_CLASSES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("islvt_bigru_ctc")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets" / "islvt_continuous"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "islvt_continuous"


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one epoch of training."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        feature_lengths = batch["feature_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        log_probs, sub_lengths = model(features, feature_lengths)
        loss = criterion(log_probs, targets, sub_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss encountered, skipping batch")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / total_batches if total_batches > 0 else 0.0


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> Dict[str, Any]:
    """Run thorough evaluation on dataset split."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    all_decoded_records: List[Dict[str, Any]] = []
    total_target_tokens = 0
    total_pred_tokens = 0
    total_correct_tokens = 0
    total_edit_distance = 0
    exact_matches_recording = 0
    total_recordings = 0

    total_timesteps = 0
    total_blank_timesteps = 0

    # Per-class counters: {class_name: {"tp": int, "fp": int, "fn": int, "support": int}}
    per_class_stats = {
        name: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for name in ["i", "go", "eat", "food"]
    }

    # Sentence-level tracking: {sentence_id: List[bool]} (whether each take was exact match)
    sentence_matches: Dict[int, List[bool]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            feature_lengths = batch["feature_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            log_probs, sub_lengths = model(features, feature_lengths)
            loss = criterion(log_probs, targets, sub_lengths, target_lengths)
            total_loss += loss.item()
            total_batches += 1

            decoded_batch = decode_greedy(log_probs, sub_lengths, blank_idx=BLANK_INDEX)

            for item, dec in zip(batch["items"], decoded_batch):
                ref_tokens = item["target_glosses"]
                pred_tokens = dec["tokens"]

                edit_dist = compute_levenshtein(pred_tokens, ref_tokens)
                total_edit_distance += edit_dist
                is_exact = (pred_tokens == ref_tokens)
                if is_exact:
                    exact_matches_recording += 1
                total_recordings += 1

                sentence_matches[item["sentence_id"]].append(is_exact)

                # Blank tracking
                t_frames = dec["total_frames"]
                total_timesteps += t_frames
                blank_count = sum(1 for p in dec["raw_preds"] if p == BLANK_INDEX)
                total_blank_timesteps += blank_count

                # Token matching (greedy alignment for per-class tracking)
                total_target_tokens += len(ref_tokens)
                total_pred_tokens += len(pred_tokens)

                # Count class supports
                for t in ref_tokens:
                    if t in per_class_stats:
                        per_class_stats[t]["support"] += 1

                # Token accuracy via multiset intersection
                ref_counts = defaultdict(int)
                for t in ref_tokens:
                    ref_counts[t] += 1
                pred_counts = defaultdict(int)
                for p in pred_tokens:
                    pred_counts[p] += 1

                for cls_name in per_class_stats:
                    c_ref = ref_counts[cls_name]
                    c_pred = pred_counts[cls_name]
                    tp = min(c_ref, c_pred)
                    fp = max(0, c_pred - c_ref)
                    fn = max(0, c_ref - c_pred)
                    per_class_stats[cls_name]["tp"] += tp
                    per_class_stats[cls_name]["fp"] += fp
                    per_class_stats[cls_name]["fn"] += fn
                    total_correct_tokens += tp

                all_decoded_records.append({
                    "sentence_id": item["sentence_id"],
                    "video_id": item["video_id"],
                    "camera_view": item["camera_view"],
                    "split": item["split"],
                    "isl_gloss": item["isl_gloss"],
                    "reference": ref_tokens,
                    "prediction": pred_tokens,
                    "raw_predictions": dec["raw_preds"],
                    "exact_match": is_exact,
                    "edit_distance": edit_dist,
                    "blank_percentage": dec["blank_percentage"],
                    "subsampled_timesteps": t_frames,
                })

    mean_loss = total_loss / total_batches if total_batches > 0 else 0.0
    rec_acc = (exact_matches_recording / total_recordings * 100.0) if total_recordings > 0 else 0.0

    # Sentence-level accuracy: sentence is fully correct if all camera takes match reference
    # and relaxed sentence accuracy: at least one take matches
    all_takes_correct = sum(1 for s_id, matches in sentence_matches.items() if all(matches))
    any_take_correct = sum(1 for s_id, matches in sentence_matches.items() if any(matches))
    total_sentences = len(sentence_matches)
    sent_acc_strict = (all_takes_correct / total_sentences * 100.0) if total_sentences > 0 else 0.0
    sent_acc_relaxed = (any_take_correct / total_sentences * 100.0) if total_sentences > 0 else 0.0

    # Token Error Rate (TER) = total_edit_distance / total_target_tokens
    ter = (total_edit_distance / total_target_tokens * 100.0) if total_target_tokens > 0 else 0.0
    token_acc = (total_correct_tokens / total_target_tokens * 100.0) if total_target_tokens > 0 else 0.0

    # Blank percentage across all timesteps
    overall_blank_pct = (total_blank_timesteps / total_timesteps * 100.0) if total_timesteps > 0 else 0.0
    is_blank_collapse = (overall_blank_pct >= 99.9) or (total_pred_tokens == 0)

    # Per-class metrics
    per_class_summary: Dict[str, Dict[str, float]] = {}
    for cls_name, counts in per_class_stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        supp = counts["support"]
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_summary[cls_name] = {
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "support": supp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "loss": round(mean_loss, 4),
        "total_recordings": total_recordings,
        "total_sentences": total_sentences,
        "exact_match_recording_acc": round(rec_acc, 2),
        "exact_match_sentence_strict_acc": round(sent_acc_strict, 2),
        "exact_match_sentence_relaxed_acc": round(sent_acc_relaxed, 2),
        "token_error_rate": round(ter, 2),
        "token_accuracy": round(token_acc, 2),
        "blank_percentage": round(overall_blank_pct, 2),
        "is_blank_collapse": is_blank_collapse,
        "per_class": per_class_summary,
        "records": all_decoded_records,
    }


def run_experiment(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    seed: int = 42,
    device_str: str = "cpu",
    train_camera_view: str = "all",
) -> Dict[str, Any]:
    """Train baseline Bi-GRU+CTC and evaluate thoroughly."""
    set_seed(seed)
    device = torch.device(device_str)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing datasets (train_camera_view=%s)...", train_camera_view)
    train_dataset = ContinuousISLDataset(dataset_dir, split="train", camera_view=train_camera_view)
    test_dataset_all = ContinuousISLDataset(dataset_dir, split="test", camera_view="all")
    test_dataset_front = ContinuousISLDataset(dataset_dir, split="test", camera_view="front")
    test_dataset_side = ContinuousISLDataset(dataset_dir, split="test", camera_view="side")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_continuous_fn,
    )
    test_loader_all = DataLoader(
        test_dataset_all,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_continuous_fn,
    )
    test_loader_front = DataLoader(
        test_dataset_front,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_continuous_fn,
    )
    test_loader_side = DataLoader(
        test_dataset_side,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_continuous_fn,
    )

    logger.info(
        "Train set: %d recordings | Test set: %d recordings (Front: %d, Side: %d)",
        len(train_dataset),
        len(test_dataset_all),
        len(test_dataset_front),
        len(test_dataset_side),
    )

    model = BiGRUCTCModel(
        input_dim=300,
        proj_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model instantiated: %d trainable parameters", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    loss_history: List[float] = []

    logger.info("Starting training for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip=grad_clip)
        scheduler.step()
        loss_history.append(epoch_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info("Epoch %3d/%3d | CTC Train Loss: %.4f", epoch, epochs, epoch_loss)

    # Save model checkpoint
    checkpoint_name = (
        "bigru_ctc_baseline.pt" if train_camera_view == "all" else f"bigru_ctc_baseline_{train_camera_view}.pt"
    )
    checkpoint_path = model_dir / checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": 300,
                "proj_dim": 128,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_classes": NUM_CLASSES,
                "dropout": 0.3,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "seed": seed,
                "train_camera_view": train_camera_view,
            },
            "vocab": GLOSS_TO_ID,
            "final_train_loss": loss_history[-1],
            "loss_history": loss_history,
        },
        checkpoint_path,
    )
    logger.info("Saved model checkpoint to %s", checkpoint_path)

    # Evaluate on test sets
    logger.info("Evaluating on Test Set (All Views: 8 recordings)...")
    eval_all = evaluate(model, test_loader_all, criterion, device)

    logger.info("Evaluating on Test Set (Front View: 4 recordings)...")
    eval_front = evaluate(model, test_loader_front, criterion, device)

    logger.info("Evaluating on Test Set (Side View: 4 recordings)...")
    eval_side = evaluate(model, test_loader_side, criterion, device)

    results = {
        "run_config": {
            "train_camera_view": train_camera_view,
            "train_recordings": len(train_dataset),
            "test_recordings": len(test_dataset_all),
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "seed": seed,
            "total_parameters": total_params,
            "device": str(device),
            "checkpoint_path": str(checkpoint_path),
        },
        "training": {
            "initial_loss": round(loss_history[0], 4),
            "final_loss": round(loss_history[-1], 4),
            "loss_history": [round(l, 4) for l in loss_history],
            "validation_loss": None,
            "validation_note": "No independent validation split exists in dataset. Test set was not used for validation or early stopping.",
        },
        "evaluation_all": eval_all,
        "evaluation_front": eval_front,
        "evaluation_side": eval_side,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Bi-GRU+CTC ISLVT baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ablation", action="store_true", help="Also train front-only ablation")
    args = parser.parse_args()

    logger.info("=== Primary Experiment: Train on Both Views (24 recordings) ===")
    primary_results = run_experiment(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        train_camera_view="all",
    )

    ablation_results = None
    if args.run_ablation:
        logger.info("=== Ablation Experiment: Train on Front-Only (12 recordings) ===")
        ablation_results = run_experiment(
            dataset_dir=args.dataset_dir,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
            train_camera_view="front",
        )

    output_payload = {
        "primary_run": primary_results,
        "ablation_front_only": ablation_results,
    }

    results_path = args.dataset_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Results saved to %s", results_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 8.3.2 BASELINE EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    p_all = primary_results["evaluation_all"]
    print(f"Final Train CTC Loss:           {primary_results['training']['final_loss']}")
    print(f"Test CTC Loss (All 8 Takes):     {p_all['loss']}")
    print(f"Test Exact Match (Recording):   {p_all['exact_match_recording_acc']}% ({p_all['total_recordings']} takes)")
    print(f"Test Exact Match (Sent Strict): {p_all['exact_match_sentence_strict_acc']}% ({p_all['total_sentences']} sentences)")
    print(f"Test Exact Match (Sent Relax):  {p_all['exact_match_sentence_relaxed_acc']}% ({p_all['total_sentences']} sentences)")
    print(f"Token Accuracy:                 {p_all['token_accuracy']}%")
    print(f"Token Error Rate (TER):         {p_all['token_error_rate']}%")
    print(f"CTC Blank Percentage:           {p_all['blank_percentage']}%")
    print(f"Blank-Only Collapse Detected?:  {p_all['is_blank_collapse']}")
    print("-" * 80)
    print("PER-RECORDING TEST PREDICTIONS:")
    for r in p_all["records"]:
        match_sym = "MATCH" if r["exact_match"] else "MISMATCH"
        print(
            f"  Sent {r['sentence_id']:2d} [{r['camera_view']:5s}]: "
            f"Ref={r['reference']} -> Pred={r['prediction']} ({match_sym}, Blank={r['blank_percentage']:.1f}%)"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
