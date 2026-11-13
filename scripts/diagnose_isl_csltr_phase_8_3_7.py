"""
Phase 8.3.7: Step A - Diagnostic Audit Script.

Investigates:
1. Gloss frequency distribution in training split.
2. Number of glosses appearing only once or very few times (<3, <5).
3. Target sequence length distribution across splits.
4. CTC input length (subsampled T') versus target length (S).
5. Repeated label constraints and CTC feasibility.
6. Baseline (Phase 8.3.6) blank posterior distribution and temporal region analysis.
7. Baseline per-gloss prediction frequency, precision, recall, and F1.
8. Predicted sequence length versus reference sequence length.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.isl_csltr_dataset import ISLCSLTRDataset, collate_isl_csltr_batch
from ai.continuous.isl_csltr_model import (
    ISLCSLTRContinuousModel,
    compute_sequence_metrics,
    decode_greedy_ctc,
)


def run_diagnostic_audit() -> Dict[str, Any]:
    dataset_root = PROJECT_ROOT / "datasets" / "isl_csltr_continuous"
    train_manifest_path = dataset_root / "manifests" / "train_manifest.json"
    val_manifest_path = dataset_root / "manifests" / "validation_manifest.json"
    test_manifest_path = dataset_root / "manifests" / "test_manifest.json"
    vocab_path = dataset_root / "metadata" / "vocabulary.json"

    baseline_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous"
    baseline_ckpt_path = baseline_dir / "best_checkpoint.pt"
    baseline_config_path = baseline_dir / "model_config.json"

    out_dir = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "phase_8_3_7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load manifests
    with open(train_manifest_path, "r", encoding="utf-8") as f:
        train_records = [r for r in json.load(f) if r.get("is_canonical", True)]
    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_records = [r for r in json.load(f) if r.get("is_canonical", True)]
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_records = [r for r in json.load(f) if r.get("is_canonical", True)]
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    gloss_to_id = vocab_data["gloss_to_id"]
    id_to_gloss = {int(v): k for k, v in gloss_to_id.items()}
    num_classes = vocab_data.get("num_classes", len(gloss_to_id))

    print(f"Loaded manifests: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    print(f"Total vocabulary classes: {num_classes} (blank=0)")

    # 2. Gloss frequency distribution in training
    train_gloss_counter = Counter()
    for r in train_records:
        for t in r["tokens"]:
            train_gloss_counter[t] += 1

    total_tokens_train = sum(train_gloss_counter.values())
    unique_glosses_train = len(train_gloss_counter)

    # Frequency tiers
    freq_values = list(train_gloss_counter.values())
    count_1 = sum(1 for c in freq_values if c == 1)
    count_2 = sum(1 for c in freq_values if c == 2)
    count_under_3 = sum(1 for c in freq_values if c < 3)
    count_under_5 = sum(1 for c in freq_values if c < 5)
    count_over_20 = sum(1 for c in freq_values if c >= 20)
    count_over_50 = sum(1 for c in freq_values if c >= 50)

    top_10_glosses = train_gloss_counter.most_common(10)
    bottom_10_glosses = train_gloss_counter.most_common()[:-11:-1]

    print("\n--- Gloss Frequency Audit ---")
    print(f"Total training tokens: {total_tokens_train}")
    print(f"Unique glosses present in train: {unique_glosses_train} / 170 ISL glosses")
    print(f"Glosses appearing exactly once: {count_1} ({count_1/170:.1%})")
    print(f"Glosses appearing < 3 times: {count_under_3} ({count_under_3/170:.1%})")
    print(f"Glosses appearing < 5 times: {count_under_5} ({count_under_5/170:.1%})")
    print(f"Top 5 glosses: {train_gloss_counter.most_common(5)}")

    # 3. Target sequence length distribution
    def get_length_stats(records: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        target_lens = [len(r["tokens"]) for r in records]
        frame_lens = [r.get("video_frame_count", r.get("frame_count", 0)) for r in records]
        sub_lens = [(fl + 1) // 2 for fl in frame_lens]

        # Check CTC feasibility: sub_lens >= target_lens
        ctc_valid = [sl >= tl for sl, tl in zip(sub_lens, target_lens)]
        ratio = [sl / tl if tl > 0 else 0 for sl, tl in zip(sub_lens, target_lens)]

        # Repeated label audit: consecutive identical labels
        repeated_labels = 0
        for r in records:
            toks = r["tokens"]
            for i in range(len(toks) - 1):
                if toks[i] == toks[i + 1]:
                    repeated_labels += 1

        return {
            "num_samples": len(records),
            "target_len_min": int(np.min(target_lens)),
            "target_len_mean": float(np.mean(target_lens)),
            "target_len_median": float(np.median(target_lens)),
            "target_len_max": int(np.max(target_lens)),
            "frame_len_min": int(np.min(frame_lens)),
            "frame_len_mean": float(np.mean(frame_lens)),
            "frame_len_median": float(np.median(frame_lens)),
            "frame_len_max": int(np.max(frame_lens)),
            "subsampled_len_min": int(np.min(sub_lens)),
            "subsampled_len_mean": float(np.mean(sub_lens)),
            "subsampled_len_max": int(np.max(sub_lens)),
            "min_length_ratio_T_prime_over_S": float(np.min(ratio)),
            "mean_length_ratio_T_prime_over_S": float(np.mean(ratio)),
            "ctc_length_valid_count": int(sum(ctc_valid)),
            "ctc_length_valid_pct": float(np.mean(ctc_valid) * 100.0),
            "consecutive_identical_labels": repeated_labels,
        }

    train_len_stats = get_length_stats(train_records, "train")
    val_len_stats = get_length_stats(val_records, "val")
    test_len_stats = get_length_stats(test_records, "test")

    print("\n--- Sequence Length and CTC Feasibility ---")
    print(f"Train: Target len={train_len_stats['target_len_min']}-{train_len_stats['target_len_max']} (mean {train_len_stats['target_len_mean']:.2f}), Subsampled frames={train_len_stats['subsampled_len_min']}-{train_len_stats['subsampled_len_max']} (mean {train_len_stats['subsampled_len_mean']:.2f})")
    print(f"Train CTC validity: {train_len_stats['ctc_length_valid_pct']:.1f}% (min ratio T'/S: {train_len_stats['min_length_ratio_T_prime_over_S']:.2f})")
    print(f"Val CTC validity:   {val_len_stats['ctc_length_valid_pct']:.1f}% (min ratio T'/S: {val_len_stats['min_length_ratio_T_prime_over_S']:.2f})")
    print(f"Test CTC validity:  {test_len_stats['ctc_length_valid_pct']:.1f}% (min ratio T'/S: {test_len_stats['min_length_ratio_T_prime_over_S']:.2f})")
    print(f"Consecutive identical glosses in Train: {train_len_stats['consecutive_identical_labels']}")

    # 4. Load baseline model and evaluate posteriors & predictions on Validation set
    with open(baseline_config_path, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    device = torch.device("cpu")
    model = ISLCSLTRContinuousModel(
        input_dim=model_cfg.get("input_dim", 300),
        proj_dim=model_cfg.get("proj_dim", 128),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 2),
        num_classes=num_classes,
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    ckpt = torch.load(baseline_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = ISLCSLTRDataset(
        manifest_path=val_manifest_path,
        features_root=dataset_root / "features" / "validation",
        vocab_path=vocab_path,
        canonical_only=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_isl_csltr_batch,
    )

    all_blank_posteriors: List[float] = []
    # Divide sequence into 5 temporal quintiles: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    temporal_blank_posteriors: Dict[int, List[float]] = {i: [] for i in range(5)}
    temporal_argmax_is_blank: Dict[int, List[int]] = {i: [] for i in range(5)}

    val_refs: List[List[str]] = []
    val_hyps: List[List[str]] = []
    ref_lengths: List[int] = []
    pred_lengths: List[int] = []

    hyp_gloss_counter = Counter()

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            tokens_list = batch["tokens_list"]

            log_probs, sub_lens = model(features, input_lengths)
            probs = torch.exp(log_probs).transpose(0, 1)  # (B, T', num_classes)

            decoded = decode_greedy_ctc(log_probs, sub_lens, blank_idx=0, id_to_gloss=id_to_gloss)

            for i in range(features.size(0)):
                slen = int(sub_lens[i].item())
                sample_probs = probs[i, :slen]  # (slen, num_classes)
                blank_probs = sample_probs[:, 0].cpu().numpy()  # (slen,)
                all_blank_posteriors.extend(blank_probs.tolist())

                argmax_tokens = torch.argmax(sample_probs, dim=-1).cpu().numpy()

                for t_idx in range(slen):
                    q_idx = min(int((t_idx / slen) * 5), 4)
                    temporal_blank_posteriors[q_idx].append(float(blank_probs[t_idx]))
                    temporal_argmax_is_blank[q_idx].append(1 if argmax_tokens[t_idx] == 0 else 0)

                ref = tokens_list[i]
                hyp = decoded[i]["tokens"]
                val_refs.append(ref)
                val_hyps.append(hyp)
                ref_lengths.append(len(ref))
                pred_lengths.append(len(hyp))

                for h in hyp:
                    hyp_gloss_counter[h] += 1

    seq_metrics = compute_sequence_metrics(val_refs, val_hyps)

    # Compute temporal distribution summary
    temporal_analysis = {}
    for q_idx in range(5):
        q_label = f"Q{q_idx+1}_{q_idx*20}-{(q_idx+1)*20}%"
        q_blank_probs = temporal_blank_posteriors[q_idx]
        q_argmax = temporal_argmax_is_blank[q_idx]
        temporal_analysis[q_label] = {
            "mean_blank_posterior": float(np.mean(q_blank_probs)),
            "blank_argmax_rate": float(np.mean(q_argmax) * 100.0),
            "frame_count": len(q_blank_probs),
        }

    posteriors_np = np.array(all_blank_posteriors)
    posterior_stats = {
        "mean": float(np.mean(posteriors_np)),
        "std": float(np.std(posteriors_np)),
        "median": float(np.median(posteriors_np)),
        "min": float(np.min(posteriors_np)),
        "max": float(np.max(posteriors_np)),
        "fraction_over_0_5": float(np.mean(posteriors_np > 0.5)),
        "fraction_over_0_9": float(np.mean(posteriors_np > 0.9)),
        "fraction_over_0_99": float(np.mean(posteriors_np > 0.99)),
    }

    pred_vs_ref_len = {
        "mean_ref_length": float(np.mean(ref_lengths)),
        "mean_pred_length": float(np.mean(pred_lengths)),
        "ratio_pred_to_ref": float(np.mean(pred_lengths) / np.mean(ref_lengths)),
        "empty_prediction_count": sum(1 for pl in pred_lengths if pl == 0),
        "empty_prediction_pct": float(np.mean([pl == 0 for pl in pred_lengths]) * 100.0),
    }

    print("\n--- Blank Posterior Audit ---")
    print(f"Mean Blank Posterior: {posterior_stats['mean']:.4f} (Median: {posterior_stats['median']:.4f})")
    print(f"Frames with P(blank) > 0.90: {posterior_stats['fraction_over_0_9']:.1%}")
    print(f"Frames with P(blank) > 0.99: {posterior_stats['fraction_over_0_99']:.1%}")
    print("\n--- Blank Dominance by Temporal Region ---")
    for q_lbl, q_data in temporal_analysis.items():
        print(f"  {q_lbl}: Mean Blank Prob = {q_data['mean_blank_posterior']:.4f}, Argmax Blank = {q_data['blank_argmax_rate']:.1f}%")

    print("\n--- Sequence Length Analysis ---")
    print(f"Mean Ref Length:  {pred_vs_ref_len['mean_ref_length']:.2f} tokens")
    print(f"Mean Pred Length: {pred_vs_ref_len['mean_pred_length']:.2f} tokens (ratio {pred_vs_ref_len['ratio_pred_to_ref']:.2f})")
    print(f"Empty predictions: {pred_vs_ref_len['empty_prediction_count']} / {len(pred_lengths)} ({pred_vs_ref_len['empty_prediction_pct']:.1f}%)")

    print("\n--- Baseline Hypothesis Gloss Frequencies ---")
    print(f"Hypothesis Gloss Counts: {hyp_gloss_counter.most_common(10)}")
    print(f"Distinct glosses predicted: {len(hyp_gloss_counter)} / {num_classes}")

    diagnostic_report = {
        "dataset_audit": {
            "total_tokens_train": total_tokens_train,
            "unique_glosses_train": unique_glosses_train,
            "glosses_appearing_1x": count_1,
            "glosses_appearing_2x": count_2,
            "glosses_appearing_under_3x": count_under_3,
            "glosses_appearing_under_5x": count_under_5,
            "glosses_appearing_over_20x": count_over_20,
            "top_10_train_glosses": top_10_glosses,
            "bottom_10_train_glosses": bottom_10_glosses,
            "train_len_stats": train_len_stats,
            "val_len_stats": val_len_stats,
            "test_len_stats": test_len_stats,
        },
        "baseline_posterior_audit": {
            "posterior_stats": posterior_stats,
            "temporal_distribution": temporal_analysis,
        },
        "baseline_prediction_audit": {
            "pred_vs_ref_length": pred_vs_ref_len,
            "hyp_gloss_counts": dict(hyp_gloss_counter),
            "macro_f1": seq_metrics["macro_f1"],
            "wer": seq_metrics["wer"],
            "exact_match_rate": seq_metrics["exact_match_rate"],
            "per_gloss_top_performers": {
                g: m for g, m in seq_metrics["per_gloss"].items() if m["f1"] > 0
            },
        },
    }

    report_path = out_dir / "diagnostic_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(diagnostic_report, f, indent=2)

    print(f"\nDiagnostic audit saved to {report_path}")
    return diagnostic_report


if __name__ == "__main__":
    run_diagnostic_audit()
