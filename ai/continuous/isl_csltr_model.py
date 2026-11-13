"""
Temporal CNN + 2-Layer Bidirectional GRU + CTC Continuous ISL Model.

Phase 8.3.6 Implementation.
Ponytail design: Minimal dependencies, torch-native, zero unnecessary abstractions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_subsampled_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Compute output sequence lengths after Conv1D(kernel=3, stride=2, padding=1).
    
    Formula: floor((T + 2*1 - 3) / 2) + 1 = floor((T - 1) / 2) + 1 = (T + 1) // 2
    """
    return (input_lengths + 1) // 2


class ISLCSLTRContinuousModel(nn.Module):
    """Temporal Conv1D + 2-layer Bidirectional GRU + CTC for ISL-CSLTR Continuous Recognition."""

    def __init__(
        self,
        input_dim: int = 300,
        proj_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 171,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # 1. Feature Projection
        self.proj = nn.Linear(input_dim, proj_dim)
        self.proj_ln = nn.LayerNorm(proj_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.proj_relu = nn.ReLU()

        # 2. Temporal Subsampling (Stride-2 Conv1D)
        self.conv = nn.Conv1d(
            in_channels=proj_dim,
            out_channels=proj_dim,
            kernel_size=3,
            stride=2,
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
            subsampled_lengths: 1D Tensor of shape (B,) with subsampled lengths T'_i
        """
        b, t, _ = x.shape
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long, device=x.device)

        # 1. Projection
        h = self.proj(x)
        h = self.proj_ln(h)
        h = self.proj_dropout(h)
        h = self.proj_relu(h)

        # 2. Temporal Subsampling: conv expects (B, C, T)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = h.transpose(1, 2)  # back to (B, T', C)
        h = self.conv_ln(h)
        h = self.conv_relu(h)

        subsampled_lengths = compute_subsampled_lengths(lengths)
        # Ensure lengths stay on same device
        subsampled_lengths = subsampled_lengths.to(x.device)

        # 3. Recurrent Encoding with pack_padded_sequence
        if subsampled_lengths.min() != subsampled_lengths.max():
            packed = nn.utils.rnn.pack_padded_sequence(
                h, subsampled_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(h)

        # 4. Classifier
        out = self.head_dropout(out)
        logits = self.classifier(out)  # (B, T', num_classes)

        # PyTorch CTCLoss expects (T', B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return log_probs, subsampled_lengths


def decode_greedy_ctc(
    log_probs: torch.Tensor,
    subsampled_lengths: torch.Tensor,
    blank_idx: int = 0,
    id_to_gloss: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """Greedy CTC decoder with blank-collapse and consecutive duplicate removal.
    
    Args:
        log_probs: Tensor of shape (T', B, num_classes)
        subsampled_lengths: Tensor of shape (B,)
        blank_idx: Index of blank token (default 0)
        id_to_gloss: Optional mapping from class index to gloss string
        
    Returns:
        List of decoded result dicts per batch item.
    """
    probs_b = log_probs.detach().transpose(0, 1)  # (B, T', num_classes)
    batch_size = probs_b.shape[0]
    results: List[Dict[str, Any]] = []

    for i in range(batch_size):
        t_len = int(subsampled_lengths[i].item())
        sample_probs = probs_b[i, :t_len]
        if t_len == 0:
            results.append({
                "token_ids": [],
                "tokens": [],
                "raw_preds": [],
                "blank_percentage": 0.0,
                "total_frames": 0,
            })
            continue

        raw_preds = torch.argmax(sample_probs, dim=-1).cpu().tolist()
        blank_count = sum(1 for p in raw_preds if p == blank_idx)
        blank_pct = (blank_count / t_len * 100.0) if t_len > 0 else 0.0

        # CTC collapse: deduplicate consecutive identical tokens, then drop blanks
        collapsed: List[int] = []
        prev = None
        for p in raw_preds:
            if p != prev:
                if p != blank_idx:
                    collapsed.append(p)
                prev = p

        tokens = [
            (id_to_gloss.get(idx, f"unk_{idx}") if id_to_gloss else str(idx))
            for idx in collapsed
        ]

        results.append({
            "token_ids": collapsed,
            "tokens": tokens,
            "raw_preds": raw_preds,
            "blank_percentage": blank_pct,
            "total_frames": t_len,
        })

    return results


def compute_levenshtein(seq1: List[Any], seq2: List[Any]) -> Tuple[int, int, int, int]:
    """Compute Levenshtein edit distance and operation counts (insertions, deletions, substitutions).
    
    Returns:
        (distance, substitutions, deletions, insertions)
    """
    n, m = len(seq1), len(seq2)
    # dp[i][j] stores (cost, subs, dels, ins)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # i deletions from seq1
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # j insertions into seq1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                cost_sub = dp[i - 1][j - 1][0] + 1
                cost_del = dp[i - 1][j][0] + 1
                cost_ins = dp[i][j - 1][0] + 1

                min_cost = min(cost_sub, cost_del, cost_ins)
                if min_cost == cost_sub:
                    c, s, d, ins = dp[i - 1][j - 1]
                    dp[i][j] = (c + 1, s + 1, d, ins)
                elif min_cost == cost_del:
                    c, s, d, ins = dp[i - 1][j]
                    dp[i][j] = (c + 1, s, d + 1, ins)
                else:
                    c, s, d, ins = dp[i][j - 1]
                    dp[i][j] = (c + 1, s, d, ins + 1)

    return dp[n][m]


def compute_sequence_metrics(
    references: List[List[str]],
    hypotheses: List[List[str]],
    all_glosses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate Sequence-level metrics: WER, TER, Exact Match, per-gloss F1, Macro F1, Confusions."""
    total_samples = len(references)
    if total_samples == 0:
        return {"wer": 0.0, "ter": 0.0, "exact_match": 0.0, "total_samples": 0}

    total_ref_tokens = sum(len(r) for r in references)
    total_dist = 0
    total_subs = 0
    total_dels = 0
    total_ins = 0
    exact_matches = 0

    # Per-gloss counts
    ref_counts = Counter()
    hyp_counts = Counter()
    tp_counts = Counter()
    confusion_pairs = Counter()

    for ref, hyp in zip(references, hypotheses):
        if ref == hyp:
            exact_matches += 1

        dist, subs, dels, ins = compute_levenshtein(ref, hyp)
        total_dist += dist
        total_subs += subs
        total_dels += dels
        total_ins += ins

        for t in ref:
            ref_counts[t] += 1
        for t in hyp:
            hyp_counts[t] += 1

        # Multiset intersection for true positives
        hyp_copy = list(hyp)
        for t in ref:
            if t in hyp_copy:
                tp_counts[t] += 1
                hyp_copy.remove(t)
            else:
                # If deleted or substituted
                if hyp_copy:
                    sub_cand = hyp_copy.pop(0)
                    confusion_pairs[(t, sub_cand)] += 1
                else:
                    confusion_pairs[(t, "<deletion>")] += 1

        for extra in hyp_copy:
            confusion_pairs[("<insertion>", extra)] += 1

    wer = (total_dist / total_ref_tokens) if total_ref_tokens > 0 else 0.0
    ter = wer  # On word/token sequences, token error rate = WER
    exact_match_rate = exact_matches / total_samples

    # Compute per-gloss precision, recall, F1
    gloss_vocab = set(all_glosses) if all_glosses else (set(ref_counts.keys()) | set(hyp_counts.keys()))
    per_gloss: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []

    for g in sorted(gloss_vocab):
        tp = tp_counts[g]
        fp = hyp_counts[g] - tp
        fn = ref_counts[g] - tp
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if ref_counts[g] > 0 or hyp_counts[g] > 0:
            per_gloss[g] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ref_count": ref_counts[g],
                "hyp_count": hyp_counts[g],
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "total_samples": total_samples,
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_match_rate, 4),
        "total_ref_tokens": total_ref_tokens,
        "total_dist": total_dist,
        "substitutions": total_subs,
        "deletions": total_dels,
        "insertions": total_ins,
        "wer": round(wer, 4),
        "ter": round(ter, 4),
        "macro_f1": round(macro_f1, 4),
        "per_gloss": per_gloss,
        "top_confusions": confusion_pairs.most_common(15),
    }
