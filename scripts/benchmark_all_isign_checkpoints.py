"""
Signova AI - F-017.1: Benchmark All Available iSign T5 Checkpoints.
Evaluates:
1. t5-small (spatial, 300)
2. t5-small-motion (motion, 600)
3. t5-base (spatial, 300)
4. t5-base-motion / t5-base-motion-v2 (motion, 600)
5. t5-large (spatial, 300)
6. t5-large-motion (motion, 600)

Reuses the exact 40 genuine iSign .pose files and UID-matched annotations.
Includes controlled sensitivity tests: Original, Zero, Random, Shuffled, Reversed.
Calculates BLEU, chrF, ROUGE-L, WER, Exact Match, latency, and repeatability metrics.
"""

from collections import Counter
import csv
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import psutil
    def get_ram_usage_mb() -> float:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
except ImportError:
    def get_ram_usage_mb() -> float:
        return 0.0
import torch
from transformers import AutoTokenizer

# Official repo inference imports
OFFICIAL_INFERENCE_DIR = Path("D:/iSign-t5-pose-to-text/inference")
if str(OFFICIAL_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_INFERENCE_DIR))

from model import PoseT5
import pose_utils as official_pose_utils

sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENUINE_DIR = PROJECT_ROOT / "datasets" / "genuine_samples"
ANNOTATIONS_CSV = PROJECT_ROOT / "datasets" / "iSign_v1.1.csv"
MODEL_REPO_ROOT = Path("D:/iSign-t5-pose-to-text")
TOKENIZER_DIR = PROJECT_ROOT / "models" / "tokenizer" / "t5-small"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "datasets" / "f017_1_checkpoint_benchmark_report.json"

CHECKPOINTS_TO_EVALUATE = [
    {
        "id": "t5-small",
        "folder": "t5-small",
        "base_model": "t5-small",
        "pose_dim": 300,
        "use_motion": False,
        "display_name": "t5-small (Spatial)",
    },
    {
        "id": "t5-small-motion",
        "folder": "t5-small-motion",
        "base_model": "t5-small",
        "pose_dim": 600,
        "use_motion": True,
        "display_name": "t5-small-motion (Spatial+Motion)",
    },
    {
        "id": "t5-base",
        "folder": "t5-base",
        "base_model": "t5-base",
        "pose_dim": 300,
        "use_motion": False,
        "display_name": "t5-base (Spatial)",
    },
    {
        "id": "t5-base-motion-v2",
        "folder": "t5-base-motion",  # local directory name
        "base_model": "t5-base",
        "pose_dim": 600,
        "use_motion": True,
        "display_name": "t5-base-motion-v2 (Spatial+Motion)",
    },
    {
        "id": "t5-large",
        "folder": "t5-large",
        "base_model": "t5-large",
        "pose_dim": 300,
        "use_motion": False,
        "display_name": "t5-large (Spatial)",
    },
    {
        "id": "t5-large-motion",
        "folder": "t5-large-motion",
        "base_model": "t5-large",
        "pose_dim": 600,
        "use_motion": True,
        "display_name": "t5-large-motion (Spatial+Motion)",
    },
]

# ---------------------------------------------------------------------------
# Metric Implementations (BLEU-4, chrF, ROUGE-L, WER)
# ---------------------------------------------------------------------------
def _get_ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu(references: List[str], hypotheses: List[str], max_n: int = 4) -> float:
    """Standard sentence-level smoothed BLEU-4."""
    total_score = 0.0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.lower().split()
        hyp_tokens = hyp.lower().split()
        if not hyp_tokens:
            continue
        bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens)))
        precisions = []
        for n in range(1, max_n + 1):
            ref_ng = Counter(_get_ngrams(ref_tokens, n))
            hyp_ng = Counter(_get_ngrams(hyp_tokens, n))
            total_hyp = max(1, sum(hyp_ng.values()))
            clipped = sum(min(count, ref_ng[ng]) for ng, count in hyp_ng.items())
            precisions.append((clipped + 1e-7) / (total_hyp + 1e-7))
        log_mean = sum(math.log(p) for p in precisions) / max_n
        total_score += bp * math.exp(log_mean)
    return round((total_score / max(1, len(hypotheses))) * 100.0, 3)


def compute_chrf(references: List[str], hypotheses: List[str], n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (Popovic 2015)."""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        r_chars = ref.lower().replace(" ", "")
        h_chars = hyp.lower().replace(" ", "")
        if not r_chars or not h_chars:
            scores.append(0.0)
            continue
        f_scores = []
        for i in range(1, n + 1):
            r_ng = Counter([r_chars[j : j + i] for j in range(len(r_chars) - i + 1)])
            h_ng = Counter([h_chars[j : j + i] for j in range(len(h_chars) - i + 1)])
            clipped = sum(min(count, r_ng[ng]) for ng, count in h_ng.items())
            prec = clipped / max(1, sum(h_ng.values()))
            rec = clipped / max(1, sum(r_ng.values()))
            if prec + rec > 0:
                f = (1.0 + beta**2) * prec * rec / ((beta**2 * prec) + rec)
            else:
                f = 0.0
            f_scores.append(f)
        scores.append(sum(f_scores) / n)
    return round((sum(scores) / max(1, len(scores))) * 100.0, 3)


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """Longest Common Subsequence (LCS) F1 score."""
    def lcs(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    scores = []
    for ref, hyp in zip(references, hypotheses):
        r = ref.lower().split()
        h = hyp.lower().split()
        if not r or not h:
            scores.append(0.0)
            continue
        match = lcs(r, h)
        rec = match / len(r)
        prec = match / len(h)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        scores.append(f1)
    return round((sum(scores) / max(1, len(scores))) * 100.0, 3)


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Word Error Rate via Levenshtein distance on words."""
    def edit_dist(r: List[str], h: List[str]) -> int:
        m, n = len(r), len(h)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if r[i - 1] == h[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[m][n]

    total_ed = 0
    total_len = 0
    for ref, hyp in zip(references, hypotheses):
        r = ref.lower().split()
        h = hyp.lower().split()
        total_ed += edit_dist(r, h)
        total_len += max(1, len(r))
    return round((total_ed / max(1, total_len)) * 100.0, 3)


# ---------------------------------------------------------------------------
# Benchmark Execution Engine
# ---------------------------------------------------------------------------


def load_dataset() -> List[Dict[str, Any]]:
    annotations = {}
    with open(ANNOTATIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                annotations[row[0].strip()] = row[1].strip()

    samples = []
    pose_files = sorted(list(GENUINE_DIR.glob("*.pose")))
    for pf in pose_files:
        uid = pf.stem
        ref = annotations.get(uid)
        if ref is not None:
            samples.append({
                "uid": uid,
                "path": str(pf),
                "reference": ref,
            })
    logger.info("Loaded %d genuine samples matched with reference annotations.", len(samples))
    return samples


def run_benchmark():
    print("=" * 80)
    print("Signova AI - F-017.1: Benchmark All Available iSign T5 Checkpoints")
    print("=" * 80)

    # 1. Inspect checkpoints
    print("\n[Step 1] Inspecting local checkpoints...")
    checkpoint_status = []
    for spec in CHECKPOINTS_TO_EVALUATE:
        ckpt_dir = MODEL_REPO_ROOT / spec["folder"]
        enc_file = ckpt_dir / "pose_encoder.pt"
        t5_safetensors = ckpt_dir / "t5" / "model.safetensors"
        t5_config = ckpt_dir / "t5" / "config.json"

        exists = ckpt_dir.exists() and enc_file.exists() and t5_safetensors.exists() and t5_config.exists()
        size_mb = 0.0
        if exists:
            size_mb = round((enc_file.stat().st_size + t5_safetensors.stat().st_size) / (1024 * 1024), 2)

        checkpoint_status.append({
            "id": spec["id"],
            "display_name": spec["display_name"],
            "path": str(ckpt_dir),
            "exists": exists,
            "size_mb": size_mb,
            "pose_dim": spec["pose_dim"],
            "use_motion": spec["use_motion"],
        })
        print(f" - {spec['display_name']:35} | Exists: {exists} | Size: {size_mb} MB")

    # 2. Fixed evaluation set
    print("\n[Step 2] Preparing fixed evaluation set...")
    samples = load_dataset()
    if len(samples) != 40:
        logger.warning("Expected 40 samples, found %d", len(samples))

    # Representative 5 samples for Step 4 controls
    control_sample_indices = [0, 9, 19, 29, 39]  # spread across 40 samples
    control_samples = [samples[i] for i in control_sample_indices]
    print(f"Selected {len(control_samples)} representative samples for Step 4 controlled tests.")

    # Shared tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)

    partial_cache_path = PROJECT_ROOT / "datasets" / "benchmark_partial.json"
    benchmark_results = []
    control_results_all = {}

    if partial_cache_path.exists():
        try:
            cached_data = json.loads(partial_cache_path.read_text(encoding="utf-8"))
            benchmark_results = cached_data.get("results", [])
            control_results_all = cached_data.get("controlled_tests", {})
            print(f"Loaded {len(benchmark_results)} checkpoints from partial cache: {[r['model_id'] for r in benchmark_results]}")
        except Exception as e:
            print("Failed to load partial cache:", e)

    completed_ids = {r["model_id"] for r in benchmark_results}

    # 3. Evaluate each checkpoint sequentially
    for spec in CHECKPOINTS_TO_EVALUATE:
        if spec["id"] in completed_ids:
            print(f"\n[CACHE HIT] Checkpoint {spec['display_name']} already evaluated. Skipping.")
            continue

        print("\n" + "=" * 70)
        print(f"EVALUATING CHECKPOINT: {spec['display_name']}")
        print("=" * 70)

        ckpt_dir = MODEL_REPO_ROOT / spec["folder"]
        if not ckpt_dir.exists():
            print(f"Skipping {spec['id']}: directory not found.")
            continue

        # Load model & measure load time
        t_load_start = time.perf_counter()
        ram_before = get_ram_usage_mb()

        model = PoseT5(
            model_name=spec["base_model"],
            pose_dim=spec["pose_dim"],
            local_model_path=str(ckpt_dir / "t5"),
        )
        encoder_weights = torch.load(str(ckpt_dir / "pose_encoder.pt"), map_location="cpu")
        model.pose_encoder.load_state_dict(encoder_weights)
        model.eval()

        load_duration_s = round(time.perf_counter() - t_load_start, 2)
        ram_after = get_ram_usage_mb()
        ram_delta = round(ram_after - ram_before, 2)
        print(f"Loaded in {load_duration_s}s | RAM: {ram_after} MB (delta: +{ram_delta} MB)")

        # Run 40 samples
        model_predictions = []
        references = []
        sample_latencies = []
        sample_records = []

        for idx, s in enumerate(samples):
            t_inf_start = time.perf_counter()
            pose_arr, actual_len = official_pose_utils.preprocess_pose(
                s["path"],
                max_length=500,
                normalization="zscore",
                pose_dim=spec["pose_dim"],
                use_motion=spec["use_motion"],
            )

            pose_t = torch.from_numpy(pose_arr).unsqueeze(0).float()
            len_t = torch.tensor([actual_len], dtype=torch.long)

            with torch.inference_mode():
                out_ids = model.generate(
                    pose=pose_t,
                    pose_length=len_t,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True,
                )

            pred_text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
            inf_time_ms = round((time.perf_counter() - t_inf_start) * 1000.0, 1)

            model_predictions.append(pred_text)
            references.append(s["reference"])
            sample_latencies.append(inf_time_ms)

            match = (pred_text.strip().lower() == s["reference"].strip().lower())
            sample_records.append({
                "uid": s["uid"],
                "reference": s["reference"],
                "prediction": pred_text,
                "exact_match": match,
                "latency_ms": inf_time_ms,
            })

            if (idx + 1) % 10 == 0 or idx == 0:
                print(f" [{idx+1}/{len(samples)}] Latency: {inf_time_ms}ms | Pred: {pred_text[:60]}...")

        # Calculate metrics
        bleu = compute_bleu(references, model_predictions)
        chrf = compute_chrf(references, model_predictions)
        rouge_l = compute_rouge_l(references, model_predictions)
        wer = compute_wer(references, model_predictions)
        exact_matches = sum(1 for r in sample_records if r["exact_match"])
        exact_match_rate = round((exact_matches / len(samples)) * 100.0, 2)
        avg_latency = round(float(np.mean(sample_latencies)), 1)

        unique_preds = len(set(model_predictions))
        pred_counts = Counter(model_predictions)
        repeated_preds = {k: v for k, v in pred_counts.items() if v > 1}
        empty_preds = sum(1 for p in model_predictions if not p.strip())

        print("\n--- Model Performance Summary ---")
        print(f"BLEU: {bleu} | chrF: {chrf} | ROUGE-L: {rouge_l} | WER: {wer}% | Exact Match: {exact_matches}/{len(samples)} ({exact_match_rate}%)")
        print(f"Unique Outputs: {unique_preds}/{len(samples)} | Repeated Output Count: {len(repeated_preds)} | Avg Latency: {avg_latency}ms")

        # Step 4: Controlled input tests on 5 samples
        print("\n[Step 4] Running controlled input tests on 5 samples...")
        ctrl_records = []
        for cs in control_samples:
            base_pose, base_len = official_pose_utils.preprocess_pose(
                cs["path"],
                max_length=500,
                normalization="zscore",
                pose_dim=spec["pose_dim"],
                use_motion=spec["use_motion"],
            )

            # 1. Original
            p_orig = torch.from_numpy(base_pose).unsqueeze(0).float()
            l_orig = torch.tensor([base_len], dtype=torch.long)

            # 2. Zero sequence
            p_zero = torch.zeros_like(p_orig)

            # 3. Random sequence
            np.random.seed(42)
            p_rand = torch.from_numpy(np.random.randn(*base_pose.shape).astype(np.float32)).unsqueeze(0)

            # 4. Temporally shuffled sequence
            shuff_idx = np.random.permutation(base_len)
            shuff_pose = base_pose.copy()
            shuff_pose[:base_len] = base_pose[shuff_idx]
            p_shuff = torch.from_numpy(shuff_pose).unsqueeze(0).float()

            # 5. Temporally reversed sequence
            rev_pose = base_pose.copy()
            rev_pose[:base_len] = base_pose[:base_len][::-1]
            p_rev = torch.from_numpy(rev_pose).unsqueeze(0).float()

            controls_dict = {}
            with torch.inference_mode():
                for c_name, tensor_in in [
                    ("original", p_orig),
                    ("zero", p_zero),
                    ("random", p_rand),
                    ("shuffled", p_shuff),
                    ("reversed", p_rev),
                ]:
                    gen = model.generate(pose=tensor_in, pose_length=l_orig, max_length=128, num_beams=4)
                    dec = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
                    controls_dict[c_name] = dec

            ctrl_records.append({
                "uid": cs["uid"],
                "reference": cs["reference"],
                "controls": controls_dict,
            })
            print(f"UID {cs['uid']}:")
            for c_name, pred_val in controls_dict.items():
                print(f"   {c_name:10}: {pred_val}")

        control_results_all[spec["id"]] = ctrl_records

        benchmark_results.append({
            "model_id": spec["id"],
            "display_name": spec["display_name"],
            "base_model": spec["base_model"],
            "pose_dim": spec["pose_dim"],
            "use_motion": spec["use_motion"],
            "load_duration_s": load_duration_s,
            "ram_mb": ram_after,
            "avg_latency_ms": avg_latency,
            "bleu": bleu,
            "chrf": chrf,
            "rouge_l": rouge_l,
            "wer": wer,
            "exact_matches": exact_matches,
            "exact_match_rate": exact_match_rate,
            "unique_predictions": unique_preds,
            "repeated_predictions": repeated_preds,
            "empty_predictions": empty_preds,
            "sample_records": sample_records,
        })

        partial_cache_path.write_text(
            json.dumps({"results": benchmark_results, "controlled_tests": control_results_all}, indent=2),
            encoding="utf-8"
        )
        print(f"[CACHE SAVED] Saved partial results ({len(benchmark_results)} checkpoints done) to {partial_cache_path}")

        # Memory cleanup before next checkpoint
        del model
        del encoder_weights
        gc.collect()

    # Compile final comparison
    final_report = {
        "title": "F-017.1: iSign T5 Multi-Checkpoint Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_checkpoints_evaluated": len(benchmark_results),
        "evaluation_samples_count": len(samples),
        "checkpoints_status": checkpoint_status,
        "results": benchmark_results,
        "controlled_tests": control_results_all,
    }

    OUTPUT_REPORT_PATH.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"Benchmark completed successfully! Report written to: {OUTPUT_REPORT_PATH}")
    print("=" * 80)

    # Print summary table
    print("\n| Model | BLEU | chrF | ROUGE-L | WER | Exact Match | Avg Latency |")
    print("|---|---|---|---|---|---|---|")
    for r in benchmark_results:
        print(f"| {r['display_name']} | {r['bleu']} | {r['chrf']} | {r['rouge_l']} | {r['wer']}% | {r['exact_matches']}/{len(samples)} ({r['exact_match_rate']}%) | {r['avg_latency_ms']}ms |")


if __name__ == "__main__":
    run_benchmark()
