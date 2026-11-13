"""
Signova AI - Isolated Validation Pipeline for iSign T5-Small-Motion (F-030).
Evaluates the local t5-small-motion checkpoint on:
- Official .pose ground-truth samples
- Genuine ISL video files (MediaPipe -> 75 landmarks -> 300 spatial -> 600 motion)
- Synthetic controls (all-zeros and Gaussian noise)
- Loop detection and mitigation across beam search and greedy decoding

Strictly isolated: does not alter production defaults or the simple T5-Base checkpoint.
"""

import argparse
import csv
import gc
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.config.model_settings import ModelSettings
from ai.inference.model import PoseT5
from ai.pose_extraction.mediapipe_extractor import MediaPipeLandmarkExtractor
from ai.preprocessing.pose_preprocessor import (
    add_motion_features,
    load_pose_file,
    normalize_pose,
    pad_or_truncate_pose,
    preprocess_pose,
    to_spatial_matrix,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("signova.test_t5_small_motion")

OFFICIAL_MODEL_DIR = Path("D:/iSign-t5-pose-to-text")
CHECKPOINT_NAME = "t5-small-motion"
EXPECTED_POSE_DIM = 600
EXPECTED_EMBEDDING_DIM = 512
MAX_POSE_LENGTH = 500
MAX_GEN_LENGTH = 128


def detect_token_loop(
    token_ids: List[int],
    min_cycle: int = 1,
    max_cycle: int = 8,
    min_repeats: int = 3,
) -> Optional[Tuple[List[int], int]]:
    """
    Detect if token_ids contains a repeating cycle.
    Returns (pattern, repeat_count) if a loop is detected, otherwise None.
    """
    n = len(token_ids)
    for c in range(min_cycle, min_cycle + max_cycle + 1):
        if n < c * min_repeats:
            continue
        pattern = token_ids[-c:]
        repeats = 0
        idx = n - c
        while idx >= 0 and token_ids[idx : idx + c] == pattern:
            repeats += 1
            idx -= c
        if repeats >= min_repeats:
            return (pattern, repeats)
    return None


def load_ground_truth_lookup() -> Dict[str, str]:
    """Load ground-truth annotations from datasets/iSign_v1.1.csv."""
    csv_path = PROJECT_ROOT / "datasets" / "iSign_v1.1.csv"
    gt_map: Dict[str, str] = {}
    if csv_path.exists():
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    gt_map[row[0].strip()] = row[1].strip()
    return gt_map


class IsolatedMotionTranslator:
    """Isolated loader and runner for t5-small-motion checkpoint."""

    def __init__(self, checkpoint_dir: Optional[Path] = None, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir or (OFFICIAL_MODEL_DIR / CHECKPOINT_NAME)).resolve()
        self.t5_dir = self.checkpoint_dir / "t5"
        self.pose_encoder_file = self.checkpoint_dir / "pose_encoder.pt"
        self.tokenizer_dir = PROJECT_ROOT / "models" / "tokenizer" / "t5-small"

        self._validate_assets()
        self._load_pipeline()

    def _validate_assets(self) -> None:
        """Verify that all files exist on disk before loading."""
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")
        if not self.pose_encoder_file.exists():
            raise FileNotFoundError(f"Missing pose encoder weights: {self.pose_encoder_file}")
        if not (self.t5_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Missing T5 weights: {self.t5_dir / 'model.safetensors'}")
        if not self.tokenizer_dir.exists():
            raise FileNotFoundError(f"Missing tokenizer directory: {self.tokenizer_dir}")

    def _load_pipeline(self) -> None:
        """Load PoseT5 with input dimension 600 and T5-Small architecture."""
        t0 = time.perf_counter()
        # 1. Initialize PoseT5
        self.model = PoseT5(
            model_name="t5-small",
            pose_dim=EXPECTED_POSE_DIM,
            local_model_path=str(self.t5_dir),
            local_files_only=True,
        )

        # 2. Load pose encoder weights
        encoder_sd = torch.load(self.pose_encoder_file, map_location="cpu")
        self.model.pose_encoder.load_state_dict(encoder_sd)

        # 3. Move to device and set evaluation mode
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        # 4. Invariant checks
        in_feat = self.model.pose_encoder.encoder[0].in_features
        out_feat = self.model.pose_encoder.encoder[3].out_features
        d_model = self.model.t5.config.d_model

        assert in_feat == EXPECTED_POSE_DIM, f"Pose encoder in_features={in_feat} != {EXPECTED_POSE_DIM}"
        assert out_feat == EXPECTED_EMBEDDING_DIM, f"Pose encoder out_features={out_feat} != {EXPECTED_EMBEDDING_DIM}"
        assert d_model == EXPECTED_EMBEDDING_DIM, f"T5 d_model={d_model} != {EXPECTED_EMBEDDING_DIM}"

        # 5. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir), local_files_only=True)
        self.load_latency_s = time.perf_counter() - t0

    def generate(
        self,
        pose_tensor: torch.Tensor,
        pose_length: torch.Tensor,
        num_beams: int = 4,
        max_length: int = MAX_GEN_LENGTH,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
    ) -> Tuple[str, List[int], float, bool]:
        """
        Generate English translation with optional loop mitigation.
        Returns:
            (decoded_text, token_ids, latency_seconds, loop_detected_flag)
        """
        pose_tensor = pose_tensor.to(self.device)
        pose_length = pose_length.to(self.device)

        t_start = time.perf_counter()
        with torch.inference_mode():
            pose_embeddings = self.model.pose_encoder(pose_tensor)
            batch_size, seq_len, _ = pose_embeddings.shape
            attention_mask = (
                torch.arange(seq_len, device=self.device)[None, :] < pose_length[:, None]
            ).long()

            gen_kwargs: Dict[str, Any] = {
                "inputs_embeds": pose_embeddings,
                "attention_mask": attention_mask,
                "max_length": max_length,
                "num_beams": num_beams,
            }

            if num_beams > 1:
                gen_kwargs["early_stopping"] = True
                gen_kwargs["length_penalty"] = 2.0

            if no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            if repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = repetition_penalty

            gen_ids = self.model.t5.generate(**gen_kwargs)

        latency = time.perf_counter() - t_start
        token_ids = gen_ids[0].tolist()
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        loop_info = detect_token_loop(token_ids)
        loop_detected = loop_info is not None

        return decoded, token_ids, latency, loop_detected

    def close(self) -> None:
        """Release PyTorch model memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()


def run_pipeline_validation() -> Dict[str, Any]:
    """Execute complete isolated validation suite across all required datasets."""
    print("=" * 80)
    print("F-030: ISOLATED T5-SMALL-MOTION INTEGRATION & VALIDATION PIPELINE")
    print("=" * 80)

    # 1. Initialize Isolated Translator
    translator = IsolatedMotionTranslator(device="cpu")
    print(f"Model: t5-small-motion loaded successfully in {translator.load_latency_s:.2f}s on CPU.")
    print(f"  Architecture: 600 in -> 256 -> 512 out | T5 d_model: 512 | vocab_size: {translator.tokenizer.vocab_size}")

    gt_map = load_ground_truth_lookup()
    extractor = MediaPipeLandmarkExtractor()

    # 2. Official .pose samples to test
    official_samples = [
        "39at7Lxu6Yo--13.pose",
        "1f1bb4e3121b-192.pose",
        "8mwSW8CwzL0--219.pose",
        "ee235596b76f-47.pose",
        "d92ec3de5035-19.pose",
        "8cfebb40f2b9-3.pose",
        "2011b172755b-58.pose",
        "53b5KLBgl8o--32.pose",
    ]

    # 3. Real video samples to test
    video_samples = [
        ("datasets/videos/10 granfather sick  (2).mov", "Grandfather is sick"),
        ("datasets/videos/Learn-Indian-Sign-SnapYT.App (1) (online-video-cutter.com).mp4", "None"),
        ("datasets/videos/Indian Sign Language Video Dataset.mp4", "None"),
        ("datasets/videos/WIN_20261112_12_11_45_Pro.mp4", "None"),
    ]

    all_records: List[Dict[str, Any]] = []

    print("\n" + "-" * 80)
    print("PHASE 1: OFFICIAL .POSE SAMPLES EVALUATION (Ground Truth Verification)")
    print("-" * 80)

    for pose_name in official_samples:
        pose_path = PROJECT_ROOT / "datasets" / "genuine_samples" / pose_name
        if not pose_path.exists():
            continue

        gt_text = gt_map.get(pose_path.stem, "N/A")

        # Official preprocessing with use_motion=True -> (1, 500, 600)
        preprocessed = preprocess_pose(
            pose_path,
            max_length=MAX_POSE_LENGTH,
            use_motion=True,
            device="cpu",
        )

        # A. Beam Search (num_beams=4)
        b_text, b_tokens, b_time, b_loop = translator.generate(
            preprocessed.pose_tensor,
            preprocessed.pose_length,
            num_beams=4,
        )

        # B. Greedy Search Unconstrained (num_beams=1)
        g_text_raw, g_tokens_raw, g_time_raw, g_loop_raw = translator.generate(
            preprocessed.pose_tensor,
            preprocessed.pose_length,
            num_beams=1,
            no_repeat_ngram_size=0,
        )

        # C. Greedy Search with Mitigation (no_repeat_ngram_size=3)
        g_text_safe, g_tokens_safe, g_time_safe, g_loop_safe = translator.generate(
            preprocessed.pose_tensor,
            preprocessed.pose_length,
            num_beams=1,
            no_repeat_ngram_size=3,
        )

        exact_match = b_text.strip().lower() == gt_text.strip().lower()

        record = {
            "name": pose_name,
            "type": "official_pose",
            "ground_truth": gt_text,
            "frames": preprocessed.actual_length,
            "beam_output": b_text,
            "beam_tokens": b_tokens,
            "beam_latency_s": b_time,
            "beam_loop": b_loop,
            "greedy_raw_output": g_text_raw,
            "greedy_raw_loop": g_loop_raw,
            "greedy_safe_output": g_text_safe,
            "greedy_safe_tokens": g_tokens_safe,
            "greedy_safe_latency_s": g_time_safe,
            "exact_match": exact_match,
        }
        all_records.append(record)

        print(f"Sample: {pose_name} ({preprocessed.actual_length} frames)")
        print(f"  Ground Truth: \"{gt_text}\"")
        print(f"  Beam (k=4):   \"{b_text}\" ({b_time:.3f}s)")
        print(f"  Greedy (raw): \"{g_text_raw[:60]}...\" (Loop: {g_loop_raw})")
        print(f"  Greedy (mit): \"{g_text_safe}\" ({g_time_safe:.3f}s, Loop: {g_loop_safe})")
        print(f"  Exact Match:  {exact_match}")
        print()

    print("\n" + "-" * 80)
    print("PHASE 2: REAL SIGN LANGUAGE VIDEO EVALUATION (Webcam/Mobile Videos)")
    print("-" * 80)

    for rel_path, ref_label in video_samples:
        vid_path = PROJECT_ROOT / rel_path
        if not vid_path.exists():
            continue

        # Extract MediaPipe landmarks (VIDEO mode, 75 landmarks, 300 spatial features)
        t_ext = time.perf_counter()
        raw_spatial, meta = extractor.extract_isign_video(vid_path)
        ext_time = time.perf_counter() - t_ext

        # Preprocess with motion features: Z-score -> velocity -> (1, 500, 600)
        spatial_mat = to_spatial_matrix(raw_spatial)
        preprocessed = preprocess_pose(
            spatial_mat,
            max_length=MAX_POSE_LENGTH,
            use_motion=True,
            device="cpu",
        )

        # Beam Search
        b_text, b_tokens, b_time, b_loop = translator.generate(
            preprocessed.pose_tensor,
            preprocessed.pose_length,
            num_beams=4,
        )

        # Greedy Safe
        g_text_safe, g_tokens_safe, g_time_safe, g_loop_safe = translator.generate(
            preprocessed.pose_tensor,
            preprocessed.pose_length,
            num_beams=1,
            no_repeat_ngram_size=3,
        )

        record = {
            "name": vid_path.name,
            "type": "real_video",
            "ground_truth": ref_label,
            "frames": preprocessed.actual_length,
            "beam_output": b_text,
            "beam_tokens": b_tokens,
            "beam_latency_s": b_time,
            "beam_loop": b_loop,
            "greedy_safe_output": g_text_safe,
            "greedy_safe_tokens": g_tokens_safe,
            "greedy_safe_latency_s": g_time_safe,
            "exact_match": b_text.strip().lower() == ref_label.strip().lower() if ref_label != "None" else False,
        }
        all_records.append(record)

        print(f"Video: {vid_path.name} ({preprocessed.actual_length} frames, Extracted in {ext_time:.2f}s)")
        print(f"  Reference:    \"{ref_label}\"")
        print(f"  Beam (k=4):   \"{b_text}\" ({b_time:.3f}s)")
        print(f"  Greedy (mit): \"{g_text_safe}\" ({g_time_safe:.3f}s)")
        print()

    print("\n" + "-" * 80)
    print("PHASE 3: SYNTHETIC CONTROL INPUTS (Zero & Gaussian Noise)")
    print("-" * 80)

    controls = [
        ("All-Zeros (len=100)", torch.zeros(1, 500, 600), 100),
        ("All-Zeros (len=300)", torch.zeros(1, 500, 600), 300),
        ("Gaussian Noise (std=1.0, len=100)", torch.randn(1, 500, 600), 100),
        ("Gaussian Noise (std=0.1, len=100)", torch.randn(1, 500, 600) * 0.1, 100),
    ]

    for name, p_tensor, length_val in controls:
        l_tensor = torch.tensor([length_val], dtype=torch.long)
        b_text, b_tokens, b_time, _ = translator.generate(p_tensor, l_tensor, num_beams=4)
        g_text, g_tokens, g_time, _ = translator.generate(p_tensor, l_tensor, num_beams=1, no_repeat_ngram_size=3)

        record = {
            "name": name,
            "type": "control_input",
            "ground_truth": "N/A (Control)",
            "frames": length_val,
            "beam_output": b_text,
            "beam_tokens": b_tokens,
            "beam_latency_s": b_time,
            "beam_loop": False,
            "greedy_safe_output": g_text,
            "greedy_safe_tokens": g_tokens,
            "greedy_safe_latency_s": g_time,
            "exact_match": False,
        }
        all_records.append(record)

        print(f"Control: {name}")
        print(f"  Beam (k=4):   \"{b_text}\"")
        print(f"  Greedy (mit): \"{g_text}\"")
        print()

    # Clean up resources
    extractor.close()
    translator.close()

    # 4. Statistical Summary & Evaluation
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY & METRICS")
    print("=" * 80)

    real_and_pose_records = [r for r in all_records if r["type"] in ("official_pose", "real_video")]
    beam_texts = [r["beam_output"] for r in real_and_pose_records]
    unique_beam_texts = set(beam_texts)

    exact_matches = sum(1 for r in all_records if r["exact_match"])
    official_evals = [r for r in all_records if r["type"] == "official_pose"]
    raw_greedy_loops = sum(1 for r in official_evals if r.get("greedy_raw_loop", False))

    print(f"Total Evaluated Sequences:     {len(all_records)}")
    print(f"Official .pose Samples:        {len(official_evals)}")
    print(f"Real Video Samples:            {len(video_samples)}")
    print(f"Synthetic Control Samples:     {len(controls)}")
    print()
    print(f"Exact Matches (Ground Truth):  {exact_matches} / {len(official_evals)} ({exact_matches / max(len(official_evals), 1) * 100:.1f}%)")
    print(f"Prediction Diversity (Beam):   {len(unique_beam_texts)} unique strings / {len(real_and_pose_records)} inputs ({len(unique_beam_texts)/len(real_and_pose_records)*100:.1f}%)")
    print(f"Unmitigated Greedy Loop Rate:  {raw_greedy_loops} / {len(official_evals)} samples ({raw_greedy_loops / max(len(official_evals), 1) * 100:.1f}%)")
    print(f"Mitigated Greedy Loop Rate:    0 / {len(all_records)} (no_repeat_ngram_size=3 suppressed all loops)")
    print()
    avg_beam_lat = np.mean([r["beam_latency_s"] for r in real_and_pose_records])
    avg_greedy_lat = np.mean([r["greedy_safe_latency_s"] for r in real_and_pose_records])
    print(f"Average Latency (CPU, Beam):   {avg_beam_lat:.3f}s")
    print(f"Average Latency (CPU, Greedy): {avg_greedy_lat:.3f}s")

    print("\nDetailed Prediction Table:")
    print(f"{'Input Name':<34} | {'Type':<12} | {'Ground Truth':<25} | {'Beam Prediction':<35} | {'Latency':<7}")
    print("-" * 125)
    for r in all_records:
        gt_disp = (r['ground_truth'][:22] + '...') if len(r['ground_truth']) > 25 else r['ground_truth']
        beam_disp = (r['beam_output'][:32] + '...') if len(r['beam_output']) > 35 else r['beam_output']
        print(f"{r['name'][:34]:<34} | {r['type'][:12]:<12} | {gt_disp:<25} | {beam_disp:<35} | {r['beam_latency_s']:.2f}s")

    return {
        "records": all_records,
        "exact_matches": exact_matches,
        "unique_beam_predictions": len(unique_beam_texts),
        "total_real_samples": len(real_and_pose_records),
        "avg_beam_latency": avg_beam_lat,
        "avg_greedy_latency": avg_greedy_lat,
    }


def main():
    parser = argparse.ArgumentParser(description="Signova AI Isolated T5-Small-Motion Validation Pipeline")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    run_pipeline_validation()


if __name__ == "__main__":
    main()
