"""
Detailed diagnostic script for Phase 8.3.3: Bi-GRU + CTC Baseline Error Analysis.
Inspects frame-level logits, softmax distributions, blank vs token probabilities,
training set memorization vs test failure, and camera view differences.
"""

from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from ai.continuous.bigru_ctc import (
    BiGRUCTCModel,
    ContinuousISLDataset,
    decode_greedy,
    compute_levenshtein,
    ID_TO_GLOSS,
    GLOSS_TO_ID,
    BLANK_INDEX,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "islvt_continuous"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "islvt_continuous" / "bigru_ctc_baseline.pt"


def main():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = BiGRUCTCModel(
        **{
            k: v
            for k, v in ckpt["config"].items()
            if k in ["input_dim", "proj_dim", "hidden_dim", "num_layers", "num_classes", "dropout"]
        }
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_ds = ContinuousISLDataset(DATASET_DIR, split="train")
    test_ds = ContinuousISLDataset(DATASET_DIR, split="test")

    print("=" * 80)
    print("1. TRAIN SET RECONSTRUCTION / MEMORIZATION ANALYSIS (24 SAMPLES)")
    print("=" * 80)
    train_exact = 0
    train_pred_counts = defaultdict(int)
    for item in train_ds:
        feat = item["features"].unsqueeze(0)
        length = torch.tensor([item["feature_length"]])
        with torch.no_grad():
            log_probs, sub_len = model(feat, length)
            probs = F.softmax(log_probs.squeeze(1), dim=-1)
            dec = decode_greedy(log_probs, sub_len)[0]

        is_match = dec["tokens"] == item["target_glosses"]
        if is_match:
            train_exact += 1
        for tok in dec["tokens"]:
            train_pred_counts[tok] += 1

        mean_p = probs.mean(dim=0).tolist()
        print(
            f"Train Sent {item['sentence_id']:2d} [{item['camera_view']:5s}]: "
            f"Ref={item['target_glosses']} -> Pred={dec['tokens']} (Match={is_match}) | "
            f"Mean Probs [blank={mean_p[0]:.3f}, i={mean_p[1]:.3f}, go={mean_p[2]:.3f}, eat={mean_p[3]:.3f}, food={mean_p[4]:.3f}]"
        )

    print(f"\nTrain Exact Match: {train_exact}/{len(train_ds)} ({train_exact/len(train_ds)*100:.1f}%)")
    print(f"Train Predicted Token Frequencies: {dict(train_pred_counts)}")

    print("\n" + "=" * 80)
    print("2. TEST SET DETAILED FRAME-LEVEL DIAGNOSTICS (8 SAMPLES)")
    print("=" * 80)
    test_pred_counts = defaultdict(int)
    for item in test_ds:
        feat = item["features"].unsqueeze(0)
        length = torch.tensor([item["feature_length"]])
        with torch.no_grad():
            log_probs, sub_len = model(feat, length)
            probs = F.softmax(log_probs.squeeze(1), dim=-1)
            dec = decode_greedy(log_probs, sub_len)[0]

        for tok in dec["tokens"]:
            test_pred_counts[tok] += 1

        mean_p = probs.mean(dim=0).tolist()
        peak_p = probs.max(dim=0).values.tolist()
        raw_preds = dec["raw_preds"]
        raw_freq = {ID_TO_GLOSS[i]: raw_preds.count(i) for i in range(5) if raw_preds.count(i) > 0}

        print(f"\n--- Test Sent {item['sentence_id']} [{item['camera_view']}] ---")
        print(f"  English: \"{item['english']}\" | Full ISL: \"{item['isl_gloss']}\"")
        print(f"  Target Ref: {item['target_glosses']} (Length {len(item['target_glosses'])})")
        print(f"  Predicted:  {dec['tokens']} (Exact={dec['tokens'] == item['target_glosses']})")
        print(f"  Frames: Original={item['feature_length']}, Subsampled={dec['total_frames']}")
        print(f"  Mean Class Probs: [blank={mean_p[0]:.4f}, i={mean_p[1]:.4f}, go={mean_p[2]:.4f}, eat={mean_p[3]:.4f}, food={mean_p[4]:.4f}]")
        print(f"  Peak Class Probs: [blank={peak_p[0]:.4f}, i={peak_p[1]:.4f}, go={peak_p[2]:.4f}, eat={peak_p[3]:.4f}, food={peak_p[4]:.4f}]")
        print(f"  Raw Frame Pred Counts: {raw_freq}")
        print(f"  CTC Blank % (frame-level): {dec['blank_percentage']:.2f}%")

    print(f"\nTest Predicted Token Frequencies: {dict(test_pred_counts)}")

    print("\n" + "=" * 80)
    print("3. SENTENCE 26 (TRAIN, LENGTH 3: ['eat', 'i', 'eat']) DEEP DIVE")
    print("=" * 80)
    for item in train_ds:
        if item["sentence_id"] == 26:
            feat = item["features"].unsqueeze(0)
            length = torch.tensor([item["feature_length"]])
            with torch.no_grad():
                log_probs, sub_len = model(feat, length)
                probs = F.softmax(log_probs.squeeze(1), dim=-1)
                dec = decode_greedy(log_probs, sub_len)[0]

            raw_preds = dec["raw_preds"]
            raw_tokens = [ID_TO_GLOSS[p] for p in raw_preds]
            print(f"Sent 26 [{item['camera_view']}]: Frames={dec['total_frames']}")
            print(f"  Decoded: {dec['tokens']}")
            # Show segments of consecutive raw predictions
            segments = []
            cur_tok = raw_tokens[0]
            cur_len = 1
            for t in raw_tokens[1:]:
                if t == cur_tok:
                    cur_len += 1
                else:
                    segments.append(f"{cur_tok}x{cur_len}")
                    cur_tok = t
                    cur_len = 1
            segments.append(f"{cur_tok}x{cur_len}")
            print(f"  Raw Prediction Run-Length Segments: {' -> '.join(segments)}")

    print("\n" + "=" * 80)
    print("4. SENTENCE 50 & 72 (TEST, LENGTH 2) DEEP DIVE")
    print("=" * 80)
    for item in test_ds:
        if item["sentence_id"] in [50, 72]:
            feat = item["features"].unsqueeze(0)
            length = torch.tensor([item["feature_length"]])
            with torch.no_grad():
                log_probs, sub_len = model(feat, length)
                probs = F.softmax(log_probs.squeeze(1), dim=-1)
                dec = decode_greedy(log_probs, sub_len)[0]

            raw_preds = dec["raw_preds"]
            raw_tokens = [ID_TO_GLOSS[p] for p in raw_preds]
            segments = []
            cur_tok = raw_tokens[0]
            cur_len = 1
            for t in raw_tokens[1:]:
                if t == cur_tok:
                    cur_len += 1
                else:
                    segments.append(f"{cur_tok}x{cur_len}")
                    cur_tok = t
                    cur_len = 1
            segments.append(f"{cur_tok}x{cur_len}")
            print(f"Sent {item['sentence_id']} [{item['camera_view']}]: Ref={item['target_glosses']} | Decoded={dec['tokens']}")
            print(f"  Raw Prediction Segments: {' -> '.join(segments)}")
            # Check maximum probability attained by the second token
            target2 = item['target_glosses'][1]
            target2_id = GLOSS_TO_ID[target2]
            target2_probs = probs[:, target2_id].tolist()
            print(f"  Target Token 2 ('{target2}') Probs: max={max(target2_probs):.4f}, mean={np.mean(target2_probs):.4f}")


if __name__ == "__main__":
    main()
