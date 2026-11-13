"""
F-032 Diagnostic Script: Handedness, Confidence, and Temporal Reliability Analysis.
"""

import csv
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    IDX_TO_CLASS,
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    LANDMARKS_DIR,
    SPLITS_DIR,
)
from ai.isolated.inference.classifier import IsolatedSignRecognizer
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample
from ai.isolated.features.feature_extractor import extract_isolated_features


def analyze_handedness_in_splits():
    """Inspects LH vs RH activity in train, val, and test splits."""
    stats = {}
    for split in ["train", "validation", "test"]:
        split_csv = SPLITS_DIR / f"{split}.csv"
        with open(split_csv, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        lh_dominant = 0
        rh_dominant = 0
        two_handed = 0
        no_hand = 0

        for r in rows:
            vid = r["video_id"]
            npz_path = LANDMARKS_DIR / split / f"{vid}.npz"
            if not npz_path.exists():
                continue
            d = np.load(npz_path)
            kps = d["landmarks"]  # (T, 75, 4)
            # LH: 33..53, RH: 54..74
            lh_conf = np.mean(kps[:, 33:54, 3] > 0.05)
            rh_conf = np.mean(kps[:, 54:75, 3] > 0.05)

            if lh_conf > 0.1 and rh_conf > 0.1:
                two_handed += 1
            elif rh_conf > 0.1:
                rh_dominant += 1
            elif lh_conf > 0.1:
                lh_dominant += 1
            else:
                no_hand += 1

        stats[split] = {
            "total": len(rows),
            "rh_dominant": rh_dominant,
            "lh_dominant": lh_dominant,
            "two_handed": two_handed,
            "no_hand": no_hand,
        }
    return stats


def test_handedness_canonicalization(recognizer: IsolatedSignRecognizer):
    """
    Evaluates effect of canonical hand mirroring on left-handed signers.
    For horizontal mirroring:
      - X coordinate is flipped: x' = -x
      - Left hand (33..53) and Right hand (54..74) are swapped.
      - Pose symmetric keypoints (e.g. left vs right shoulder/elbow/wrist) are swapped.
    """
    with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Pose pairs to swap on horizontal flip:
    # (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    # (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    POSE_SWAP_PAIRS = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
        (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]

    def mirror_sequence(kps: np.ndarray) -> np.ndarray:
        mirrored = kps.copy()
        # Flip X coordinate
        mirrored[..., 0] = -mirrored[..., 0]
        # Swap pose left/right
        for left_idx, right_idx in POSE_SWAP_PAIRS:
            tmp = mirrored[:, left_idx, :].copy()
            mirrored[:, left_idx, :] = mirrored[:, right_idx, :]
            mirrored[:, right_idx, :] = tmp
        # Swap Left Hand (33..54) and Right Hand (54..75)
        tmp_lh = mirrored[:, 33:54, :].copy()
        mirrored[:, 33:54, :] = mirrored[:, 54:75, :]
        mirrored[:, 54:75, :] = tmp_lh
        return mirrored

    results = []
    for r in rows:
        vid = r["video_id"]
        gt = r["class_label"]
        signer = r["signer_id"]
        d = np.load(LANDMARKS_DIR / "test" / f"{vid}.npz")
        kps = d["landmarks"]

        # Check which hand is active
        lh_act = float(np.mean(kps[:, 33:54, 3] > 0.05))
        rh_act = float(np.mean(kps[:, 54:75, 3] > 0.05))

        # Original prediction
        orig_pred = recognizer.predict_landmarks(kps)
        
        # Mirrored prediction
        mirrored_kps = mirror_sequence(kps)
        mirrored_pred = recognizer.predict_landmarks(mirrored_kps)

        # Smart / Canonical prediction:
        # If LH is active and RH is inactive (lh_act > 0.1 and rh_act <= 0.05), use mirrored!
        if lh_act > 0.10 and rh_act <= 0.05:
            smart_pred = mirrored_pred
            canonical_used = "mirrored"
        else:
            smart_pred = orig_pred
            canonical_used = "original"

        results.append({
            "video_id": vid,
            "gt": gt,
            "signer": signer,
            "lh_act": round(lh_act, 3),
            "rh_act": round(rh_act, 3),
            "orig_pred": orig_pred["predicted_sign"],
            "orig_conf": orig_pred["confidence"],
            "orig_correct": (orig_pred["predicted_sign"] == gt),
            "mirrored_pred": mirrored_pred["predicted_sign"],
            "mirrored_conf": mirrored_pred["confidence"],
            "mirrored_correct": (mirrored_pred["predicted_sign"] == gt),
            "smart_pred": smart_pred["predicted_sign"],
            "smart_conf": smart_pred["confidence"],
            "smart_correct": (smart_pred["predicted_sign"] == gt),
            "canonical_used": canonical_used,
        })

    return results


def main():
    print("=== PRIORITY 1: HANDEDNESS DISTRIBUTION IN SPLITS ===")
    stats = analyze_handedness_in_splits()
    for s, v in stats.items():
        print(f"Split: {s:<10} | Total: {v['total']:>3} | RH dominant: {v['rh_dominant']:>3} | LH dominant: {v['lh_dominant']:>3} | Two-handed: {v['two_handed']:>3} | No hand: {v['no_hand']:>3}")

    recognizer = IsolatedSignRecognizer(checkpoint_path=ISOLATED_CHECKPOINT_PATH)
    recognizer.load_model()

    print("\n=== EVALUATING HANDEDNESS CANONICALIZATION ON TEST SET ===")
    results = test_handedness_canonicalization(recognizer)

    orig_correct = sum(1 for r in results if r["orig_correct"])
    smart_correct = sum(1 for r in results if r["smart_correct"])
    print(f"Original Test Accuracy: {orig_correct}/{len(results)} ({orig_correct/len(results)*100:.2f}%)")
    print(f"Canonical Test Accuracy: {smart_correct}/{len(results)} ({smart_correct/len(results)*100:.2f}%)")

    user014_res = [r for r in results if r["signer"] == "User014"]
    u14_orig = sum(1 for r in user014_res if r["orig_correct"])
    u14_smart = sum(1 for r in user014_res if r["smart_correct"])
    print(f"\nUser014 (Left-Handed Signer) Results:")
    print(f"Original User014: {u14_orig}/{len(user014_res)} ({u14_orig/len(user014_res)*100:.2f}%)")
    print(f"Canonical User014: {u14_smart}/{len(user014_res)} ({u14_smart/len(user014_res)*100:.2f}%)")
    for r in user014_res:
        print(f"  {r['video_id']} | GT: {r['gt']:<10} | Orig: {r['orig_pred']:<10} ({r['orig_conf']:.2f}) -> Mirrored: {r['mirrored_pred']:<10} ({r['mirrored_conf']:.2f}) | Corr: {r['smart_correct']}")

    rh_res = [r for r in results if r["signer"] != "User014"]
    rh_orig = sum(1 for r in rh_res if r["orig_correct"])
    rh_smart = sum(1 for r in rh_res if r["smart_correct"])
    print(f"\nRight-Handed Signers (22 clips) Integrity Check:")
    print(f"Original RH: {rh_orig}/{len(rh_res)} ({rh_orig/len(rh_res)*100:.2f}%)")
    print(f"Canonical RH: {rh_smart}/{len(rh_res)} ({rh_smart/len(rh_res)*100:.2f}%)")


if __name__ == "__main__":
    main()
