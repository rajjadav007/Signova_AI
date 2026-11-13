"""
Audit script for F-033: Final Reliability Audit for Signova AI.
Read-only script to audit F-032 claims, metrics, and experiments.
"""

import csv
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from ai.isolated.evaluation.metrics import compute_isolated_metrics
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.inference.classifier import IsolatedSignRecognizer
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample


def task1_verify_splits():
    print("=== TASK 1: SPLIT INTEGRITY & DATA LEAKAGE AUDIT ===")
    splits = ["train", "validation", "test"]
    split_ids = {}
    split_files = {}

    for s in splits:
        p = SPLITS_DIR / f"{s}.csv"
        with open(p, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        split_ids[s] = set(r["video_id"] for r in rows)
        split_files[s] = set(r["relative_video_path"] for r in rows)
        print(f"Split {s:<10}: {len(rows)} clips ({len(split_ids[s])} unique IDs)")

    # Check overlaps
    train_val_overlap = split_ids["train"].intersection(split_ids["validation"])
    train_test_overlap = split_ids["train"].intersection(split_ids["test"])
    val_test_overlap = split_ids["validation"].intersection(split_ids["test"])

    print(f"Overlap Train & Val:  {len(train_val_overlap)}")
    print(f"Overlap Train & Test: {len(train_test_overlap)}")
    print(f"Overlap Val & Test:   {len(val_test_overlap)}")

    file_train_val = split_files["train"].intersection(split_files["validation"])
    file_train_test = split_files["train"].intersection(split_files["test"])
    file_val_test = split_files["validation"].intersection(split_files["test"])
    print(f"File Overlap Train & Val:  {len(file_train_val)}")
    print(f"File Overlap Train & Test: {len(file_train_test)}")
    print(f"File Overlap Val & Test:   {len(file_val_test)}")

    # Class mappings check
    print("\nClass mappings check:")
    for idx, c in IDX_TO_CLASS.items():
        assert CLASS_TO_IDX[c] == idx
    print(f"Total target classes: {len(ISOLATED_CLASSES)}: {ISOLATED_CLASSES}")


def task1_verify_test_metrics(rec: IsolatedSignRecognizer):
    print("\n=== TASK 1: VERIFY TEST METRICS ON 30 HELD-OUT CLIPS ===")
    with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    y_true = []
    y_pred = []
    y_probs = []
    per_sample = []

    for r in rows:
        vid = r["video_id"]
        gt = r["class_label"]
        npz_p = LANDMARKS_DIR / "test" / f"{vid}.npz"
        d = np.load(npz_p)
        kps = d["landmarks"]

        res = rec.predict_landmarks(kps, top_k=10)
        p_sign = res["predicted_sign"]
        conf = res["confidence"]
        status = res["status"]

        gt_idx = CLASS_TO_IDX[gt]
        pred_idx = CLASS_TO_IDX[p_sign]
        y_true.append(gt_idx)
        y_pred.append(pred_idx)

        prob_vec = np.zeros(10, dtype=np.float32)
        for p in res["top_predictions"]:
            prob_vec[CLASS_TO_IDX[p["sign"]]] = p["confidence"]
        y_probs.append(prob_vec)

        corr = (p_sign == gt)
        per_sample.append({
            "video_id": vid,
            "gt": gt,
            "pred": p_sign,
            "conf": conf,
            "status": status,
            "corr": corr,
            "signer": r["signer_id"],
            "source": r["source_dataset"],
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    metrics = compute_isolated_metrics(y_true, y_pred, y_probs, target_names=ISOLATED_CLASSES)
    print(f"Overall Top-1 Accuracy: {metrics['accuracy']:.4f} ({np.sum(y_true == y_pred)}/{len(y_true)})")
    print(f"Overall Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    print(f"Macro Precision:        {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:           {metrics['macro_recall']:.4f}")
    print(f"Macro F1:               {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:            {metrics['weighted_f1']:.4f}")

    print("\nPer-class Metrics:")
    for c in ISOLATED_CLASSES:
        v = metrics["per_class"][c]
        print(f"  {c:<12} | P: {v['precision']:.2f} | R: {v['recall']:.2f} | F1: {v['f1-score']:.2f} | Support: {int(v['support'])}")

    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print("      " + " ".join(f"{c[:4]:>5}" for c in ISOLATED_CLASSES))
    for i, row in enumerate(cm):
        print(f"{ISOLATED_CLASSES[i][:5]:<5} " + " ".join(f"{x:>5}" for x in row))

    return per_sample, y_probs, y_true, y_pred


def task2_verify_handedness(rec: IsolatedSignRecognizer):
    print("\n=== TASK 2: HANDEDNESS DETERMINATION & MIRRORING AUDIT ===")
    
    # Check splits
    for s in ["train", "validation", "test"]:
        p = SPLITS_DIR / f"{s}.csv"
        with open(p, mode="r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        lh_dom = []
        rh_dom = []
        two_h = []
        no_h = []

        for r in rows:
            vid = r["video_id"]
            d = np.load(LANDMARKS_DIR / s / f"{vid}.npz")
            kps = d["landmarks"]
            lh_act = float(np.mean(kps[:, 33:54, 3] > 0.05))
            rh_act = float(np.mean(kps[:, 54:75, 3] > 0.05))

            if lh_act > 0.10 and rh_act > 0.10:
                two_h.append((vid, lh_act, rh_act))
            elif rh_act > 0.10:
                rh_dom.append((vid, lh_act, rh_act))
            elif lh_act > 0.10:
                lh_dom.append((vid, lh_act, rh_act))
            else:
                no_h.append((vid, lh_act, rh_act))

        print(f"Split {s:<10} | Total: {len(rows):>3} | RH dominant: {len(rh_dom):>3} | LH dominant: {len(lh_dom):>3} | Two-handed: {len(two_h):>3} | No hand: {len(no_h):>3}")
        if s == "test":
            print(f"  Test LH dominant clips ({len(lh_dom)}): {[x[0] for x in lh_dom]}")

    # Reproduce Mirroring Experiment
    POSE_SWAP_PAIRS = [
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
        (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
    ]

    def mirror_x_only(kps: np.ndarray) -> np.ndarray:
        m = kps.copy()
        m[..., 0] = -m[..., 0]
        return m

    def mirror_x_and_swap(kps: np.ndarray) -> np.ndarray:
        m = kps.copy()
        m[..., 0] = -m[..., 0]
        for l, r in POSE_SWAP_PAIRS:
            tmp = m[:, l, :].copy()
            m[:, l, :] = m[:, r, :]
            m[:, r, :] = tmp
        tmp_lh = m[:, 33:54, :].copy()
        m[:, 33:54, :] = m[:, 54:75, :]
        m[:, 54:75, :] = tmp_lh
        return m

    with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print("\n--- REPRODUCING MIRRORING EXPERIMENT ON TEST SET ---")
    cond1_preds = []  # Original
    cond2_preds = []  # Mirror X only on LH-dominant
    cond3_preds = []  # Mirror X + swap on LH-dominant

    affected_clips = []

    for r in test_rows:
        vid = r["video_id"]
        gt = r["class_label"]
        signer = r["signer_id"]
        d = np.load(LANDMARKS_DIR / "test" / f"{vid}.npz")
        kps = d["landmarks"]

        lh_act = float(np.mean(kps[:, 33:54, 3] > 0.05))
        rh_act = float(np.mean(kps[:, 54:75, 3] > 0.05))
        is_lh_dominant = (lh_act > 0.10 and rh_act <= 0.10)

        p1 = rec.predict_landmarks(kps)
        cond1_preds.append((p1["predicted_sign"] == gt, p1["predicted_sign"], p1["confidence"]))

        if is_lh_dominant:
            # Condition 2: Mirror X only
            p2 = rec.predict_landmarks(mirror_x_only(kps))
            # Condition 3: Mirror X + Swap
            p3 = rec.predict_landmarks(mirror_x_and_swap(kps))

            affected_clips.append({
                "video_id": vid,
                "gt": gt,
                "signer": signer,
                "lh_act": round(lh_act, 3),
                "rh_act": round(rh_act, 3),
                "orig": (p1["predicted_sign"], p1["confidence"], p1["predicted_sign"] == gt),
                "mirror_x_only": (p2["predicted_sign"], p2["confidence"], p2["predicted_sign"] == gt),
                "mirror_swap": (p3["predicted_sign"], p3["confidence"], p3["predicted_sign"] == gt),
            })
            cond2_preds.append((p2["predicted_sign"] == gt, p2["predicted_sign"], p2["confidence"]))
            cond3_preds.append((p3["predicted_sign"] == gt, p3["predicted_sign"], p3["confidence"]))
        else:
            cond2_preds.append((p1["predicted_sign"] == gt, p1["predicted_sign"], p1["confidence"]))
            cond3_preds.append((p1["predicted_sign"] == gt, p1["predicted_sign"], p1["confidence"]))

    c1_corr = sum(1 for x in cond1_preds if x[0])
    c2_corr = sum(1 for x in cond2_preds if x[0])
    c3_corr = sum(1 for x in cond3_preds if x[0])

    print(f"Condition 1 (Original Preprocessing):                    {c1_corr}/30 ({c1_corr/30*100:.2f}%)")
    print(f"Condition 2 (Mirror X only on LH-dominant):               {c2_corr}/30 ({c2_corr/30*100:.2f}%)")
    print(f"Condition 3 (Mirror X + Landmark/Hand Swap on LH-dominant): {c3_corr}/30 ({c3_corr/30*100:.2f}%)")

    # Per-signer breakdown across conditions
    print("\nPer-signer Accuracy Comparison across Conditions:")
    for s_id in sorted(list(set(r["signer_id"] for r in test_rows))):
        s_idx = [i for i, r in enumerate(test_rows) if r["signer_id"] == s_id]
        s1 = sum(1 for i in s_idx if cond1_preds[i][0])
        s2 = sum(1 for i in s_idx if cond2_preds[i][0])
        s3 = sum(1 for i in s_idx if cond3_preds[i][0])
        tot = len(s_idx)
        print(f"  Signer {s_id:<18} (N={tot:>2}): C1={s1}/{tot} ({s1/tot*100:5.1f}%) | C2={s2}/{tot} ({s2/tot*100:5.1f}%) | C3={s3}/{tot} ({s3/tot*100:5.1f}%)")

    print("\nDetailed Prediction Changes for Every Affected Clip (LH-dominant):")
    for a in affected_clips:
        print(f"  {a['video_id']} | GT: {a['gt']:<10} | Signer: {a['signer']} | LH={a['lh_act']:.2f}, RH={a['rh_act']:.2f}")
        print(f"     Cond 1 (Orig):       {a['orig'][0]:<10} (conf={a['orig'][1]:.3f}) | Corr: {a['orig'][2]}")
        print(f"     Cond 2 (X-only):     {a['mirror_x_only'][0]:<10} (conf={a['mirror_x_only'][1]:.3f}) | Corr: {a['mirror_x_only'][2]}")
        print(f"     Cond 3 (X + Swap):   {a['mirror_swap'][0]:<10} (conf={a['mirror_swap'][1]:.3f}) | Corr: {a['mirror_swap'][2]}")


def task3_verify_confidence_and_rejection(rec: IsolatedSignRecognizer):
    print("\n=== TASK 3: CONFIDENCE & REJECTION METRICS AUDIT ===")
    with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    corr_conf = []
    incorr_conf = []
    margins = []
    entropies = []
    corr_margins = []
    incorr_margins = []
    
    accepted_samples = []
    rejected_samples = []

    for r in rows:
        vid = r["video_id"]
        gt = r["class_label"]
        d = np.load(LANDMARKS_DIR / "test" / f"{vid}.npz")
        res = rec.predict_landmarks(d["landmarks"], top_k=10)
        
        conf = res["confidence"]
        top_preds = res["top_predictions"]
        p1 = top_preds[0]["confidence"]
        p2 = top_preds[1]["confidence"] if len(top_preds) > 1 else 0.0
        margin = p1 - p2
        margins.append(margin)

        probs = np.array([p["confidence"] for p in top_preds])
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        entropies.append(entropy)

        is_corr = (res["predicted_sign"] == gt)
        if is_corr:
            corr_conf.append(conf)
            corr_margins.append(margin)
        else:
            incorr_conf.append(conf)
            incorr_margins.append(margin)

        if conf >= 0.80:
            accepted_samples.append((vid, gt, res["predicted_sign"], conf, is_corr))
        else:
            rejected_samples.append((vid, gt, res["predicted_sign"], conf, is_corr))

    print(f"Mean Confidence (Correct, N={len(corr_conf)}):     {np.mean(corr_conf):.4f}")
    print(f"Mean Confidence (Incorrect, N={len(incorr_conf)}): {np.mean(incorr_conf):.4f}")
    print(f"Mean Top-1 vs Top-2 Margin (Overall):              {np.mean(margins):.4f}")
    print(f"Mean Margin (Correct):                             {np.mean(corr_margins):.4f}")
    print(f"Mean Margin (Incorrect):                           {np.mean(incorr_margins):.4f}")
    print(f"Mean Entropy (Overall):                            {np.mean(entropies):.4f}")

    print(f"\nAt Threshold tau = 0.80:")
    print(f"  Total Samples:        {len(rows)}")
    print(f"  Accepted Samples:     {len(accepted_samples)} (Coverage: {len(accepted_samples)/len(rows)*100:.2f}%)")
    acc_corr = sum(1 for x in accepted_samples if x[4])
    acc_incorr = sum(1 for x in accepted_samples if not x[4])
    print(f"  Accuracy on Accepted: {acc_corr}/{len(accepted_samples)} ({acc_corr/len(accepted_samples)*100:.2f}%)")
    print(f"  Accepted Errors:      {acc_incorr}")

    print(f"  Rejected Samples:     {len(rejected_samples)} (Rejection Rate: {len(rejected_samples)/len(rows)*100:.2f}%)")
    rej_corr = sum(1 for x in rejected_samples if x[4])
    rej_incorr = sum(1 for x in rejected_samples if not x[4])
    print(f"  Rejected Correct:     {rej_corr}")
    print(f"  Rejected Errors:      {rej_incorr}")
    print(f"  Verification of claim 'Zero correct predictions were rejected': {rej_corr == 0}")
    if rej_corr > 0:
        print(f"    WARNING: Rejected correct samples: {[x for x in rejected_samples if x[4]]}")
    print(f"  Rejected Samples Details:")
    for x in rejected_samples:
        print(f"    {x[0]} | GT: {x[1]:<10} | Pred: {x[2]:<10} | Conf: {x[3]:.3f} | Correct: {x[4]}")


def task4_audit_motion_energy(rec: IsolatedSignRecognizer):
    print("\n=== TASK 4: MOTION-ENERGY REJECTION RULES AUDIT ===")
    with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Evaluate on test set
    test_evals = []
    for r in rows:
        vid = r["video_id"]
        gt = r["class_label"]
        d = np.load(LANDMARKS_DIR / "test" / f"{vid}.npz")
        kps = d["landmarks"]

        normed = spatial_normalize(kps)
        resampled = temporal_resample(normed, target_length=64)
        feats = extract_isolated_features(resampled)
        
        # Hand velocity: 300 spatial + 300 velocity. Hands are index 33..74 (42 landmarks)
        # Spatial is feats[:, :300], Velocity is feats[:, 300:600]
        # In 75 landmarks, each landmark has 4 channels [vx, vy, vz, conf]
        # Hands: landmark 33 to 74 -> channels 33*4 to 75*4
        hand_vel = feats[:, 300 + 33*4 : 300 + 75*4].reshape(64, 42, 4)[..., :3]
        hand_energy = float(np.mean(np.linalg.norm(hand_vel, axis=-1)))

        pred = rec.predict_landmarks(kps, top_k=2)
        p1 = pred["top_predictions"][0]["confidence"]
        p2 = pred["top_predictions"][1]["confidence"] if len(pred["top_predictions"]) > 1 else 0.0
        margin = p1 - p2
        is_corr = (pred["predicted_sign"] == gt)

        # Proposed rules:
        # Rule A: Reject if hand_energy < 0.02
        # Rule B: Reject if hand_energy > 0.25
        # Rule C: Reject if margin < 0.25
        rej_low_energy = (hand_energy < 0.02)
        rej_high_energy = (hand_energy > 0.25)
        rej_margin = (margin < 0.25)
        rej_any = rej_low_energy or rej_high_energy or rej_margin

        test_evals.append({
            "vid": vid,
            "gt": gt,
            "pred": pred["predicted_sign"],
            "conf": p1,
            "margin": margin,
            "hand_energy": hand_energy,
            "is_corr": is_corr,
            "rej_low_energy": rej_low_energy,
            "rej_high_energy": rej_high_energy,
            "rej_margin": rej_margin,
            "rej_any": rej_any,
        })

    # Summary on test set
    total_corr = sum(1 for x in test_evals if x["is_corr"])
    total_incorr = sum(1 for x in test_evals if not x["is_corr"])

    low_rej_corr = sum(1 for x in test_evals if x["is_corr"] and x["rej_low_energy"])
    low_rej_incorr = sum(1 for x in test_evals if not x["is_corr"] and x["rej_low_energy"])

    high_rej_corr = sum(1 for x in test_evals if x["is_corr"] and x["rej_high_energy"])
    high_rej_incorr = sum(1 for x in test_evals if not x["is_corr"] and x["rej_high_energy"])

    margin_rej_corr = sum(1 for x in test_evals if x["is_corr"] and x["rej_margin"])
    margin_rej_incorr = sum(1 for x in test_evals if not x["is_corr"] and x["rej_margin"])

    any_rej_corr = sum(1 for x in test_evals if x["is_corr"] and x["rej_any"])
    any_rej_incorr = sum(1 for x in test_evals if not x["is_corr"] and x["rej_any"])

    print(f"Test Set In-Vocab Effect (Total Correct={total_corr}, Total Errors={total_incorr}):")
    print(f"  Rule A (Energy < 0.02): Rejects {low_rej_corr} correct signs, {low_rej_incorr} error signs")
    print(f"  Rule B (Energy > 0.25): Rejects {high_rej_corr} correct signs, {high_rej_incorr} error signs")
    print(f"  Rule C (Margin < 0.25): Rejects {margin_rej_corr} correct signs, {margin_rej_incorr} error signs")
    print(f"  Combined Rules (A||B||C): Rejects {any_rej_corr} correct signs, {any_rej_incorr} error signs")

    # Min, Max, Mean energy on test set
    energies = [x["hand_energy"] for x in test_evals]
    print(f"  Test Hand Energy: Min={np.min(energies):.4f}, Max={np.max(energies):.4f}, Mean={np.mean(energies):.4f}")

    # Check on OOV samples
    cache_dir = PROJECT_ROOT / "datasets" / ".oov_cache"
    print("\nOOV Samples Effect:")
    if cache_dir.exists():
        for fpath in cache_dir.glob("*.npz"):
            kps = np.load(fpath)["kps"]
            normed = spatial_normalize(kps)
            resampled = temporal_resample(normed, target_length=64)
            feats = extract_isolated_features(resampled)
            hand_vel = feats[:, 300 + 33*4 : 300 + 75*4].reshape(64, 42, 4)[..., :3]
            energy = float(np.mean(np.linalg.norm(hand_vel, axis=-1)))
            pred = rec.predict_landmarks(kps, top_k=2)
            p1 = pred["top_predictions"][0]["confidence"]
            p2 = pred["top_predictions"][1]["confidence"] if len(pred["top_predictions"]) > 1 else 0.0
            margin = p1 - p2
            rej = (energy < 0.02) or (energy > 0.25) or (margin < 0.25)
            print(f"  OOV {fpath.stem:<20} | Energy: {energy:.4f} | Margin: {margin:.3f} | Rej: {rej}")


def task5_verify_temporal_preprocessing():
    print("\n=== TASK 5: TEMPORAL PREPROCESSING INFERENCE VS TRAINING MATCH ===")
    from ai.isolated.training.dataset_loader import load_split_dataset
    
    # Load 1 sample via training loader
    X_train, y_train, meta_train = load_split_dataset(split="test", sequence_length=64, from_precomputed_features=False)
    
    # Re-create via inference pipeline
    with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    diffs = []
    for i, r in enumerate(rows):
        vid = r["video_id"]
        d = np.load(LANDMARKS_DIR / "test" / f"{vid}.npz")
        kps = d["landmarks"]

        # Inference preprocessing
        normed = spatial_normalize(kps)
        resampled = temporal_resample(normed, target_length=64)
        feats = extract_isolated_features(resampled)

        # Compare to X_train[i]
        diff = np.max(np.abs(feats - X_train[i]))
        diffs.append(diff)

    max_diff = max(diffs)
    print(f"Maximum difference across all 30 test clips: {max_diff:.8e}")
    print(f"Exact match verified: {max_diff == 0.0}")


if __name__ == "__main__":
    task1_verify_splits()
    rec = IsolatedSignRecognizer(checkpoint_path=ISOLATED_CHECKPOINT_PATH)
    rec.load_model()
    task1_verify_test_metrics(rec)
    task2_verify_handedness(rec)
    task3_verify_confidence_and_rejection(rec)
    task4_audit_motion_energy(rec)
    task5_verify_temporal_preprocessing()
