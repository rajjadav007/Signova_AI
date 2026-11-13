"""
Confidence and Margin Analysis on In-Vocabulary and Out-of-Vocabulary Samples.
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
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    LANDMARKS_DIR,
    SPLITS_DIR,
)
from ai.isolated.features.feature_extractor import extract_isolated_features
from ai.isolated.inference.classifier import IsolatedSignRecognizer
from ai.isolated.preprocessing.normalizer import spatial_normalize, temporal_resample

rec = IsolatedSignRecognizer(checkpoint_path=ISOLATED_CHECKPOINT_PATH)
rec.load_model()

# 1. Analyze Test Split
with open(SPLITS_DIR / "test.csv", mode="r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

test_stats = []
for r in rows:
    vid = r["video_id"]
    gt = r["class_label"]
    d = np.load(LANDMARKS_DIR / "test" / f"{vid}.npz")
    kps = d["landmarks"]

    normed = spatial_normalize(kps)
    resampled = temporal_resample(normed, target_length=64)
    feats = extract_isolated_features(resampled)
    probs = rec.model.predict_proba(feats[np.newaxis, ...])[0]
    
    top2_idx = np.argsort(probs)[-2:]
    p1 = float(probs[top2_idx[1]])
    p2 = float(probs[top2_idx[0]])
    pred_class = ISOLATED_CLASSES[top2_idx[1]]
    margin = p1 - p2
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

    # Hand velocity energy: slice 33..75 velocity
    hand_vel_energy = float(np.mean(np.linalg.norm(feats[:, 300 + 33*4 : 300 + 75*4].reshape(64, 42, 4)[..., :3], axis=-1)))

    is_corr = (pred_class == gt)
    test_stats.append({
        "vid": vid,
        "gt": gt,
        "pred": pred_class,
        "p1": p1,
        "p2": p2,
        "margin": margin,
        "entropy": entropy,
        "hand_energy": hand_vel_energy,
        "is_correct": is_corr,
    })

print("=== IN-VOCABULARY TEST SAMPLES CONFIDENCE & MARGIN ===")
corr_p1 = [s["p1"] for s in test_stats if s["is_correct"]]
incorr_p1 = [s["p1"] for s in test_stats if not s["is_correct"]]
corr_margin = [s["margin"] for s in test_stats if s["is_correct"]]
incorr_margin = [s["margin"] for s in test_stats if not s["is_correct"]]
corr_ent = [s["entropy"] for s in test_stats if s["is_correct"]]
incorr_ent = [s["entropy"] for s in test_stats if not s["is_correct"]]

print(f"Correct predictions (N={len(corr_p1)}): Mean p1 = {np.mean(corr_p1):.3f}, Mean margin = {np.mean(corr_margin):.3f}, Mean entropy = {np.mean(corr_ent):.3f}")
print(f"Incorrect predictions (N={len(incorr_p1)}): Mean p1 = {np.mean(incorr_p1):.3f}, Mean margin = {np.mean(incorr_margin):.3f}, Mean entropy = {np.mean(incorr_ent):.3f}")

# 2. Analyze Out-of-Vocabulary Samples from Cached Landmarks
cache_dir = PROJECT_ROOT / "datasets" / ".oov_cache"
oov_stats = []
if cache_dir.exists():
    for fpath in cache_dir.glob("*.npz"):
        data = np.load(fpath)
        kps = data["kps"]
        normed = spatial_normalize(kps)
        resampled = temporal_resample(normed, target_length=64)
        feats = extract_isolated_features(resampled)
        probs = rec.model.predict_proba(feats[np.newaxis, ...])[0]
        top2_idx = np.argsort(probs)[-2:]
        p1 = float(probs[top2_idx[1]])
        p2 = float(probs[top2_idx[0]])
        margin = p1 - p2
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        hand_vel_energy = float(np.mean(np.linalg.norm(feats[:, 300 + 33*4 : 300 + 75*4].reshape(64, 42, 4)[..., :3], axis=-1)))
        
        oov_stats.append({
            "name": fpath.stem,
            "pred": ISOLATED_CLASSES[top2_idx[1]],
            "p1": p1,
            "margin": margin,
            "entropy": entropy,
            "hand_energy": hand_vel_energy,
        })

print("\n=== OUT-OF-VOCABULARY SAMPLES CONFIDENCE & MARGIN ===")
for o in oov_stats:
    print(f"OOV: {o['name']:<20} | Pred: {o['pred']:<10} | p1: {o['p1']:.3f} | margin: {o['margin']:.3f} | entropy: {o['entropy']:.3f} | hand_energy: {o['hand_energy']:.4f}")

oov_p1 = [o["p1"] for o in oov_stats]
oov_margin = [o["margin"] for o in oov_stats]
print(f"\nOOV Summary (N={len(oov_stats)}): Mean p1 = {np.mean(oov_p1):.3f}, Mean margin = {np.mean(oov_margin):.3f}")
