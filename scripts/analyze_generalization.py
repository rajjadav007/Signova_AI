"""
F-023: Generalization Analysis, Bilateral Model Training & Benchmark Comparison.
Preserves original F-022 checkpoint and saves improved model to:
  models/checkpoints/isolated/isolated_10class_improved.joblib
Outputs:
  - models/checkpoints/isolated/generalization_analysis.json
  - docs/reports/figures/generalization_gap_comparison.png
  - docs/reports/figures/signer_breakdown_comparison.png
  - docs/reports/figures/confusion_matrix_improved_test.png
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

# Matplotlib headless config
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "C:/tmp")) / "matplotlib_signova"))
import matplotlib.pyplot as plt

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    IDX_TO_CLASS,
    ISOLATED_CHECKPOINT_DIR,
    ISOLATED_CHECKPOINT_PATH,
    ISOLATED_CLASSES,
    NUM_CLASSES,
)
from ai.isolated.evaluation.metrics import compute_isolated_metrics
from ai.isolated.training.dataset_loader import load_split_dataset
from scripts.experiment_generalization import build_bilateral_train_data, train_custom_gru

FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"
IMPROVED_CHECKPOINT_PATH = ISOLATED_CHECKPOINT_DIR / "isolated_10class_improved.joblib"


def run_full_analysis():
    print("=" * 80, flush=True)
    print("F-023: GENERALIZATION ANALYSIS & MODEL IMPROVEMENT PIPELINE", flush=True)
    print("=" * 80, flush=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    print("\n[STEP 1] Loading precomputed datasets...", flush=True)
    X_train_orig, y_train_orig, meta_train = load_split_dataset("train", from_precomputed_features=True)
    X_train_bilateral, y_train_bilateral = build_bilateral_train_data()
    X_val, y_val, meta_val = load_split_dataset("validation", from_precomputed_features=True)
    X_test, y_test, meta_test = load_split_dataset("test", from_precomputed_features=True)

    print(f"  Train (Original):  {X_train_orig.shape}", flush=True)
    print(f"  Train (Bilateral): {X_train_bilateral.shape} (129 right-handed + 129 left-handed mirrored)", flush=True)
    print(f"  Validation:        {X_val.shape}", flush=True)
    print(f"  Test:              {X_test.shape}", flush=True)

    # 2. Load original baseline checkpoint
    print("\n[STEP 2] Loading Baseline Model (F-022)...", flush=True)
    assert ISOLATED_CHECKPOINT_PATH.exists(), f"Missing baseline checkpoint: {ISOLATED_CHECKPOINT_PATH}"
    baseline_model = joblib.load(ISOLATED_CHECKPOINT_PATH)

    # 3. Train Improved Model with Bilateral Augmentation
    print("\n[STEP 3] Training Improved Model with Bilateral Augmentation (Exp 2)...", flush=True)
    improved_model, improved_val_metrics, train_time = train_custom_gru(
        X_train=X_train_bilateral,
        y_train=y_train_bilateral,
        X_val=X_val,
        y_val=y_val,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        weight_decay=1e-4,
        epochs=60,
        patience=15,
        random_seed=42,
    )
    print(f"  Trained in {train_time}s | Val Acc: {improved_val_metrics['accuracy']:.4f} | Val Macro F1: {improved_val_metrics['macro_f1']:.4f}", flush=True)

    # 4. Comprehensive Evaluation on Validation and Held-out Test
    print("\n[STEP 4] Evaluating Baseline vs Improved on Validation & Test...", flush=True)

    # Baseline predictions
    base_val_probs = baseline_model.predict_proba(X_val)
    base_val_preds = np.argmax(base_val_probs, axis=1)
    base_val_m = compute_isolated_metrics(y_val, base_val_preds, y_probs=base_val_probs)

    base_test_probs = baseline_model.predict_proba(X_test)
    base_test_preds = np.argmax(base_test_probs, axis=1)
    base_test_m = compute_isolated_metrics(y_test, base_test_preds, y_probs=base_test_probs)

    # Improved predictions
    imp_val_probs = improved_model.predict_proba(X_val)
    imp_val_preds = np.argmax(imp_val_probs, axis=1)
    imp_val_m = compute_isolated_metrics(y_val, imp_val_preds, y_probs=imp_val_probs)

    imp_test_probs = improved_model.predict_proba(X_test)
    imp_test_preds = np.argmax(imp_test_probs, axis=1)
    imp_test_m = compute_isolated_metrics(y_test, imp_test_preds, y_probs=imp_test_probs)

    # Print Comparison Table
    print("\n" + "=" * 80, flush=True)
    print("MODEL PERFORMANCE COMPARISON: BASELINE vs IMPROVED (BILATERAL)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Metric':<22} {'Baseline (Val)':<16} {'Improved (Val)':<16} {'Baseline (Test)':<16} {'Improved (Test)':<16}", flush=True)
    print("-" * 86, flush=True)
    print(f"{'Accuracy':<22} {base_val_m['accuracy']:<16.4f} {imp_val_m['accuracy']:<16.4f} {base_test_m['accuracy']:<16.4f} {imp_test_m['accuracy']:<16.4f}", flush=True)
    print(f"{'Macro Precision':<22} {base_val_m['macro_precision']:<16.4f} {imp_val_m['macro_precision']:<16.4f} {base_test_m['macro_precision']:<16.4f} {imp_test_m['macro_precision']:<16.4f}", flush=True)
    print(f"{'Macro Recall':<22} {base_val_m['macro_recall']:<16.4f} {imp_val_m['macro_recall']:<16.4f} {base_test_m['macro_recall']:<16.4f} {imp_test_m['macro_recall']:<16.4f}", flush=True)
    print(f"{'Macro F1':<22} {base_val_m['macro_f1']:<16.4f} {imp_val_m['macro_f1']:<16.4f} {base_test_m['macro_f1']:<16.4f} {imp_test_m['macro_f1']:<16.4f}", flush=True)
    print(f"{'Weighted F1':<22} {base_val_m['weighted_f1']:<16.4f} {imp_val_m['weighted_f1']:<16.4f} {base_test_m['weighted_f1']:<16.4f} {imp_test_m['weighted_f1']:<16.4f}", flush=True)
    print(f"{'Top-3 Accuracy':<22} {base_val_m['top3_accuracy']:<16.4f} {imp_val_m['top3_accuracy']:<16.4f} {base_test_m['top3_accuracy']:<16.4f} {imp_test_m['top3_accuracy']:<16.4f}", flush=True)

    # Generalization gap calculation
    base_gap = base_val_m["accuracy"] - base_test_m["accuracy"]
    imp_gap = imp_val_m["accuracy"] - imp_test_m["accuracy"]
    print(f"{'Generalization Gap':<22} {'-':<16} {'-':<16} {base_gap:<16.4f} {imp_gap:<16.4f}", flush=True)

    # Signer breakdown on Test Set
    print("\n" + "=" * 80, flush=True)
    print("SIGNER BREAKDOWN ON HELD-OUT TEST SPLIT (N=30)", flush=True)
    print("=" * 80, flush=True)
    signers = sorted(list(set(m["signer_id"] for m in meta_test)))
    print(f"{'Signer ID':<22} {'Dominance':<16} {'Clips':<8} {'Baseline Acc':<16} {'Improved Acc':<16}", flush=True)
    print("-" * 78, flush=True)

    signer_stats = {}
    for s in signers:
        indices = [i for i, m in enumerate(meta_test) if m["signer_id"] == s]
        n_clips = len(indices)
        base_correct = sum(1 for i in indices if base_test_preds[i] == y_test[i])
        imp_correct = sum(1 for i in indices if imp_test_preds[i] == y_test[i])
        b_acc = base_correct / n_clips
        i_acc = imp_correct / n_clips

        dom = "Left-handed" if s == "User014" else ("Right-handed" if s.startswith("User") else "Formal Dict / Mixed")
        print(f"{s:<22} {dom:<16} {n_clips:<8} {b_acc:<16.1%} {i_acc:<16.1%}", flush=True)
        signer_stats[s] = {
            "dominance": dom,
            "clips": n_clips,
            "baseline_accuracy": round(b_acc, 4),
            "improved_accuracy": round(i_acc, 4),
        }

    # Per-class breakdown on Test Set
    print("\n" + "=" * 80, flush=True)
    print("PER-CLASS TEST BREAKDOWN: BASELINE vs IMPROVED", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Class':<14} {'Support':<8} {'Base Prec':<12} {'Base Rec':<12} {'Base F1':<12} {'Imp Prec':<12} {'Imp Rec':<12} {'Imp F1':<12}", flush=True)
    print("-" * 86, flush=True)
    for cls in ISOLATED_CLASSES:
        bp = base_test_m["per_class"][cls]
        ip = imp_test_m["per_class"][cls]
        supp = int(bp["support"])
        print(f"{cls:<14} {supp:<8} {bp['precision']:<12.4f} {bp['recall']:<12.4f} {bp['f1-score']:<12.4f} {ip['precision']:<12.4f} {ip['recall']:<12.4f} {ip['f1-score']:<12.4f}", flush=True)

    # 5. Low-data class analysis
    print("\n" + "=" * 80, flush=True)
    print("LOW-DATA CLASSES vs MAJORITY CLASSES (TEST SPLIT)", flush=True)
    print("=" * 80, flush=True)
    for c in ["stop", "sorry"]:
        bp = base_test_m["per_class"][c]
        ip = imp_test_m["per_class"][c]
        print(f"Minority '{c}': Base F1 = {bp['f1-score']:.4f} -> Improved F1 = {ip['f1-score']:.4f} (Support = {int(bp['support'])})", flush=True)

    maj_classes = [c for c in ISOLATED_CLASSES if c not in ("stop", "sorry")]
    base_maj_f1 = np.mean([base_test_m["per_class"][c]["f1-score"] for c in maj_classes])
    imp_maj_f1 = np.mean([imp_test_m["per_class"][c]["f1-score"] for c in maj_classes])
    print(f"8 Majority Classes Average: Base F1 = {base_maj_f1:.4f} -> Improved F1 = {imp_maj_f1:.4f}", flush=True)

    # 6. Confidence Calibration & Rejection Analysis
    print("\n" + "=" * 80, flush=True)
    print("CONFIDENCE CALIBRATION & REJECTION ANALYSIS (IMPROVED MODEL)", flush=True)
    print("=" * 80, flush=True)

    # Multi-class Brier score
    y_val_one_hot = np.zeros_like(imp_val_probs)
    for i, label in enumerate(y_val):
        y_val_one_hot[i, label] = 1.0
    imp_brier = float(np.mean(np.sum((imp_val_probs - y_val_one_hot) ** 2, axis=1)))
    print(f"Improved Model Validation Brier Score: {imp_brier:.4f}", flush=True)

    # Threshold sweep on Validation
    val_max_conf = np.max(imp_val_probs, axis=1)
    rejection_sweep = []
    print(f"\n{'Threshold (tau)':<16} {'Coverage %':<12} {'Accepted':<10} {'Rejected':<10} {'Accepted Acc':<14}", flush=True)
    print("-" * 62, flush=True)
    for tau in np.arange(0.20, 0.85, 0.05):
        acc_mask = val_max_conf >= tau
        n_acc = int(np.sum(acc_mask))
        n_rej = int(np.sum(~acc_mask))
        cov = round(n_acc / len(y_val) * 100.0, 1)
        acc = round(float(np.mean(imp_val_preds[acc_mask] == y_val[acc_mask])), 4) if n_acc > 0 else 1.0
        rejection_sweep.append({
            "threshold": round(float(tau), 2),
            "coverage_pct": cov,
            "accepted_count": n_acc,
            "rejected_count": n_rej,
            "accuracy_on_accepted": acc,
        })
        print(f"{tau:<16.2f} {cov:<12.1f}% {n_acc:<10} {n_rej:<10} {acc:<14.4f}", flush=True)

    # 7. Save Improved Model Checkpoint (preserving original)
    joblib.dump(improved_model, IMPROVED_CHECKPOINT_PATH)
    imp_size_mb = round(IMPROVED_CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 2)
    print(f"\nSaved improved model checkpoint to: {IMPROVED_CHECKPOINT_PATH} ({imp_size_mb} MB)", flush=True)

    # Save complete analysis JSON
    analysis_data = {
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generalization_gap": {
            "baseline": {
                "val_accuracy": base_val_m["accuracy"],
                "test_accuracy": base_test_m["accuracy"],
                "gap": round(base_gap, 4),
            },
            "improved": {
                "val_accuracy": imp_val_m["accuracy"],
                "test_accuracy": imp_test_m["accuracy"],
                "gap": round(imp_gap, 4),
                "test_accuracy_gain": round(imp_test_m["accuracy"] - base_test_m["accuracy"], 4),
            },
        },
        "baseline_test_metrics": base_test_m,
        "improved_test_metrics": imp_test_m,
        "baseline_val_metrics": base_val_m,
        "improved_val_metrics": imp_val_m,
        "signer_breakdown": signer_stats,
        "rejection_sweep": rejection_sweep,
    }

    analysis_json_path = ISOLATED_CHECKPOINT_DIR / "generalization_analysis.json"
    with open(analysis_json_path, mode="w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Saved generalization analysis to: {analysis_json_path}", flush=True)

    # 8. Generate Visual Verification Figures
    generate_figures(base_val_m, base_test_m, imp_val_m, imp_test_m, signer_stats, imp_test_m["confusion_matrix"])
    print("\n" + "=" * 80, flush=True)
    print("F-023 ANALYSIS & IMPROVEMENT COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 80, flush=True)


def generate_figures(b_val, b_test, i_val, i_test, signer_stats, cm_test):
    print("Generating visual comparison figures...", flush=True)

    # Figure 1: Generalization Gap Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Validation Accuracy", "Test Accuracy", "Test Macro F1", "Test Top-3 Acc"]
    base_scores = [b_val["accuracy"], b_test["accuracy"], b_test["macro_f1"], b_test["top3_accuracy"]]
    imp_scores = [i_val["accuracy"], i_test["accuracy"], i_test["macro_f1"], i_test["top3_accuracy"]]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, base_scores, width, label="Baseline (F-022)", color="#7f7f7f")
    ax.bar(x + width / 2, imp_scores, width, label="Improved (Bilateral F-023)", color="#2ca02c")

    ax.set_title("Generalization Gap Comparison: Baseline vs Improved Model")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim([0.0, 1.05])
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    for i in range(len(categories)):
        ax.text(i - width / 2, base_scores[i] + 0.02, f"{base_scores[i]:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, imp_scores[i] + 0.02, f"{imp_scores[i]:.2f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig1 = FIGURES_DIR / "generalization_gap_comparison.png"
    fig.savefig(fig1, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig1.name}", flush=True)

    # Figure 2: Signer Breakdown Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    signers = list(signer_stats.keys())
    b_accs = [signer_stats[s]["baseline_accuracy"] for s in signers]
    i_accs = [signer_stats[s]["improved_accuracy"] for s in signers]

    x = np.arange(len(signers))
    ax.bar(x - width / 2, b_accs, width, label="Baseline (F-022)", color="#d62728")
    ax.bar(x + width / 2, i_accs, width, label="Improved (Bilateral)", color="#1f77b4")

    ax.set_title("Test Accuracy by Signer: Identifying Left-Hand Bias Impact")
    ax.set_xticks(x)
    labels = [f"{s}\n({signer_stats[s]['dominance'][:4]})" for s in signers]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim([0.0, 1.15])
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig2 = FIGURES_DIR / "signer_breakdown_comparison.png"
    fig.savefig(fig2, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig2.name}", flush=True)

    # Figure 3: Improved Confusion Matrix (Test)
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_arr = np.array(cm_test)
    im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Improved Model Held-Out Test Confusion Matrix (Acc: {i_test['accuracy']:.2%})")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(ISOLATED_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(ISOLATED_CLASSES)
    ax.set_xlabel("Predicted Sign")
    ax.set_ylabel("True Sign")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_arr[i, j]
            color = "white" if val > cm_arr.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    plt.tight_layout()
    fig3 = FIGURES_DIR / "confusion_matrix_improved_test.png"
    fig.savefig(fig3, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig3.name}", flush=True)


if __name__ == "__main__":
    run_full_analysis()
