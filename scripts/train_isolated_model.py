"""
F-022: Model Training and Evaluation Pipeline for Isolated ISL Sign Recognition.
Trains, compares, and evaluates:
  1. Random Forest (with statistical temporal pooling)
  2. Support Vector Machine (RBF kernel, balanced weights, probability Platt scaling)
  3. PyTorch Bidirectional GRU (full (64, 600) sequence dynamics)
Outputs:
  - models/checkpoints/isolated/isolated_10class_best.joblib
  - models/checkpoints/isolated/model_metadata.json
  - docs/reports/figures/model_comparison.png
  - docs/reports/figures/confusion_matrix_val.png
  - docs/reports/figures/confusion_matrix_test.png
  - docs/reports/figures/confidence_rejection_curve.png
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

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
from ai.isolated.training.neural_classifier import train_gru
from ai.isolated.training.tabular_pipeline import TabularSequenceClassifier

FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"


def main():
    print("=" * 80)
    print("F-022: ISOLATED ISL RECOGNITION MODEL TRAINING & BENCHMARKING")
    print("=" * 80)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ISOLATED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Inspect Training Data
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Loading and inspecting dataset splits...")
    X_train, y_train, meta_train = load_split_dataset("train", from_precomputed_features=True)
    X_val, y_val, meta_val = load_split_dataset("validation", from_precomputed_features=True)
    X_test, y_test, meta_test = load_split_dataset("test", from_precomputed_features=True)

    print(f"  Train:      {X_train.shape} samples (Labels: {len(y_train)})")
    print(f"  Validation: {X_val.shape} samples (Labels: {len(y_val)})")
    print(f"  Test:       {X_test.shape} samples (Labels: {len(y_test)})")

    assert X_train.shape[1:] == (DEFAULT_SEQUENCE_LENGTH, 600)
    assert X_val.shape[1:] == (DEFAULT_SEQUENCE_LENGTH, 600)
    assert X_test.shape[1:] == (DEFAULT_SEQUENCE_LENGTH, 600)

    # Verify no video leakage across splits
    train_ids = {m["video_id"] for m in meta_train}
    val_ids = {m["video_id"] for m in meta_val}
    test_ids = {m["video_id"] for m in meta_test}
    assert len(train_ids.intersection(val_ids)) == 0, "Data leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Data leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Data leakage between val and test!"

    # Per-class counts
    train_counts = {cls: int(np.sum(y_train == idx)) for cls, idx in CLASS_TO_IDX.items()}
    val_counts = {cls: int(np.sum(y_val == idx)) for cls, idx in CLASS_TO_IDX.items()}
    test_counts = {cls: int(np.sum(y_test == idx)) for cls, idx in CLASS_TO_IDX.items()}

    print("\nPer-class sample distribution:")
    print(f"{'Class':<14} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 46)
    for cls in ISOLATED_CLASSES:
        idx = CLASS_TO_IDX[cls]
        tot = train_counts[cls] + val_counts[cls] + test_counts[cls]
        print(f"{cls:<14} {train_counts[cls]:<8} {val_counts[cls]:<8} {test_counts[cls]:<8} {tot:<8}")

    # Compute balanced class weights for training
    classes_arr = np.arange(NUM_CLASSES)
    class_weights_arr = compute_class_weight("balanced", classes=classes_arr, y=y_train)
    print(f"\nCalculated balanced class weights for training:\n  {dict(zip(ISOLATED_CLASSES, np.round(class_weights_arr, 2)))}")

    # -------------------------------------------------------------------------
    # STEP 2 & 3: Train Candidate Models
    # -------------------------------------------------------------------------
    candidates: Dict[str, Any] = {}
    candidate_timings: Dict[str, float] = {}

    # Candidate 1: Random Forest
    print("\n[STEP 2.1] Training Candidate 1: Random Forest (Temporal Pooled, 2400-dim)...")
    t0 = time.perf_counter()
    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model = TabularSequenceClassifier(rf_base)
    rf_model.fit(X_train, y_train)
    rf_time = round(time.perf_counter() - t0, 2)
    candidates["Random Forest"] = rf_model
    candidate_timings["Random Forest"] = rf_time
    print(f"  Random Forest trained in {rf_time}s")

    # Candidate 2: Support Vector Machine (RBF kernel)
    print("\n[STEP 2.2] Training Candidate 2: SVM (RBF Kernel, C=5.0, Balanced)...")
    t0 = time.perf_counter()
    svm_base = SVC(
        C=5.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    svm_model = TabularSequenceClassifier(svm_base)
    svm_model.fit(X_train, y_train)
    svm_time = round(time.perf_counter() - t0, 2)
    candidates["SVM (RBF)"] = svm_model
    candidate_timings["SVM (RBF)"] = svm_time
    print(f"  SVM trained in {svm_time}s")

    # Candidate 3: PyTorch Bidirectional GRU
    print("\n[STEP 3] Training Candidate 3: PyTorch 2-Layer Bi-GRU (64 frames x 600 dims)...")
    gru_model, gru_history, gru_time = train_gru(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights_arr,
        epochs=60,
        batch_size=16,
        lr=1e-3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        patience=15,
        random_seed=42,
    )
    candidates["Bi-GRU"] = gru_model
    candidate_timings["Bi-GRU"] = gru_time
    print(f"  Bi-GRU trained in {gru_time}s (stopped at epoch {len(gru_history['train_loss'])})")

    # -------------------------------------------------------------------------
    # STEP 4: Validation Evaluation & Model Selection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: VALIDATION BENCHMARK EVALUATION (28 CLIPS)")
    print("=" * 80)

    val_results: Dict[str, Dict[str, Any]] = {}

    print(f"{'Model':<16} {'Val Acc':<10} {'Macro P':<10} {'Macro R':<10} {'Macro F1':<10} {'Top-3 Acc':<10} {'Train (s)':<10}")
    print("-" * 76)

    for name, model in candidates.items():
        probs = model.predict_proba(X_val)
        preds = np.argmax(probs, axis=1)
        metrics = compute_isolated_metrics(y_val, preds, y_probs=probs)
        metrics["training_time_s"] = candidate_timings[name]
        val_results[name] = metrics

        print(
            f"{name:<16} "
            f"{metrics['accuracy']:<10.4f} "
            f"{metrics['macro_precision']:<10.4f} "
            f"{metrics['macro_recall']:<10.4f} "
            f"{metrics['macro_f1']:<10.4f} "
            f"{metrics['top3_accuracy']:<10.4f} "
            f"{candidate_timings[name]:<10.2f}"
        )

    # Select champion based on Validation Macro F1
    champion_name = max(val_results.keys(), key=lambda k: (val_results[k]["macro_f1"], val_results[k]["accuracy"]))
    champion_model = candidates[champion_name]
    print(f"\n>>> CHAMPION MODEL SELECTED: {champion_name} (Val Macro F1 = {val_results[champion_name]['macro_f1']:.4f})")

    # -------------------------------------------------------------------------
    # STEP 5: Confidence Rejection & Calibration Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STEP 5: CONFIDENCE REJECTION & CALIBRATION ANALYSIS ({champion_name})")
    print("=" * 80)

    champ_val_probs = champion_model.predict_proba(X_val)
    champ_val_preds = np.argmax(champ_val_probs, axis=1)
    champ_val_max_conf = np.max(champ_val_probs, axis=1)

    # Compute multi-class Brier score
    y_val_one_hot = np.zeros_like(champ_val_probs)
    for i, label in enumerate(y_val):
        y_val_one_hot[i, label] = 1.0
    brier_score = float(np.mean(np.sum((champ_val_probs - y_val_one_hot) ** 2, axis=1)))
    print(f"Multi-class Brier Score on Validation: {brier_score:.4f} (lower is better, 0.0 is perfect)")

    # Sweep rejection threshold
    threshold_sweep = []
    print(f"\n{'Threshold (tau)':<16} {'Coverage %':<12} {'Accepted':<10} {'Rejected':<10} {'Accepted Acc':<14}")
    print("-" * 62)

    thresholds = np.arange(0.20, 0.85, 0.05)
    for tau in thresholds:
        accepted = champ_val_max_conf >= tau
        n_accepted = int(np.sum(accepted))
        n_rejected = int(np.sum(~accepted))
        coverage = round(float(n_accepted) / len(y_val) * 100.0, 1)

        if n_accepted > 0:
            acc_accepted = round(float(np.mean(champ_val_preds[accepted] == y_val[accepted])), 4)
        else:
            acc_accepted = 1.0

        threshold_sweep.append({
            "threshold": round(float(tau), 2),
            "coverage_pct": coverage,
            "accepted_count": n_accepted,
            "rejected_count": n_rejected,
            "accuracy_on_accepted": acc_accepted,
        })
        print(f"{tau:<16.2f} {coverage:<12.1f}% {n_accepted:<10} {n_rejected:<10} {acc_accepted:<14.4f}")

    # Optimal threshold recommendation: balanced point where accuracy on accepted >= 0.90 while coverage >= 80%
    optimal_tau = 0.50
    for row in threshold_sweep:
        if row["coverage_pct"] >= 75.0 and row["accuracy_on_accepted"] >= val_results[champion_name]["accuracy"]:
            optimal_tau = row["threshold"]

    print(f"\nSelected Optimal Rejection Threshold: tau = {optimal_tau:.2f}")

    # -------------------------------------------------------------------------
    # STEP 6: Low-Data Class Breakdown (Validation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: LOW-DATA CLASS ANALYSIS (STOP & SORRY vs MAJORITY)")
    print("=" * 80)

    val_per_class = val_results[champion_name]["per_class"]
    print(f"{'Class Group':<24} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)

    # Minority classes
    for c in ["stop", "sorry"]:
        m = val_per_class.get(c, {})
        print(f"Minority: {c:<14} {int(m.get('support', 0)):<10} {m.get('precision', 0.0):<12.4f} {m.get('recall', 0.0):<12.4f} {m.get('f1-score', 0.0):<12.4f}")

    # 8 Majority classes average
    maj_classes = [c for c in ISOLATED_CLASSES if c not in ("stop", "sorry")]
    maj_prec = np.mean([val_per_class[c]["precision"] for c in maj_classes])
    maj_rec = np.mean([val_per_class[c]["recall"] for c in maj_classes])
    maj_f1 = np.mean([val_per_class[c]["f1-score"] for c in maj_classes])
    maj_sup = sum(val_per_class[c]["support"] for c in maj_classes)
    print(f"Majority: 8 Classes Avg  {int(maj_sup):<10} {maj_prec:<12.4f} {maj_rec:<12.4f} {maj_f1:<12.4f}")

    # -------------------------------------------------------------------------
    # STEP 7: Final Held-Out Evaluation on Test Split (30 CLIPS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STEP 7: FINAL SINGLE EVALUATION ON HELD-OUT TEST SPLIT (30 CLIPS) - {champion_name}")
    print("=" * 80)

    test_probs = champion_model.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = compute_isolated_metrics(y_test, test_preds, y_probs=test_probs)

    print(f"  Test Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro Prec:   {test_metrics['macro_precision']:.4f}")
    print(f"  Test Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"  Test Macro F1:     {test_metrics['macro_f1']:.4f}")
    print(f"  Test Weighted F1:  {test_metrics['weighted_f1']:.4f}")
    print(f"  Test Top-3 Acc:    {test_metrics['top3_accuracy']:.4f}")

    print("\nPer-class breakdown on Test Split:")
    print(f"{'Class':<14} {'Support':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 52)
    test_per_class = test_metrics["per_class"]
    for cls in ISOLATED_CLASSES:
        m = test_per_class[cls]
        print(f"{cls:<14} {int(m['support']):<8} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f}")

    # -------------------------------------------------------------------------
    # STEP 8: Save Model Artifacts
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 8: SAVING MODEL ARTIFACTS")
    print("=" * 80)

    # Save champion model checkpoint
    joblib.dump(champion_model, ISOLATED_CHECKPOINT_PATH)
    checkpoint_size_mb = round(ISOLATED_CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 2)
    print(f"  Saved model checkpoint to: {ISOLATED_CHECKPOINT_PATH} ({checkpoint_size_mb} MB)")

    # Save metadata JSON
    metadata = {
        "status": "trained",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_model": champion_name,
        "champion_checkpoint_file": ISOLATED_CHECKPOINT_PATH.name,
        "checkpoint_size_mb": checkpoint_size_mb,
        "target_classes": ISOLATED_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "feature_dim": 600,
        "input_representation": "temporal_summary_pooled_2400" if "Tabular" in str(type(champion_model)) else "sequence_64x600",
        "optimal_rejection_threshold": optimal_tau,
        "brier_score_validation": round(brier_score, 4),
        "validation_benchmark": val_results,
        "test_results": test_metrics,
        "rejection_threshold_sweep": threshold_sweep,
    }

    metadata_path = ISOLATED_CHECKPOINT_DIR / "model_metadata.json"
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved training metadata to: {metadata_path}")

    # -------------------------------------------------------------------------
    # STEP 9: Generate Report Figures
    # -------------------------------------------------------------------------
    print("\nGenerating report figures...")

    # Figure 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(candidates))
    width = 0.22

    accs = [val_results[m]["accuracy"] for m in candidates]
    f1s = [val_results[m]["macro_f1"] for m in candidates]
    top3s = [val_results[m]["top3_accuracy"] for m in candidates]

    ax.bar(x - width, accs, width, label="Accuracy", color="#2ca02c")
    ax.bar(x, f1s, width, label="Macro F1", color="#1f77b4")
    ax.bar(x + width, top3s, width, label="Top-3 Accuracy", color="#ff7f0e")

    ax.set_title("Model Validation Performance Comparison (28 Clips)")
    ax.set_xticks(x)
    ax.set_xticklabels(candidates.keys())
    ax.set_ylim([0.0, 1.05])
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    for i in range(len(candidates)):
        ax.text(i - width, accs[i] + 0.02, f"{accs[i]:.2f}", ha="center", fontsize=9)
        ax.text(i, f1s[i] + 0.02, f"{f1s[i]:.2f}", ha="center", fontsize=9)
        ax.text(i + width, top3s[i] + 0.02, f"{top3s[i]:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig1_path.name}")

    # Figure 2: Confusion Matrix (Validation)
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_val = np.array(val_results[champion_name]["confusion_matrix"])
    im = ax.imshow(cm_val, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Validation Confusion Matrix - {champion_name} (Acc: {val_results[champion_name]['accuracy']:.2f})")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(ISOLATED_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(ISOLATED_CLASSES)
    ax.set_xlabel("Predicted Sign")
    ax.set_ylabel("True Sign")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_val[i, j]
            color = "white" if val > cm_val.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "confusion_matrix_val.png"
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig2_path.name}")

    # Figure 3: Confusion Matrix (Test)
    fig, ax = plt.subplots(figsize=(8, 7))
    cm_test = np.array(test_metrics["confusion_matrix"])
    im = ax.imshow(cm_test, cmap="Greens", interpolation="nearest")
    ax.set_title(f"Held-Out Test Confusion Matrix - {champion_name} (Acc: {test_metrics['accuracy']:.2f})")
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(ISOLATED_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(ISOLATED_CLASSES)
    ax.set_xlabel("Predicted Sign")
    ax.set_ylabel("True Sign")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_test[i, j]
            color = "white" if val > cm_test.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color)

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "confusion_matrix_test.png"
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig3_path.name}")

    # Figure 4: Confidence Rejection Tradeoff Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    taus = [row["threshold"] for row in threshold_sweep]
    covs = [row["coverage_pct"] for row in threshold_sweep]
    accs = [row["accuracy_on_accepted"] * 100.0 for row in threshold_sweep]

    ax.plot(taus, covs, marker="o", label="Coverage (%)", color="#1f77b4", lw=2)
    ax.plot(taus, accs, marker="s", label="Accuracy on Accepted (%)", color="#2ca02c", lw=2)
    ax.axvline(optimal_tau, color="red", linestyle="--", label=f"Selected tau = {optimal_tau:.2f}")

    ax.set_title(f"Confidence Threshold vs Coverage & Accuracy - {champion_name}")
    ax.set_xlabel("Confidence Rejection Threshold (tau)")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig4_path = FIGURES_DIR / "confidence_rejection_curve.png"
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure: {fig4_path.name}")

    print("\n" + "=" * 80)
    print("F-022 MODEL TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
