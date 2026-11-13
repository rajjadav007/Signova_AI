"""
Train and benchmark 20-class Isolated ISL Recognition Models.
Trains triad:
  1. Random Forest (with statistical temporal pooling 2400-d)
  2. Support Vector Machine (RBF kernel + Platt scaling)
  3. PyTorch Bidirectional GRU (full (64, 600) dynamics + bilateral mirror augmentation)

Outputs:
  - models/checkpoints/isolated/isolated_20class_best.joblib
  - models/labels/isolated_20class_labels.json
  - models/checkpoints/isolated/model_metadata_20class.json
  - docs/reports/figures/confusion_matrix_test_20class.png
  - docs/reports/figures/confusion_matrix_val_20class.png
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Headless matplotlib
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TEMP", "C:/tmp")) / "matplotlib_signova"))
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.evaluation.metrics import compute_isolated_metrics
from ai.isolated.features.feature_extractor import extract_isolated_features, pool_temporal_features
from ai.isolated.training.augmentation import IsolatedAugmenter
from ai.isolated.training.neural_classifier import GRUClassifierWrapper, IsolatedBiGRU
from ai.isolated.training.tabular_pipeline import TabularSequenceClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "isolated"
LABELS_DIR = PROJECT_ROOT / "models" / "labels"
FEATURES_DIR = PROJECT_ROOT / "datasets" / "features"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"

CLASSES_20 = [
    "food", "hello", "help", "no", "please", "sorry", "stop", "thank you", "water", "yes",
    "father", "mother", "sister", "teacher", "you", "friend", "student", "school", "market", "hospital"
]
CLASS_TO_IDX_20 = {cls: idx for idx, cls in enumerate(CLASSES_20)}
IDX_TO_CLASS_20 = {idx: cls for idx, cls in enumerate(CLASSES_20)}





def load_dataset_split(split_name: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    csv_file = SPLITS_DIR / f"{split_name}_20class.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Missing {csv_file}")

    df = pd.read_csv(csv_file)
    X_list = []
    y_list = []
    meta_list = []

    feat_dir = FEATURES_DIR / split_name

    for _, row in df.iterrows():
        vid = row["video_id"]
        cls = row["class_label"]
        feat_path = feat_dir / f"{vid}.npz"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature npz: {feat_path}")

        data = np.load(feat_path)
        feat_600 = data["features"]  # (64, 600)
        X_list.append(feat_600)
        y_list.append(CLASS_TO_IDX_20[cls])
        meta_list.append(dict(row))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, meta_list


def build_bilateral_train_data(X_train: np.ndarray, y_train: np.ndarray, meta_train: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    augmenter = IsolatedAugmenter(seed=42)
    X_mirrored = []
    y_mirrored = []

    train_dir = FEATURES_DIR / "train"
    for i, meta in enumerate(meta_train):
        vid = meta["video_id"]
        npz_file = train_dir / f"{vid}.npz"
        data = np.load(npz_file)
        normed_kps = data["normalized_landmarks"]  # (64, 75, 4)

        flipped_kps = augmenter.apply_horizontal_flip(normed_kps)
        flipped_feat = extract_isolated_features(flipped_kps)
        X_mirrored.append(flipped_feat)
        y_mirrored.append(y_train[i])

    X_mirrored_arr = np.asarray(X_mirrored, dtype=np.float32)
    y_mirrored_arr = np.asarray(y_mirrored, dtype=np.int64)

    X_combined = np.concatenate([X_train, X_mirrored_arr], axis=0)
    y_combined = np.concatenate([y_train, y_mirrored_arr], axis=0)
    return X_combined, y_combined


def train_gru_20class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 15,
    random_seed: int = 42,
) -> Tuple[GRUClassifierWrapper, Dict[str, Any], float]:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cpu")
    model = IsolatedBiGRU(
        input_dim=600,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=20,
        dropout=dropout,
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_macro_f1": [], "val_accuracy": []}
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            jitter = torch.randn_like(batch_x) * 0.002
            batch_x_aug = batch_x + jitter

            optimizer.zero_grad()
            logits = model(batch_x_aug)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))

        # Eval on val
        wrapper = GRUClassifierWrapper(model, device="cpu")
        val_probs = wrapper.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_isolated_metrics(y_val, val_preds, y_probs=val_probs, target_names=CLASSES_20)

        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(round(avg_loss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs} - Loss: {avg_loss:.4f} - Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f})", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping triggered at epoch {epoch} (Patience={patience})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = round(time.perf_counter() - t_start, 2)
    champion_wrapper = GRUClassifierWrapper(model, device="cpu")
    return champion_wrapper, history, train_time


def run_training_pipeline():
    print("=" * 80)
    print("SIGNVOVA AI: 20-CLASS ISOLATED ISL MODEL TRAINING & EVALUATION")
    print("=" * 80)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    print("\n[1/6] Loading 20-class splits...")
    X_train, y_train, meta_train = load_dataset_split("train")
    X_val, y_val, meta_val = load_dataset_split("validation")
    X_test, y_test, meta_test = load_dataset_split("test")

    print(f"  Train:      {X_train.shape} samples")
    print(f"  Validation: {X_val.shape} samples")
    print(f"  Test:       {X_test.shape} samples")

    # 2. Bilateral Mirror Augmentation
    print("\n[2/6] Applying Bilateral Mirror Augmentation to training set...")
    X_train_aug, y_train_aug = build_bilateral_train_data(X_train, y_train, meta_train)
    print(f"  Augmented Train: {X_train_aug.shape} samples (2x size)")

    # Compute class weights
    classes_present = np.unique(y_train_aug)
    weights = compute_class_weight("balanced", classes=classes_present, y=y_train_aug)
    full_weights = np.ones(20, dtype=np.float32)
    for c, w in zip(classes_present, weights):
        full_weights[c] = float(w)

    # Pooled features for tabular baselines
    print("\n[3/6] Generating temporal pooled features (2400-d) for baselines...")
    X_train_pool = pool_temporal_features(X_train_aug)
    X_val_pool = pool_temporal_features(X_val)
    X_test_pool = pool_temporal_features(X_test)

    # 3. Model Benchmark Triad
    print("\n[4/6] Training benchmark triad on validation set...")
    results = {}

    # A. Random Forest
    print("  Training Random Forest (300 trees)...")
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, max_depth=16, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train_pool, y_train_aug)
    rf_time = round(time.perf_counter() - t0, 2)
    rf_wrapper = TabularSequenceClassifier(rf)
    rf_val_probs = rf_wrapper.predict_proba(X_val)
    rf_val_preds = np.argmax(rf_val_probs, axis=1)
    rf_val_metrics = compute_isolated_metrics(y_val, rf_val_preds, y_probs=rf_val_probs, target_names=CLASSES_20)
    results["Random Forest"] = {"model": rf_wrapper, "val_metrics": rf_val_metrics, "time_sec": rf_time}
    print(f"    RF Val Accuracy: {rf_val_metrics['accuracy']:.4f} | Macro F1: {rf_val_metrics['macro_f1']:.4f} ({rf_time}s)")

    # B. Linear Classifier (StandardScaler + SGDClassifier with log loss)
    print("  Training Linear Baseline (StandardScaler + SGDClassifier)...")
    t0 = time.perf_counter()
    linear_clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, random_state=42),
    )
    linear_clf.fit(X_train_pool, y_train_aug)
    linear_time = round(time.perf_counter() - t0, 2)
    linear_wrapper = TabularSequenceClassifier(linear_clf)
    linear_val_probs = linear_wrapper.predict_proba(X_val)
    linear_val_preds = np.argmax(linear_val_probs, axis=1)
    linear_val_metrics = compute_isolated_metrics(y_val, linear_val_preds, y_probs=linear_val_probs, target_names=CLASSES_20)
    results["Linear (SGD)"] = {"model": linear_wrapper, "val_metrics": linear_val_metrics, "time_sec": linear_time}
    print(f"    Linear Val Accuracy: {linear_val_metrics['accuracy']:.4f} | Macro F1: {linear_val_metrics['macro_f1']:.4f} ({linear_time}s)")

    # C. Bi-GRU
    print("  Training PyTorch Bi-GRU (hidden=128, 2-layer, AdamW)...")
    gru_wrapper, gru_hist, gru_time = train_gru_20class(
        X_train=X_train_aug,
        y_train=y_train_aug,
        X_val=X_val,
        y_val=y_val,
        class_weights=full_weights,
        epochs=60,
        patience=15,
    )
    gru_val_probs = gru_wrapper.predict_proba(X_val)
    gru_val_preds = np.argmax(gru_val_probs, axis=1)
    gru_val_metrics = compute_isolated_metrics(y_val, gru_val_preds, y_probs=gru_val_probs, target_names=CLASSES_20)
    results["Bi-GRU"] = {"model": gru_wrapper, "val_metrics": gru_val_metrics, "time_sec": gru_time}
    print(f"    Bi-GRU Val Accuracy: {gru_val_metrics['accuracy']:.4f} | Macro F1: {gru_val_metrics['macro_f1']:.4f} ({gru_time}s)")

    # 4. Champion Selection
    print("\n[5/6] Model comparison and champion selection:")
    for name, res in results.items():
        m = res["val_metrics"]
        print(f"  {name:<16} Acc={m['accuracy']:.4f} | F1={m['macro_f1']:.4f} | Prec={m['macro_precision']:.4f} | Rec={m['macro_recall']:.4f}")

    # Best model by Macro F1
    champion_name = max(results.keys(), key=lambda k: results[k]["val_metrics"]["macro_f1"])
    champion = results[champion_name]["model"]
    champion.classes_ = np.arange(20)
    print(f"\nCHAMPION MODEL SELECTED: {champion_name} (Val Macro F1: {results[champion_name]['val_metrics']['macro_f1']:.4f})")

    # 5. Final Independent Test Evaluation
    print("\n[6/6] Final independent evaluation on completely unseen TEST split (66 samples)...")
    test_probs = champion.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = compute_isolated_metrics(y_test, test_preds, y_probs=test_probs, target_names=CLASSES_20)

    print(f"\n=== TEST EVALUATION METRICS (20 CLASSES) ===")
    print(f"  Overall Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Macro Precision:  {test_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {test_metrics['macro_recall']:.4f}")
    print(f"  Macro F1:         {test_metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    print(f"  Top-3 Accuracy:   {test_metrics.get('top3_accuracy', 0.0):.4f}")

    print("\nPer-class metrics (Test):")
    print(f"{'Class':<14} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    print("-" * 52)
    per_cls = test_metrics["per_class"]
    for idx, cls in enumerate(CLASSES_20):
        cls_rep = per_cls.get(cls, per_cls.get(str(idx), {}))
        p = cls_rep.get("precision", 0.0)
        r = cls_rep.get("recall", 0.0)
        f1 = cls_rep.get("f1-score", 0.0)
        sup = cls_rep.get("support", 0)
        print(f"{cls:<14} {p:<10.4f} {r:<10.4f} {f1:<10.4f} {sup:<8}")

    # Specific FOOD vs WATER evaluation
    food_idx = CLASS_TO_IDX_20["food"]
    water_idx = CLASS_TO_IDX_20["water"]
    cm = np.array(test_metrics["confusion_matrix"])
    food_as_water = int(cm[food_idx, water_idx])
    water_as_food = int(cm[water_idx, food_idx])
    print(f"\n=== SPECIFIC FOOD VS WATER CONFUSION ===")
    print(f"  Food samples predicted as Water: {food_as_water} / {int(np.sum(cm[food_idx, :]))}")
    print(f"  Water samples predicted as Food: {water_as_food} / {int(np.sum(cm[water_idx, :]))}")
    print(f"  Confusion status: {'ZERO CONFUSION' if food_as_water == 0 and water_as_food == 0 else 'CROSS-TALK DETECTED'}")

    # 6. Save Checkpoints and Labels
    champion_path = CHECKPOINT_DIR / "isolated_20class_best.joblib"
    joblib.dump(champion, champion_path)
    print(f"\nSaved champion checkpoint: {champion_path} ({champion_path.stat().st_size / (1024*1024):.2f} MB)")

    # Save 20-class labels
    labels_path = LABELS_DIR / "isolated_20class_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(CLASSES_20, f, indent=2)
    print(f"Saved 20-class labels: {labels_path}")

    # Save metadata
    meta_path = CHECKPOINT_DIR / "model_metadata_20class.json"
    metadata = {
        "status": "improved_20class",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_model": f"{champion_name} (Bilateral Mirror Augmented)",
        "champion_checkpoint_file": "isolated_20class_best.joblib",
        "baseline_backup_checkpoint": "isolated_10class_best.joblib",
        "labels_file": "isolated_20class_labels.json",
        "num_classes": 20,
        "classes": CLASSES_20,
        "optimal_rejection_threshold": 0.50,
        "food_water_ambiguity_margin": 0.20,
        "checkpoint_size_mb": round(champion_path.stat().st_size / (1024*1024), 2),
        "validation_benchmark": {k: v["val_metrics"] for k, v in results.items()},
        "test_benchmark": test_metrics,
        "food_vs_water_confusion": {
            "food_as_water": food_as_water,
            "water_as_food": water_as_food,
            "status": "ZERO CONFUSION" if food_as_water == 0 and water_as_food == 0 else "CROSS-TALK",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # 7. Plot confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(CLASSES_20)))
    ax.set_yticks(np.arange(len(CLASSES_20)))
    ax.set_xticklabels(CLASSES_20, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES_20, fontsize=9)
    ax.set_title(f"Test Confusion Matrix - 20 Classes ({champion_name})", fontsize=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    for i in range(len(CLASSES_20)):
        for j in range(len(CLASSES_20)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix_test_20class.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")

    print("\nTraining and evaluation successfully completed!")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    run_training_pipeline()
