"""
F-023 Controlled Generalization Experiments.
Evaluates on Train and Validation splits ONLY.
Compares:
  - Exp 1: Baseline Bi-GRU (original 129 train samples)
  - Exp 2: Bi-GRU + Bilateral Mirror Augmentation (258 train samples)
  - Exp 3: Bi-GRU + Bilateral Mirror + Regularization (dropout=0.4, weight_decay=1e-3, label_smoothing=0.05)
  - Exp 4: Random Forest + Bilateral Mirror Augmentation (pooled 2400-d)
  - Exp 5: Lightweight 1-Layer Bi-GRU (hidden=96) + Bilateral Mirror
"""

import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.isolated.config.isolated_settings import (
    CLASS_TO_IDX,
    DEFAULT_SEQUENCE_LENGTH,
    FEATURES_DIR,
    ISOLATED_CLASSES,
    NUM_CLASSES,
    SPLITS_DIR,
)
from ai.isolated.evaluation.metrics import compute_isolated_metrics
from ai.isolated.features.feature_extractor import extract_isolated_features, pool_temporal_features
from ai.isolated.training.augmentation import IsolatedAugmenter
from ai.isolated.training.dataset_loader import load_split_dataset
from ai.isolated.training.neural_classifier import GRUClassifierWrapper, IsolatedBiGRU
from ai.isolated.training.tabular_pipeline import TabularSequenceClassifier


def build_bilateral_train_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs bilateral training dataset by pairing each training sample
    with its anatomically mirrored counterpart (258 samples total).
    """
    X_orig, y_orig, meta_orig = load_split_dataset("train", from_precomputed_features=True)

    augmenter = IsolatedAugmenter(seed=42)
    X_mirrored = []
    y_mirrored = []

    train_dir = FEATURES_DIR / "train"
    for i, meta in enumerate(meta_orig):
        vid = meta["video_id"]
        npz_file = train_dir / f"{vid}.npz"
        data = np.load(npz_file)
        normed_kps = data["normalized_landmarks"]  # (64, 75, 4)

        # Apply horizontal flip
        flipped_kps = augmenter.apply_horizontal_flip(normed_kps)
        # Extract features (spatial 300 + velocity 300 = 600)
        flipped_feat = extract_isolated_features(flipped_kps)
        X_mirrored.append(flipped_feat)
        y_mirrored.append(y_orig[i])

    X_mirrored_arr = np.asarray(X_mirrored, dtype=np.float32)
    y_mirrored_arr = np.asarray(y_mirrored, dtype=np.int64)

    X_combined = np.concatenate([X_orig, X_mirrored_arr], axis=0)
    y_combined = np.concatenate([y_orig, y_mirrored_arr], axis=0)

    return X_combined, y_combined


def train_custom_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 60,
    patience: int = 15,
    random_seed: int = 42,
) -> Tuple[GRUClassifierWrapper, Dict[str, Any], float]:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device = torch.device("cpu")
    model = IsolatedBiGRU(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=NUM_CLASSES,
        dropout=dropout,
    ).to(device)

    # Balanced class weights
    classes_arr = np.arange(NUM_CLASSES)
    cw = compute_class_weight("balanced", classes=classes_arr, y=y_train)
    weight_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Add small jitter during training
            jitter = torch.randn_like(bx) * 0.002
            optimizer.zero_grad()
            logits = model(bx + jitter)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation eval
        wrapper = GRUClassifierWrapper(model, device="cpu")
        val_probs = wrapper.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_isolated_metrics(y_val, val_preds, y_probs=val_probs)
        val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = round(time.perf_counter() - t_start, 2)
    if best_state is not None:
        model.load_state_dict(best_state)

    best_wrapper = GRUClassifierWrapper(model, device="cpu")
    val_probs = best_wrapper.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    final_metrics = compute_isolated_metrics(y_val, val_preds, y_probs=val_probs)
    final_metrics["train_time_s"] = train_time
    final_metrics["epochs_run"] = epoch

    return best_wrapper, final_metrics, train_time


def main():
    print("=" * 80)
    print("F-023: CONTROLLED GENERALIZATION EXPERIMENTS (TRAIN & VAL ONLY)")
    print("=" * 80)

    # 1. Load data
    X_train_orig, y_train_orig, _ = load_split_dataset("train", from_precomputed_features=True)
    X_train_bilateral, y_train_bilateral = build_bilateral_train_data()
    X_val, y_val, _ = load_split_dataset("validation", from_precomputed_features=True)

    print(f"Original Train:  {X_train_orig.shape}")
    print(f"Bilateral Train: {X_train_bilateral.shape} (Includes left-hand mirrored samples)")
    print(f"Validation:      {X_val.shape}")

    results: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Any] = {}

    # Experiment 1: Baseline Bi-GRU (Original Train)
    print("\n--- Running Exp 1: Baseline Bi-GRU (129 train samples) ---")
    m1, met1, t1 = train_custom_gru(
        X_train_orig, y_train_orig, X_val, y_val,
        hidden_dim=128, num_layers=2, dropout=0.3, weight_decay=1e-4, label_smoothing=0.0
    )
    results["Exp 1: Baseline Bi-GRU"] = met1
    models["Exp 1: Baseline Bi-GRU"] = m1
    print(f"  Val Acc: {met1['accuracy']:.4f} | Macro F1: {met1['macro_f1']:.4f} | Top-3: {met1['top3_accuracy']:.4f}")

    # Experiment 2: Bi-GRU + Bilateral Mirror Augmentation
    print("\n--- Running Exp 2: Bi-GRU + Bilateral Mirror Augmentation (258 train samples) ---")
    m2, met2, t2 = train_custom_gru(
        X_train_bilateral, y_train_bilateral, X_val, y_val,
        hidden_dim=128, num_layers=2, dropout=0.3, weight_decay=1e-4, label_smoothing=0.0
    )
    results["Exp 2: Bi-GRU + Bilateral Mirror"] = met2
    models["Exp 2: Bi-GRU + Bilateral Mirror"] = m2
    print(f"  Val Acc: {met2['accuracy']:.4f} | Macro F1: {met2['macro_f1']:.4f} | Top-3: {met2['top3_accuracy']:.4f}")

    # Experiment 3: Bi-GRU + Bilateral Mirror + Regularization
    print("\n--- Running Exp 3: Bi-GRU + Bilateral + Regularization (dropout=0.4, wd=1e-3, ls=0.05) ---")
    m3, met3, t3 = train_custom_gru(
        X_train_bilateral, y_train_bilateral, X_val, y_val,
        hidden_dim=128, num_layers=2, dropout=0.4, weight_decay=1e-3, label_smoothing=0.05
    )
    results["Exp 3: Bi-GRU + Bilateral + Reg"] = met3
    models["Exp 3: Bi-GRU + Bilateral + Reg"] = m3
    print(f"  Val Acc: {met3['accuracy']:.4f} | Macro F1: {met3['macro_f1']:.4f} | Top-3: {met3['top3_accuracy']:.4f}")

    # Experiment 4: Random Forest + Bilateral Mirror Augmentation
    print("\n--- Running Exp 4: Random Forest (Pooled 2400-d) + Bilateral Mirror ---")
    t0 = time.perf_counter()
    rf_base = RandomForestClassifier(n_estimators=250, max_depth=16, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_pipe = TabularSequenceClassifier(rf_base)
    rf_pipe.fit(X_train_bilateral, y_train_bilateral)
    t4 = round(time.perf_counter() - t0, 2)
    rf_probs = rf_pipe.predict_proba(X_val)
    rf_preds = np.argmax(rf_probs, axis=1)
    met4 = compute_isolated_metrics(y_val, rf_preds, y_probs=rf_probs)
    met4["train_time_s"] = t4
    results["Exp 4: Random Forest + Bilateral"] = met4
    models["Exp 4: Random Forest + Bilateral"] = rf_pipe
    print(f"  Val Acc: {met4['accuracy']:.4f} | Macro F1: {met4['macro_f1']:.4f} | Top-3: {met4['top3_accuracy']:.4f}")

    # Experiment 5: Lightweight 1-Layer Bi-GRU (hidden=96) + Bilateral Mirror
    print("\n--- Running Exp 5: Lightweight 1-Layer Bi-GRU (hidden=96) + Bilateral Mirror ---")
    m5, met5, t5 = train_custom_gru(
        X_train_bilateral, y_train_bilateral, X_val, y_val,
        hidden_dim=96, num_layers=1, dropout=0.2, weight_decay=5e-4, label_smoothing=0.02
    )
    results["Exp 5: 1-Layer Bi-GRU (h=96) + Bilateral"] = met5
    models["Exp 5: 1-Layer Bi-GRU (h=96) + Bilateral"] = m5
    print(f"  Val Acc: {met5['accuracy']:.4f} | Macro F1: {met5['macro_f1']:.4f} | Top-3: {met5['top3_accuracy']:.4f}")

    # Comparison Table
    print("\n" + "=" * 80)
    print("VALIDATION EXPERIMENT COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<38} {'Val Acc':<10} {'Macro P':<10} {'Macro R':<10} {'Macro F1':<10} {'Top-3':<8} {'Time (s)':<8}")
    print("-" * 94)
    for name, m in results.items():
        print(f"{name:<38} {m['accuracy']:<10.4f} {m['macro_precision']:<10.4f} {m['macro_recall']:<10.4f} {m['macro_f1']:<10.4f} {m['top3_accuracy']:<8.4f} {m['train_time_s']:<8.2f}")

    # Save experiment results JSON
    exp_json_path = PROJECT_ROOT / "docs" / "reports" / "f023_validation_experiments.json"
    clean_results = {k: {mk: mv for mk, mv in v.items() if mk != "confusion_matrix" and mk != "per_class"} for k, v in results.items()}
    with open(exp_json_path, mode="w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved experiment results to: {exp_json_path}")


if __name__ == "__main__":
    main()
