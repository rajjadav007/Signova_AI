"""
Signova AI - Five-Class ISL Sentence Sequence Classifier Training and Evaluation.
Phases 6, 7, 8, and 10:
- BiGRU with Bahdanau Self-Attention Pooling.
- Signer-independent data splitting:
    Train: Signers 1, 3, 4, 5, 6 (25 sequences)
    Validation: Signer 7 (5 sequences)
    Test: Signer 2 (5 sequences) - strictly quarantined until final evaluation.
- Evaluates Top-1, Top-3, Macro F1, Per-class metrics, confusion matrices, and latency.
- Saves all artifacts to models/isl_csltr_continuous/five_class_sequence/.
"""

import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.continuous.five_class_model import FiveClassBiGRUAttentionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_five_class")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "five_class_sequence" / "five_class_dataset.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "five_class_sequence"
ROOT_ARTIFACTS_DIR = Path(r"D:\Projects\models\isl_csltr_continuous\five_class_sequence")

CLASS_NAMES = [
    "comb your hair",
    "help me",
    "i am fine. thank you sir.",
    "i need water",
    "who are you"
]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    """
    Dataset for variable-length landmark sequences with optional on-the-fly augmentation.
    """

    def __init__(
        self,
        features_list: List[np.ndarray],
        labels: np.ndarray,
        signers: np.ndarray,
        seq_ids: np.ndarray,
        augment: bool = False
    ):
        self.features = features_list
        self.labels = labels
        self.signers = signers
        self.seq_ids = seq_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.features)

    def _augment_sequence(self, x: np.ndarray) -> np.ndarray:
        """
        On-the-fly augmentation:
        - Temporal speed jitter: resample time length by 0.85x - 1.15x
        - Spatial Gaussian noise: sigma=0.01 on present coordinates
        - Frame dropout: randomly drop 5% of frames
        """
        t, d = x.shape
        # 1. Temporal speed jitter
        if random.random() < 0.7 and t > 10:
            scale = random.uniform(0.85, 1.15)
            new_t = max(8, int(round(t * scale)))
            indices = np.linspace(0, t - 1, new_t)
            # Linear interpolation
            x_resampled = np.zeros((new_t, d), dtype=np.float32)
            orig_indices = np.arange(t)
            for j in range(d):
                x_resampled[:, j] = np.interp(indices, orig_indices, x[:, j])
            x = x_resampled
            t = new_t

        # 2. Gaussian coordinate noise on present landmarks
        if random.random() < 0.7:
            x_reshaped = x.reshape(t, 75, 4).copy()
            # Only add noise to present keypoints (channel 3 > 0.0)
            vis = x_reshaped[:, :, 3]
            noise = np.random.normal(0.0, 0.01, size=(t, 75, 3)).astype(np.float32)
            mask_present = (vis > 0.0)[:, :, None]
            x_reshaped[:, :, :3] += noise * mask_present
            x = x_reshaped.reshape(t, d)

        # 3. Random frame dropout
        if random.random() < 0.5 and t > 12:
            keep_mask = np.random.rand(t) > 0.05
            if np.sum(keep_mask) >= 8:
                x = x[keep_mask]

        return x.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        feat = self.features[idx]
        if self.augment:
            feat = self._augment_sequence(feat)
        return (
            torch.from_numpy(feat).float(),
            int(self.labels[idx]),
            int(self.signers[idx]),
            str(self.seq_ids[idx])
        )


def collate_variable_length(batch):
    """
    Pad variable-length sequences to max length in the batch and produce boolean mask.
    """
    feats, labels, signers, seq_ids = zip(*batch)
    lengths = [len(f) for f in feats]
    max_len = max(lengths)
    dim = feats[0].shape[1]

    padded_feats = torch.zeros(len(feats), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(feats), max_len, dtype=torch.bool)

    for i, (f, l) in enumerate(zip(feats, lengths)):
        padded_feats[i, :l] = f
        mask[i, :l] = True

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_feats, mask, labels_tensor, signers, seq_ids


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int], List[List[float]], List[float]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []
    latencies = []

    with torch.no_grad():
        for x, mask, y, _, _ in dataloader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            t0 = time.perf_counter()
            logits, _ = model(x, mask=mask)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(y)
            latencies.append(elapsed_ms)

            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(all_targets) if all_targets else 0.0
    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets) if all_targets else 0.0
    return avg_loss, acc, all_preds, all_targets, all_probs, latencies


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    data = np.load(DATA_PATH, allow_pickle=True)
    train_feats = list(data["train_features"])
    train_labels = data["train_labels"]
    train_signers = data["train_signers"]
    train_seq_ids = data["train_seq_ids"]

    val_feats = list(data["val_features"])
    val_labels = data["val_labels"]
    val_signers = data["val_signers"]
    val_seq_ids = data["val_seq_ids"]

    test_feats = list(data["test_features"])
    test_labels = data["test_labels"]
    test_signers = data["test_signers"]
    test_seq_ids = data["test_seq_ids"]

    logger.info(f"Loaded: Train={len(train_feats)} sequences (Signers {sorted(list(set(train_signers)))}), "
                f"Val={len(val_feats)} sequences (Signers {sorted(list(set(val_signers)))}), "
                f"Test={len(test_feats)} sequences (Signers {sorted(list(set(test_signers)))})")

    # 2. Datasets & Loaders
    train_dataset = SequenceDataset(train_feats, train_labels, train_signers, train_seq_ids, augment=True)
    val_dataset = SequenceDataset(val_feats, val_labels, val_signers, val_seq_ids, augment=False)
    test_dataset = SequenceDataset(test_feats, test_labels, test_signers, test_seq_ids, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_variable_length)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_variable_length)

    # 3. Model setup
    model = FiveClassBiGRUAttentionModel(
        input_dim=300,
        proj_dim=128,
        hidden_dim=64,
        num_layers=1,
        num_classes=5,
        dropout=0.3
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized: {param_count:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 80
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    patience = 25
    no_improve_count = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_checkpoint_path = ARTIFACTS_DIR / "best_model.pt"

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, mask, y, _, _ in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

        scheduler.step()

        train_loss = epoch_loss / total
        train_acc = correct / total

        val_loss, val_acc, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(float(scheduler.get_last_lr()[0]))

        if val_loss < best_val_loss or (val_loss == best_val_loss and val_acc > best_val_acc):
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_count = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            # Also save to root artifacts mirror
            torch.save(model.state_dict(), ROOT_ARTIFACTS_DIR / "best_model.pt")
        else:
            no_improve_count += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == best_epoch:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} Acc: {train_acc:.2f} | "
                        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f} (Best Val Loss: {best_val_loss:.4f} @ Ep {best_epoch})")

        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs).")
            break

    training_time = time.time() - t_start
    logger.info(f"Training completed in {training_time:.2f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}")

    # 4. Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(best_checkpoint_path))
    model.eval()

    # 5. Evaluate on Train, Validation, and Test
    # Evaluation loader for train without augmentation
    train_eval_dataset = SequenceDataset(train_feats, train_labels, train_signers, train_seq_ids, augment=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=len(train_eval_dataset), shuffle=False, collate_fn=collate_variable_length)

    tr_loss, tr_acc, tr_preds, tr_targets, _, _ = evaluate(model, train_eval_loader, criterion, device)
    val_loss, val_acc, val_preds, val_targets, val_probs, _ = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, test_preds, test_targets, test_probs, test_latencies = evaluate(model, test_loader, criterion, device)

    # Compute Top-3 accuracy on Test
    test_probs_np = np.array(test_probs)
    top3_correct = 0
    for i, target in enumerate(test_targets):
        top3_classes = np.argsort(test_probs_np[i])[-3:]
        if target in top3_classes:
            top3_correct += 1
    top3_acc = top3_correct / len(test_targets)

    # Classification reports and per-class metrics on test
    precision, recall, f1, support = precision_recall_fscore_support(
        test_targets, test_preds, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    macro_p = float(np.mean(precision))
    macro_r = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    conf_matrix = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2, 3, 4]).tolist()

    per_class_metrics = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class_metrics[cname] = {
            "class_id": i,
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    # Prediction detail table for test sequences
    test_detail = []
    for i in range(len(test_targets)):
        true_id = test_targets[i]
        pred_id = test_preds[i]
        probs = [round(float(p), 4) for p in test_probs[i]]
        test_detail.append({
            "sequence_id": str(test_seq_ids[i]),
            "signer_id": int(test_signers[i]),
            "ground_truth_id": true_id,
            "ground_truth_sentence": CLASS_NAMES[true_id],
            "predicted_id": pred_id,
            "predicted_sentence": CLASS_NAMES[pred_id],
            "is_correct": bool(true_id == pred_id),
            "confidence": probs[pred_id],
            "probabilities": probs
        })

    avg_test_latency_ms = float(np.mean(test_latencies)) if test_latencies else 0.0

    eval_results = {
        "dataset_split": {
            "train_signers": [1, 3, 4, 5, 6],
            "validation_signers": [7],
            "test_signers": [2],
            "train_sequences": len(train_feats),
            "validation_sequences": len(val_feats),
            "test_sequences": len(test_feats)
        },
        "model_architecture": {
            "type": "FiveClassBiGRUAttentionModel",
            "parameter_count": param_count,
            "input_dim": 300,
            "hidden_dim": 64,
            "num_layers": 1,
            "num_classes": 5
        },
        "training": {
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "training_time_seconds": round(training_time, 2)
        },
        "metrics": {
            "train_accuracy": round(tr_acc, 4),
            "train_loss": round(tr_loss, 4),
            "validation_accuracy": round(val_acc, 4),
            "validation_loss": round(val_loss, 4),
            "test_top1_accuracy": round(test_acc, 4),
            "test_top3_accuracy": round(top3_acc, 4),
            "test_macro_precision": round(macro_p, 4),
            "test_macro_recall": round(macro_r, 4),
            "test_macro_f1": round(macro_f1, 4),
            "per_class": per_class_metrics,
            "test_confusion_matrix": conf_matrix,
            "test_correct_count": sum(d["is_correct"] for d in test_detail),
            "test_total_count": len(test_detail),
            "average_latency_ms_per_sequence": round(avg_test_latency_ms, 2)
        }
    }

    # Save all required artifacts
    class_mapping = {
        "classes": [
            {"class_id": i, "sentence": CLASS_NAMES[i]} for i in range(5)
        ],
        "num_classes": 5
    }

    model_config = {
        "model_type": "FiveClassBiGRUAttentionModel",
        "input_dim": 300,
        "proj_dim": 128,
        "hidden_dim": 64,
        "num_layers": 1,
        "num_classes": 5,
        "dropout": 0.3,
        "parameter_count": param_count
    }

    # Save to ARTIFACTS_DIR
    with open(ARTIFACTS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    with open(ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    with open(ARTIFACTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(ARTIFACTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    with open(ARTIFACTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_detail, f, indent=2)

    # Mirror to root
    with open(ROOT_ARTIFACTS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_detail, f, indent=2)

    # Copy preprocessing_config.json to artifacts directory
    prep_src = PROJECT_ROOT / "data" / "processed" / "five_class_sequence" / "preprocessing_config.json"
    if prep_src.exists():
        with open(prep_src, "r", encoding="utf-8") as f_in:
            prep_data = json.load(f_in)
        with open(ARTIFACTS_DIR / "preprocessing_config.json", "w", encoding="utf-8") as f_out:
            json.dump(prep_data, f_out, indent=2)
        with open(ROOT_ARTIFACTS_DIR / "preprocessing_config.json", "w", encoding="utf-8") as f_out:
            json.dump(prep_data, f_out, indent=2)

    model_file_size = (ARTIFACTS_DIR / "best_model.pt").stat().st_size
    logger.info(f"Model saved to {best_checkpoint_path} (Size: {model_file_size/1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("PHASE 8: EVALUATION REPORT")
    print("=" * 60)
    print(f"Training Accuracy:   {tr_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}% (Signer 7)")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}% (Signer 2)")
    print(f"Test Top-3 Accuracy: {top3_acc * 100:.2f}%")
    print(f"Test Macro F1:       {macro_f1:.4f}")
    print(f"Model Parameters:    {param_count:,}")
    print(f"Model File Size:     {model_file_size / 1024:.1f} KB")
    print(f"Avg Inference Latency per Sequence: {avg_test_latency_ms:.2f} ms")

    print("\n--- Per-Class Test Performance ---")
    print("| Class | Sentence | Precision | Recall | F1-Score | Support |")
    print("|---|---|---:|---:|---:|---:|")
    for cname, m in per_class_metrics.items():
        print(f"| {m['class_id']} | {cname} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1_score']:.2f} | {m['support']} |")

    print("\n--- Test Confusion Matrix (Rows: Ground Truth, Cols: Predicted) ---")
    print("     " + " ".join([f"[{i}]" for i in range(5)]))
    for i, row in enumerate(conf_matrix):
        print(f"[{i}]  " + "   ".join([str(val) for val in row]))

    print("\n--- Test Predictions vs Ground Truth ---")
    for d in test_detail:
        mark = "✓" if d["is_correct"] else "✗"
        print(f"[{mark}] {d['sequence_id']}: Actual='{d['ground_truth_sentence']}' | Predicted='{d['predicted_sentence']}' (conf: {d['confidence']:.2f})")


if __name__ == "__main__":
    train()
