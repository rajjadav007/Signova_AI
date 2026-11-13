"""
Signova AI - 25-Class ISL Sentence Sequence Classifier Training and Evaluation.
Phases 6, 7, 8, and 10:
- BiGRU with Bahdanau Self-Attention Pooling.
- Signer-independent data splitting:
    Train: Signers 1, 3, 4, 5, 6 (125 sequences)
    Validation: Signer 7 (25 sequences)
    Test: Signer 2 (25 sequences) - strictly quarantined until final evaluation.
- Evaluates Top-1, Top-5, Macro F1, Per-class metrics, confusion matrices, and latency.
- Saves all artifacts to models/isl_csltr_continuous/twenty_five_class_sequence/.
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

from ai.continuous.twenty_five_class_model import TwentyFiveClassBiGRUAttentionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_twenty_five_class")

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "twenty_five_class_sequence" / "twenty_five_class_dataset.npz"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "isl_csltr_continuous" / "twenty_five_class_sequence"
ROOT_ARTIFACTS_DIR = Path(r"D:\Projects\models\isl_csltr_continuous\twenty_five_class_sequence")

CLASS_NAMES = [
    "comb your hair",
    "help me",
    "i am fine. thank you sir.",
    "i need water",
    "who are you",
    "are you free today",
    "congratulations",
    "do not worry",
    "go and sleep",
    "had your food",
    "he came by train",
    "he she is my friend",
    "hi how are you",
    "how old are you",
    "i am crying",
    "i am feeling bored",
    "i am feeling cold",
    "i am hungry",
    "i am sitting in the class",
    "i like you i love you",
    "i promise",
    "shall we go outside",
    "take care of yourself",
    "thank you so much",
    "what are you doing"
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
            x_resampled = np.zeros((new_t, d), dtype=np.float32)
            orig_indices = np.arange(t)
            for j in range(d):
                x_resampled[:, j] = np.interp(indices, orig_indices, x[:, j])
            x = x_resampled
            t = new_t

        # 2. Gaussian coordinate noise on present landmarks
        if random.random() < 0.7:
            x_reshaped = x.reshape(t, 75, 4).copy()
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
        return torch.from_numpy(feat).float(), int(self.labels[idx]), int(self.signers[idx]), str(self.seq_ids[idx])


def pad_collate_fn(batch):
    """
    Pad sequences to max sequence length in batch and return attention mask.
    """
    sequences, labels, signers, seq_ids = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[-1]

    padded = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len] = seq
        mask[i, :seq_len] = True

    labels = torch.tensor(labels, dtype=torch.long)
    signers = torch.tensor(signers, dtype=torch.long)

    return padded, mask, labels, signers, seq_ids


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_seq_ids = []
    inference_times_ms = []

    with torch.no_grad():
        for padded, mask, labels, _, seq_ids in dataloader:
            padded = padded.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            t0 = time.perf_counter()
            logits, _ = model(padded, mask=mask)
            elapsed = (time.perf_counter() - t0) * 1000.0
            inference_times_ms.append(elapsed / len(labels))

            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_seq_ids.extend(seq_ids)

    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    top1_correct = int(np.sum(all_preds == all_labels))
    top1_accuracy = float(top1_correct / n_samples)

    # Top-5 accuracy
    top5_correct = 0
    for i in range(n_samples):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if all_labels[i] in top5_indices:
            top5_correct += 1
    top5_accuracy = float(top5_correct / n_samples)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return {
        "loss": float(avg_loss),
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "mean_latency_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        "predictions": all_preds.tolist(),
        "ground_truth": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
        "sequence_ids": all_seq_ids,
    }


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Aggregated dataset not found at {DATA_PATH}. Run extract_twenty_five_class_features.py first.")

    raw_data = np.load(DATA_PATH, allow_pickle=True)
    train_feats = list(raw_data["train_features"])
    train_labels = raw_data["train_labels"]
    train_signers = raw_data["train_signers"]
    train_seq_ids = raw_data["train_seq_ids"]

    val_feats = list(raw_data["val_features"])
    val_labels = raw_data["val_labels"]
    val_signers = raw_data["val_signers"]
    val_seq_ids = raw_data["val_seq_ids"]

    test_feats = list(raw_data["test_features"])
    test_labels = raw_data["test_labels"]
    test_signers = raw_data["test_signers"]
    test_seq_ids = raw_data["test_seq_ids"]

    logger.info(f"Loaded dataset: Train={len(train_feats)} seqs, Val={len(val_feats)} seqs, Test={len(test_feats)} seqs")

    # Verify signer isolation
    train_s = set(int(s) for s in train_signers)
    val_s = set(int(s) for s in val_signers)
    test_s = set(int(s) for s in test_signers)

    assert not (train_s & val_s), "Leakage between train and val signers!"
    assert not (train_s & test_s), "Leakage between train and test signers!"
    assert not (val_s & test_s), "Leakage between val and test signers!"
    logger.info(f"Signer isolation confirmed: Train={train_s}, Val={val_s}, Test={test_s}")

    train_ds = SequenceDataset(train_feats, train_labels, train_signers, train_seq_ids, augment=True)
    val_ds = SequenceDataset(val_feats, val_labels, val_signers, val_seq_ids, augment=False)
    test_ds = SequenceDataset(test_feats, test_labels, test_signers, test_seq_ids, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=pad_collate_fn)

    model = TwentyFiveClassBiGRUAttentionModel(
        input_dim=300,
        proj_dim=128,
        hidden_dim=64,
        num_layers=1,
        num_classes=25,
        dropout=0.3
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized TwentyFiveClassBiGRUAttentionModel with {param_count:,} trainable parameters.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )

    epochs = 75
    best_val_score = -1.0
    best_epoch = -1
    best_state_dict = None
    patience = 20
    no_improve_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": []
    }

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0

        for padded, mask, labels, _, _ in train_loader:
            padded = padded.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(padded, mask=mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item() * len(labels)

        avg_train_loss = train_loss_total / len(train_feats)
        val_eval = evaluate(model, val_loader, device, eval_criterion)

        scheduler.step(val_eval["top1_accuracy"])
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(val_eval["loss"]))
        history["val_acc"].append(float(val_eval["top1_accuracy"]))
        history["val_macro_f1"].append(float(val_eval["macro_f1"]))
        history["lr"].append(float(curr_lr))

        score = val_eval["top1_accuracy"] + 0.1 * val_eval["macro_f1"] - 0.05 * val_eval["loss"]
        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            logger.info(
                f"Epoch {epoch:03d}: Train Loss={avg_train_loss:.4f} | "
                f"Val Loss={val_eval['loss']:.4f} | Val Acc={val_eval['top1_accuracy']*100:.2f}% | "
                f"Val Top5={val_eval['top5_accuracy']*100:.2f}% | Val F1={val_eval['macro_f1']:.4f} [BEST]"
            )
        else:
            no_improve_epochs += 1
            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch:03d}: Train Loss={avg_train_loss:.4f} | "
                    f"Val Loss={val_eval['loss']:.4f} | Val Acc={val_eval['top1_accuracy']*100:.2f}% | "
                    f"Val F1={val_eval['macro_f1']:.4f}"
                )

        if no_improve_epochs >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_train_time = time.time() - t_start
    logger.info(f"Training completed in {total_train_time:.1f}s. Best validation epoch: {best_epoch}")

    # Load best checkpoint
    model.load_state_dict(best_state_dict)
    model.to(device)

    # Final Validation Evaluation
    final_val_eval = evaluate(model, val_loader, device, eval_criterion)
    logger.info(f"Final Val Top-1 Acc: {final_val_eval['top1_accuracy']*100:.2f}%, Top-5: {final_val_eval['top5_accuracy']*100:.2f}%, F1: {final_val_eval['macro_f1']:.4f}")

    # Final Test Set Evaluation (Signer 2 quarantined until now)
    final_test_eval = evaluate(model, test_loader, device, eval_criterion)
    logger.info(f"Final Test Top-1 Acc: {final_test_eval['top1_accuracy']*100:.2f}%, Top-5: {final_test_eval['top5_accuracy']*100:.2f}%, F1: {final_test_eval['macro_f1']:.4f}")

    # Save best_model.pt
    torch.save(best_state_dict, ARTIFACTS_DIR / "best_model.pt")
    torch.save(best_state_dict, ROOT_ARTIFACTS_DIR / "best_model.pt")
    logger.info(f"Saved best_model.pt to {ARTIFACTS_DIR / 'best_model.pt'}")

    # Save class_mapping.json
    class_mapping = {
        "classes": [{"class_id": i, "sentence": name} for i, name in enumerate(CLASS_NAMES)],
        "num_classes": 25
    }
    with open(ARTIFACTS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    # Save model_config.json
    model_config = model.get_config()
    with open(ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # Copy / update preprocessing_config.json
    prep_path = PROJECT_ROOT / "data" / "processed" / "twenty_five_class_sequence" / "preprocessing_config.json"
    if prep_path.exists():
        with open(prep_path, "r", encoding="utf-8") as f:
            prep_cfg = json.load(f)
    else:
        prep_cfg = {
            "num_classes": 25,
            "classes": [{"class_id": i, "name": n} for i, n in enumerate(CLASS_NAMES)],
            "landmark_count": 75,
            "landmark_channels": 4,
            "feature_dim": 300,
            "normalization": "torso_centered",
            "missing_hand_strategy": "zero_fill"
        }
    with open(ARTIFACTS_DIR / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(prep_cfg, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump(prep_cfg, f, indent=2)

    # Save evaluation_metrics.json
    metrics = {
        "model_type": "TwentyFiveClassBiGRUAttentionModel",
        "num_classes": 25,
        "best_epoch": best_epoch,
        "train_time_seconds": round(total_train_time, 2),
        "validation_metrics": {
            "top1_accuracy": round(final_val_eval["top1_accuracy"], 4),
            "top5_accuracy": round(final_val_eval["top5_accuracy"], 4),
            "macro_f1": round(final_val_eval["macro_f1"], 4),
            "macro_precision": round(final_val_eval["macro_precision"], 4),
            "macro_recall": round(final_val_eval["macro_recall"], 4),
            "loss": round(final_val_eval["loss"], 4),
            "mean_latency_ms": round(final_val_eval["mean_latency_ms"], 2),
            "sample_count": len(val_labels),
            "signer": "Signer 7"
        },
        "test_metrics": {
            "top1_accuracy": round(final_test_eval["top1_accuracy"], 4),
            "top5_accuracy": round(final_test_eval["top5_accuracy"], 4),
            "macro_f1": round(final_test_eval["macro_f1"], 4),
            "macro_precision": round(final_test_eval["macro_precision"], 4),
            "macro_recall": round(final_test_eval["macro_recall"], 4),
            "loss": round(final_test_eval["loss"], 4),
            "mean_latency_ms": round(final_test_eval["mean_latency_ms"], 2),
            "sample_count": len(test_labels),
            "signer": "Signer 2"
        }
    }
    with open(ARTIFACTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save test_predictions.json and training_history.json
    test_preds_summary = []
    for i in range(len(test_labels)):
        probs = final_test_eval["probabilities"][i]
        top5_idx = np.argsort(probs)[-5:][::-1].tolist()
        test_preds_summary.append({
            "sequence_id": final_test_eval["sequence_ids"][i],
            "ground_truth_id": int(test_labels[i]),
            "ground_truth_sentence": CLASS_NAMES[test_labels[i]],
            "predicted_id": int(final_test_eval["predictions"][i]),
            "predicted_sentence": CLASS_NAMES[final_test_eval["predictions"][i]],
            "confidence": round(float(probs[final_test_eval["predictions"][i]]), 4),
            "top5_candidates": [
                {"class_id": idx, "sentence": CLASS_NAMES[idx], "probability": round(float(probs[idx]), 4)}
                for idx in top5_idx
            ],
            "correct": bool(test_labels[i] == final_test_eval["predictions"][i])
        })

    with open(ARTIFACTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_preds_summary, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_preds_summary, f, indent=2)

    with open(ARTIFACTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(ROOT_ARTIFACTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("All 25-class artifacts successfully saved!")


if __name__ == "__main__":
    train()
