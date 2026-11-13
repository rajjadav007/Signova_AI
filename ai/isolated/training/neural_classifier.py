"""
PyTorch Bi-directional GRU Classifier for Isolated ISL Sign Recognition.
Preserves full temporal dynamics of (64, 600) feature sequences.
"""

import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai.isolated.config.isolated_settings import DEFAULT_SEQUENCE_LENGTH, NUM_CLASSES
from ai.isolated.evaluation.metrics import compute_isolated_metrics


class IsolatedBiGRU(nn.Module):
    """
    Lightweight 2-layer Bidirectional GRU with temporal pooling.
    Input: (B, 64, 600)
    Output: (B, 10) logits
    """

    def __init__(
        self,
        input_dim: int = 600,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        gru_out, _ = self.gru(x)  # (B, T, hidden_dim * 2)
        # Temporal average pooling
        pooled = torch.mean(gru_out, dim=1)  # (B, hidden_dim * 2)
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


class GRUClassifierWrapper:
    """
    Scikit-learn compatible wrapper for IsolatedBiGRU.
    Provides predict() and predict_proba() on numpy arrays of shape (N, 64, 600).
    """

    def __init__(
        self,
        model: IsolatedBiGRU,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            # Check if flattened (N, 38400) -> reshape to (N, 64, 600)
            if X.shape[1] == DEFAULT_SEQUENCE_LENGTH * 600:
                X = X.reshape(-1, DEFAULT_SEQUENCE_LENGTH, 600)
            else:
                raise ValueError(f"Unexpected shape {X.shape} for GRUClassifierWrapper")
        elif X.ndim != 3:
            raise ValueError(f"Expected 3D array (N, 64, 600), got {X.shape}")

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def train_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Optional[np.ndarray] = None,
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    patience: int = 15,
    random_seed: int = 42,
) -> Tuple[GRUClassifierWrapper, Dict[str, Any], float]:
    """
    Trains IsolatedBiGRU with early stopping on validation macro F1.
    Returns: (best_wrapper, train_history, train_time_seconds)
    """
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

    # Class-weighted loss
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Convert data
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {
        "train_loss": [],
        "val_macro_f1": [],
        "val_accuracy": [],
    }

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Add subtle jitter augmentation during training batches
            jitter = torch.randn_like(batch_x) * 0.002
            # Apply jitter only to detected positions (where conf > 0.05)
            # Spatial coordinates are every 4th channel
            batch_x_aug = batch_x + jitter

            optimizer.zero_grad()
            logits = model(batch_x_aug)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_train_loss = float(np.mean(epoch_losses))

        # Evaluate on validation
        wrapper = GRUClassifierWrapper(model, device="cpu")
        val_probs = wrapper.predict_proba(X_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_isolated_metrics(y_val, val_preds, y_probs=val_probs)

        val_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    training_time = round(time.perf_counter() - t_start, 2)

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    best_wrapper = GRUClassifierWrapper(model, device="cpu")
    return best_wrapper, history, training_time
