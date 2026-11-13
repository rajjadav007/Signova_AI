"""
Dynamic Time Warping + K-Nearest Neighbors (DTW + KNN) Classifier for ISL Sentences.

Module: ai.continuous.dtw_knn_classifier
Task: 2.2A - Direct temporal sequence matching baseline for ISL sentence recognition.
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist


def safe_cosine_dist(A: np.ndarray, B: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix between frame arrays A and B safely.
    Handles zero vectors (missing landmarks / zero velocity/accel) without NaNs:
    - If both frame vectors are zero: distance is 0.0.
    - If one frame vector is zero and other non-zero: distance is 1.0.
    - Otherwise: standard cosine distance clipped to [0.0, 2.0].
    """
    normA = np.linalg.norm(A, axis=1, keepdims=True)
    normB = np.linalg.norm(B, axis=1, keepdims=True)
    zeroA = (normA < eps).squeeze(-1)
    zeroB = (normB < eps).squeeze(-1)
    normA_safe = np.where(zeroA[:, None], 1.0, normA)
    normB_safe = np.where(zeroB[:, None], 1.0, normB)
    sim = np.dot(A / normA_safe, (B / normB_safe).T)
    dist = 1.0 - sim
    both_zero = np.outer(zeroA, zeroB)
    one_zero = np.bitwise_xor(zeroA[:, None], zeroB[None, :])
    dist = np.where(both_zero, 0.0, dist)
    dist = np.where(one_zero, 1.0, dist)
    return np.clip(dist, 0.0, 2.0).astype(np.float32)


def compute_dtw_distance(
    A: np.ndarray,
    B: np.ndarray,
    metric: str = "cosine",
    window: int = 10,
    group_weights: Optional[Tuple[float, float, float]] = None,
    group_slices: Optional[Tuple[slice, slice, slice]] = None,
) -> float:
    """
    Compute length-normalized DTW distance with Sakoe-Chiba constraint band.
    Supports optional group-weighted feature distance (spatial, velocity, acceleration).

    Args:
        A: Sequence of shape (T1, D).
        B: Sequence of shape (T2, D).
        metric: Distance metric for frame pairs ('cosine', 'euclidean', etc.).
        window: Sakoe-Chiba constraint window width.
        group_weights: Optional tuple of weights (spatial, velocity, accel).
        group_slices: Optional tuple of slices corresponding to each group.

    Returns:
        float: Length-normalized dynamic time warping distance: dp[T1, T2] / (T1 + T2).
    """
    T1 = A.shape[0]
    T2 = B.shape[0]
    if T1 == 0 or T2 == 0:
        return 1e6

    # Pairwise frame cost matrix (T1, T2)
    if group_weights is not None and A.shape[1] == 48:
        if group_slices is None:
            group_slices = (slice(0, 24), slice(24, 40), slice(40, 48))
        w_total = float(sum(group_weights))
        if w_total <= 0.0:
            w_total = 1.0
        C = np.zeros((T1, T2), dtype=np.float32)
        for w, sl in zip(group_weights, group_slices):
            if w <= 0.0:
                continue
            A_sub = A[:, sl]
            B_sub = B[:, sl]
            if metric == "cosine":
                c_sub = safe_cosine_dist(A_sub, B_sub)
            else:
                c_sub = cdist(A_sub, B_sub, metric=metric).astype(np.float32)
            C += (float(w) / w_total) * c_sub
    else:
        if metric == "cosine":
            C = safe_cosine_dist(A, B)
        else:
            C = cdist(A, B, metric=metric).astype(np.float32)

    # Dynamic programming cost matrix
    dp = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    w = max(window, abs(T1 - T2))
    for i in range(1, T1 + 1):
        j_start = max(1, i - w)
        j_end = min(T2, i + w) + 1
        for j in range(j_start, j_end):
            c = C[i - 1, j - 1]
            m = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
            dp[i, j] = c + m

    return float(dp[T1, T2] / (T1 + T2))


def _compute_chunk_dists(
    args: Tuple[np.ndarray, List[np.ndarray], int, str, int, Optional[Tuple[float, float, float]], Optional[Tuple[slice, slice, slice]]]
) -> Tuple[int, int, np.ndarray]:
    query_seq, chunk_train_seqs, start_idx, metric, window, group_weights, group_slices = args
    n_chunk = len(chunk_train_seqs)
    dists = np.empty(n_chunk, dtype=np.float32)
    for j in range(n_chunk):
        dists[j] = compute_dtw_distance(
            query_seq,
            chunk_train_seqs[j],
            metric=metric,
            window=window,
            group_weights=group_weights,
            group_slices=group_slices,
        )
    return start_idx, start_idx + n_chunk, dists


class DTWKNNClassifier:
    """
    Non-parametric DTW + KNN Classifier for variable-length sequence recognition.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "cosine",
        window: int = 10,
        weighting: str = "distance",
        num_classes: int = 101,
        eps: float = 1e-5,
        group_weights: Optional[Tuple[float, float, float]] = None,
        group_slices: Optional[Tuple[slice, slice, slice]] = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.window = window
        self.weighting = weighting
        self.num_classes = num_classes
        self.eps = eps
        self.group_weights = group_weights
        self.group_slices = group_slices

        self.train_seqs: List[np.ndarray] = []
        self.train_labels: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X_train: List[np.ndarray], y_train: Union[np.ndarray, List[int]]) -> DTWKNNClassifier:
        """Store training exemplars and ground-truth labels."""
        self.train_seqs = [np.asarray(s, dtype=np.float32) for s in X_train]
        self.train_labels = np.asarray(y_train, dtype=np.int64)
        self.is_fitted = True
        return self

    def compute_distance_matrix(
        self,
        X_queries: List[np.ndarray],
        max_workers: int = 4,
    ) -> np.ndarray:
        """
        Compute pairwise DTW distance matrix of shape (N_queries, N_train).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before computing distances.")

        n_q = len(X_queries)
        n_tr = len(self.train_seqs)
        dist_matrix = np.empty((n_q, n_tr), dtype=np.float32)

        if n_q == 1 and max_workers > 1 and n_tr > 1:
            # Parallelize single query across exemplars
            query = X_queries[0]
            chunk_size = (n_tr + max_workers - 1) // max_workers
            tasks = []
            for w in range(max_workers):
                s = w * chunk_size
                e = min(n_tr, s + chunk_size)
                if s < e:
                    tasks.append((query, self.train_seqs[s:e], s, self.metric, self.window, self.group_weights, self.group_slices))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_compute_chunk_dists, t) for t in tasks]
                for f in as_completed(futures):
                    s, e, chunk_dists = f.result()
                    dist_matrix[0, s:e] = chunk_dists
            return dist_matrix

        if max_workers <= 1:
            for i in range(n_q):
                for j in range(n_tr):
                    dist_matrix[i, j] = compute_dtw_distance(
                        X_queries[i],
                        self.train_seqs[j],
                        metric=self.metric,
                        window=self.window,
                        group_weights=self.group_weights,
                        group_slices=self.group_slices,
                    )
        else:
            tasks = []
            for i in range(n_q):
                tasks.append((X_queries[i], self.train_seqs, 0, self.metric, self.window, self.group_weights, self.group_slices))
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_compute_chunk_dists, tasks[i]): i for i in range(n_q)}
                for f in as_completed(futures):
                    row_idx = futures[f]
                    _, _, row_dists = f.result()
                    dist_matrix[row_idx, :] = row_dists

        return dist_matrix

    def predict_from_distances(
        self,
        dist_matrix: np.ndarray,
        k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels and posterior probabilities from precomputed distance matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (preds of shape (N,), probs of shape (N, num_classes))
        """
        if k is None:
            k = self.n_neighbors

        n_samples = dist_matrix.shape[0]
        preds = np.zeros(n_samples, dtype=np.int64)
        probs = np.zeros((n_samples, self.num_classes), dtype=np.float32)

        for i in range(n_samples):
            row_dists = dist_matrix[i]
            nearest_idx = np.argsort(row_dists)[:k]
            nearest_classes = self.train_labels[nearest_idx]
            nearest_dists = row_dists[nearest_idx]

            if k == 1 or self.weighting == "uniform":
                # Uniform voting
                votes: Dict[int, float] = {}
                for cls in nearest_classes:
                    votes[cls] = votes.get(cls, 0.0) + 1.0
                total_votes = float(sum(votes.values()))
                for cls, v in votes.items():
                    probs[i, cls] = v / total_votes
                preds[i] = max(votes.keys(), key=lambda c: votes[c])
            else:
                # Inverse distance weighting
                weights = 1.0 / (nearest_dists + self.eps)
                class_weights: Dict[int, float] = {}
                for cls, w in zip(nearest_classes, weights):
                    class_weights[cls] = class_weights.get(cls, 0.0) + float(w)
                total_w = float(sum(class_weights.values()))
                for cls, w in class_weights.items():
                    probs[i, cls] = w / total_w
                preds[i] = max(class_weights.keys(), key=lambda c: class_weights[c])

        return preds, probs

    def predict_proba(
        self,
        X: List[np.ndarray],
        max_workers: int = 4,
    ) -> np.ndarray:
        """Compute posterior probabilities for query sequences."""
        dists = self.compute_distance_matrix(X, max_workers=max_workers)
        _, probs = self.predict_from_distances(dists)
        return probs

    def predict(
        self,
        X: List[np.ndarray],
        max_workers: int = 4,
    ) -> np.ndarray:
        """Predict class labels for query sequences."""
        dists = self.compute_distance_matrix(X, max_workers=max_workers)
        preds, _ = self.predict_from_distances(dists)
        return preds

    def save(self, path: Union[str, Path]) -> None:
        """Serialize fitted model to pickle."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_neighbors": self.n_neighbors,
            "metric": self.metric,
            "window": self.window,
            "weighting": self.weighting,
            "num_classes": self.num_classes,
            "eps": self.eps,
            "group_weights": self.group_weights,
            "group_slices": self.group_slices,
            "train_seqs": self.train_seqs,
            "train_labels": self.train_labels,
        }
        with open(save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> DTWKNNClassifier:
        """Deserialize model from pickle."""
        load_path = Path(path)
        with open(load_path, "rb") as f:
            payload = pickle.load(f)

        clf = cls(
            n_neighbors=payload["n_neighbors"],
            metric=payload["metric"],
            window=payload["window"],
            weighting=payload["weighting"],
            num_classes=payload["num_classes"],
            eps=payload.get("eps", 1e-5),
            group_weights=payload.get("group_weights", None),
            group_slices=payload.get("group_slices", None),
        )
        clf.fit(payload["train_seqs"], payload["train_labels"])
        return clf
