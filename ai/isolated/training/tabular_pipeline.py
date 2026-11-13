"""
Tabular Pipeline Wrapper for Isolated ISL Sign Recognition.
Wraps scikit-learn models (Random Forest, SVM) with automatic temporal feature pooling.
"""

from typing import Any, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ai.isolated.features.feature_extractor import pool_temporal_features


class TabularSequenceClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible classifier that wraps a base tabular estimator
    (RandomForestClassifier, SVC) and automatically converts (N, 64, 600)
    sequence features into (N, 2400) pooled summary representations.
    """

    def __init__(self, base_estimator: Any) -> None:
        self.base_estimator = base_estimator
        self.classes_ = getattr(base_estimator, "classes_", None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularSequenceClassifier":
        if X.ndim == 3:
            X_pooled = pool_temporal_features(X)
        else:
            X_pooled = X
        self.base_estimator.fit(X_pooled, y)
        self.classes_ = self.base_estimator.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X_pooled = pool_temporal_features(X)
        elif X.ndim == 2 and X.shape[1] == 600:
            # Single sample (T=64, 600)
            X_pooled = pool_temporal_features(X[np.newaxis, ...])
        else:
            X_pooled = X
        return self.base_estimator.predict_proba(X_pooled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X_pooled = pool_temporal_features(X)
        elif X.ndim == 2 and X.shape[1] == 600:
            X_pooled = pool_temporal_features(X[np.newaxis, ...])
        else:
            X_pooled = X
        return self.base_estimator.predict(X_pooled)
