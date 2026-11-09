"""Nearest neighbor sets and the top-k overlap metric."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_indices(X_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest training points for each test point, shape (n_test, k)."""
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    return nn.kneighbors(X_test, return_distance=False)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, e.g. attention, shape (n_test, k)."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def overlap_at_k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Overlap_k = |N_KNN ∩ N_Attn| / k, one value per test point."""
    k = a.shape[1]
    return np.array([len(set(ra) & set(rb)) / k for ra, rb in zip(a, b)])
