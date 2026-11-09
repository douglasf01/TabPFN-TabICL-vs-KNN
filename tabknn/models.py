"""The three models, all with the scikit-learn fit/predict interface."""

from sklearn.neighbors import KNeighborsClassifier
from contextlib import contextmanager
import os
from unittest.mock import patch

MODEL_NAMES = ["knn", "tabpfn", "tabicl"]


def make_model(name: str, k: int = 5, seed: int = 0, n_estimators: int = 1):
    """Use one estimator by default; KNN always uses one classifier."""
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=k)
    if name == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(random_state=seed, n_estimators=n_estimators)
    if name == "tabicl":
        from tabicl import TabICLClassifier
        return TabICLClassifier(random_state=seed, n_estimators=n_estimators)
    raise ValueError(f"Unknown model '{name}'")


@contextmanager
def cached_model_weights():
    """Reuse fixed network weights, while each estimator refits its preprocessing.

    Single-process inference only. No dataset representations or predictions
    are cached. TabPFN provides its own built-model cache; TabICL's loader is
    temporarily wrapped to retain just the immutable network and its config.
    """
    from tabicl import TabICLClassifier
    original = TabICLClassifier._load_model
    cached = {}
    old = os.environ.get("TABPFN_MODEL_CACHE_SIZE")
    os.environ["TABPFN_MODEL_CACHE_SIZE"] = "1"

    def load(estimator):
        key = (str(estimator.model_path), estimator.checkpoint_version)
        if key not in cached:
            original(estimator)
            cached[key] = (estimator.model_, estimator.model_config_, estimator.model_path_)
        estimator.model_, estimator.model_config_, estimator.model_path_ = cached[key]
    try:
        with patch.object(TabICLClassifier, "_load_model", load):
            yield
    finally:
        if old is None:
            os.environ.pop("TABPFN_MODEL_CACHE_SIZE", None)
        else:
            os.environ["TABPFN_MODEL_CACHE_SIZE"] = old
