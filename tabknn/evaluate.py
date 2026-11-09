"""Evaluation: full-context metrics and metrics with the context cut to the k nearest neighbors."""

import numpy as np
from pathlib import Path
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

from .models import make_model
from .neighbors import knn_indices


def classification_metrics(y_true, y_pred, proba=None) -> dict:
    """Accuracy plus metrics that are not inflated by predicting the majority class.

    f1_minority is the F1 of the least frequent test class. roc_auc needs
    class probabilities (columns ordered 0..C-1); with more than two classes it
    is the macro one-vs-rest average.
    """
    minority = int(np.bincount(y_true).argmin())
    metrics = dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_macro=f1_score(y_true, y_pred, average="macro", zero_division=0),
        f1_minority=f1_score(y_true, y_pred, labels=[minority], average="macro", zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
    )
    if proba is not None:
        metrics["roc_auc"] = (roc_auc_score(y_true, proba[:, 1]) if proba.shape[1] == 2
                              else roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    return metrics


def full_context_metrics(model_name, X_train, X_test, y_train, y_test, k=5, seed=0, n_estimators=1) -> dict:
    """Fit on the whole training set, score on the test set."""
    model = make_model(model_name, k=k, seed=seed, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return classification_metrics(y_test, model.predict(X_test), model.predict_proba(X_test))


def restricted_context_accuracy(model_name, X_train, X_test, y_train, y_test, k, seed=0,
                                n_estimators=1, checkpoint_path=None, progress=None) -> dict:
    """For each test point, give the model only its k nearest training points.

    The model is skipped, and the neighbors' majority label predicted, when
    - all k neighbors share one label (most models refuse a single class), or
    - all k neighbors have identical features (TabPFN refuses constant features).
    The returned dict counts how often each shortcut was taken.
    """
    neighbors = knn_indices(X_train, X_test, k)
    preds = np.zeros(len(X_test), dtype=y_train.dtype)
    one_class = np.zeros(len(X_test), dtype=bool)
    constant = np.zeros(len(X_test), dtype=bool)
    start = 0
    signature = np.array([len(X_train), len(X_test), k, seed, n_estimators])
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path and checkpoint_path.exists():
        with np.load(checkpoint_path) as saved:
            np.testing.assert_array_equal(saved["signature"], signature)
            preds[:] = saved["preds"]
            one_class[:] = saved["one_class"]
            constant[:] = saved["constant"]
            start = int(saved["completed"])

    def save(completed):
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp.npz")
            np.savez(temporary, preds=preds, one_class=one_class, constant=constant,
                     completed=completed, signature=signature)
            temporary.replace(checkpoint_path)
        if progress:
            progress(completed, len(X_test))

    for i in range(start, len(X_test)):
        idx = neighbors[i]
        labels = y_train[idx]
        if len(np.unique(labels)) == 1:
            preds[i] = labels[0]
            one_class[i] = True
        elif np.all(np.ptp(X_train[idx], axis=0) == 0):
            preds[i] = np.bincount(labels).argmax()
            constant[i] = True
        else:
            model = make_model(model_name, k=min(k, len(idx)), seed=seed, n_estimators=n_estimators)
            model.fit(X_train[idx], labels)
            preds[i] = model.predict(X_test[i : i + 1])[0]
        if (i + 1) % 20 == 0 or i + 1 == len(X_test):
            save(i + 1)
    n_one_class, n_constant = int(one_class.sum()), int(constant.sum())
    n = len(X_test)
    return dict(
        **classification_metrics(y_test, preds),
        frac_one_class=n_one_class / n,
        frac_constant_features=n_constant / n,
        frac_model_used=(n - n_one_class - n_constant) / n,
    )
