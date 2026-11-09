"""Load the five TabArena datasets from OpenML and turn them into numeric arrays."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# OpenML dataset ids. Not all of them have been checked against the TabArena-v0.1
# copies (see Limitations in the README).
DATASETS = {
    "blood-transfusion": 1464,
    "diabetes": 37,
    "maternal-health-risk": 46941,  # v3, the TabArena copy (v1 46600 / v2 46842 hold the same rows)
    "credit-g": 31,
    "qsar-biodeg": 1494,
}


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return features as a DataFrame and labels as integers 0..C-1."""
    import openml

    data_id = DATASETS[name]
    if data_id is None:
        raise ValueError(f"No OpenML id set for '{name}' yet (see DATASETS).")
    openml.config.set_root_cache_directory(str(DATA_DIR / "openml_cache"))
    dataset = openml.datasets.get_dataset(data_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    y = LabelEncoder().fit_transform(y)
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals, impute, and standardise.

    All three models see the same numeric features, so KNN distances and the
    foundation models' attention live in the same input space.
    """
    categorical = X.select_dtypes(include=["category", "object", "bool"]).columns
    numeric = X.columns.difference(categorical)
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(numeric)),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
         list(categorical)),
    ])


def get_split(name: str, seed: int = 0, test_size: float = 0.2):
    """Load a dataset and return a preprocessed, stratified train/test split."""
    X, y = load_dataset(name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    pre = make_preprocessor(X_train)
    X_train = pre.fit_transform(X_train).astype(np.float32)
    X_test = pre.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test
