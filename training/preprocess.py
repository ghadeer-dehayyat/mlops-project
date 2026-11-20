"""Data loading, preprocessing and train/test split.

Used by training/train.py. Contains no model or MLflow code.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---config---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart.csv"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"

SEED = 42
TEST_SIZE = 0.2


# --- data---
def load_data() -> pd.DataFrame:
    """Load the UCI Cleveland heart-disease data (cached in data/heart.csv)."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = DATA_PATH.with_suffix(".raw")
        print(f"Downloading dataset from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, raw)
        df = pd.read_csv(raw, names=COLUMNS, na_values="?")
        df[TARGET] = (df[TARGET] > 0).astype(int)  # 0 = healthy, 1 = disease
        df.to_csv(DATA_PATH, index=False)
        raw.unlink()
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    """Stratified 80/20 train/test split."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot categorical columns."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )
