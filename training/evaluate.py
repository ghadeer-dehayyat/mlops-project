"""Metrics, slice analysis, serving benchmarks and promotion-gate logic.

Free of MLflow so it is easy to unit-test. Thresholds can be overridden with
environment variables (handy for demonstrating the gate failing).
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- gate thresholds---
MIN_ACCURACY = float(os.getenv("MIN_ACCURACY", "0.80"))
MIN_F1 = float(os.getenv("MIN_F1", "0.80"))
MIN_RECALL = float(os.getenv("MIN_RECALL", "0.80"))  # missing a sick patient is the costly error
MAX_P95_MS = float(os.getenv("MAX_P95_MS", "50"))  # latency budget: single-row predict, p95
MIN_IMPROVEMENT = float(os.getenv("MIN_IMPROVEMENT", "0.0"))  # F1 must beat champion by more than this
MAX_MODEL_SIZE_KB = float(os.getenv("MAX_MODEL_SIZE_KB", "500"))

# --- data slices---
SLICES = {
    "sex_female": lambda X: X["sex"] == 0,
    "sex_male": lambda X: X["sex"] == 1,
    "age_lt_55": lambda X: X["age"] < 55,
    "age_ge_55": lambda X: X["age"] >= 55,
}


def compute_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return {k: float(v) for k, v in metrics.items()}


def f1_by_slice(X, y_true, y_pred) -> dict[str, tuple[float, int]]:
    """F1 per data slice -> {slice_name: (f1, n_rows)}."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {}
    for name, rule in SLICES.items():
        mask = rule(X).to_numpy()
        if mask.sum() > 0:
            out[name] = (float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), int(mask.sum()))
    return out


def p95_latency_ms(pipeline, X, n_calls: int = 200) -> float:
    """p95 latency (ms) of predicting ONE row at a time, like the API does."""
    rows = [X.iloc[[i % len(X)]] for i in range(n_calls)]
    pipeline.predict(rows[0])  # warm-up
    times = []
    for row in rows:
        start = time.perf_counter()
        pipeline.predict(row)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 95))


def model_size_kb(pipeline) -> float:
    """Size of the serialized pipeline on disk, in KB."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        joblib.dump(pipeline, path)
        return path.stat().st_size / 1024


# ---the gate---
def gate_checks(
    metrics: dict[str, float],
    p95_ms: float,
    size_kb: float,
    ) -> tuple[bool, list[str]]:

    reasons: list[str] = []

    if metrics["accuracy"] < MIN_ACCURACY:
        reasons.append(
            f"accuracy {metrics['accuracy']:.3f} < {MIN_ACCURACY}"
        )

    if metrics["f1"] < MIN_F1:
        reasons.append(
            f"f1 {metrics['f1']:.3f} < {MIN_F1}"
        )

    if metrics["recall"] < MIN_RECALL:
        reasons.append(
            f"recall {metrics['recall']:.3f} < {MIN_RECALL}"
        )

    if p95_ms > MAX_P95_MS:
        reasons.append(
            f"p95 latency {p95_ms:.1f} ms > budget {MAX_P95_MS} ms"
        )

    if size_kb > MAX_MODEL_SIZE_KB:
        reasons.append(
            f"model size {size_kb:.1f} KB > budget {MAX_MODEL_SIZE_KB} KB"
        )

    return (len(reasons) == 0, reasons)


def is_better(candidate_f1: float, champion_f1: float | None) -> bool:
    """True if there is no champion yet, or the candidate beats it."""
    return champion_f1 is None or candidate_f1 > champion_f1 + MIN_IMPROVEMENT
