"""Train candidate models, track everything in MLflow, and promote the winner
to @champion only if it passes the gate, beats the current champion and stays
within the latency budget.

Run from the repo root (MLflow server must be up: `docker compose up -d --build --wait`):
    python -m training.train              # normal run / retrain
    python -m training.train --bootstrap  # first run: register a champion if the registry has none

Exit codes: 0 = promoted, or champion kept because the candidate was not better;
            1 = the candidate failed the gate (quality floor or latency budget).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from training.evaluate import (
    MAX_P95_MS,
    SLICES,
    compute_metrics,
    f1_by_slice,
    gate_checks,
    is_better,
    MAX_MODEL_SIZE_KB,
    model_size_kb,
    p95_latency_ms,
)
from training.preprocess import (
    DATA_PATH,
    FEATURES,
    ROOT,
    SEED,
    TEST_SIZE,
    build_preprocessor,
    load_data,
    split_data,
)

# ---config---
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_PATH = ROOT / "docs" / "model_comparison.md"

EXPERIMENT_NAME = "heart-disease-risk"
REGISTERED_MODEL_NAME = "heart-disease-classifier"
CHAMPION_ALIAS = "champion"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def candidate_models() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=SEED),
            {"C": 1.0},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
            {"n_estimators": 200, "max_depth": 6},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED),
            {"n_estimators": 100, "learning_rate": 0.05},
        ),
    }


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_champion_baseline(client: MlflowClient) -> tuple[float | None, str | None]:
    """Champion F1 from the MLflow registry; falls back to the committed
    models/metrics.json (the registry is empty on an ephemeral CI server)."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        f1 = client.get_run(mv.run_id).data.metrics.get("test_f1")
        if f1 is not None:
            return float(f1), "mlflow registry"
    except Exception:
        pass
    if METRICS_PATH.exists():
        try:
            return float(json.loads(METRICS_PATH.read_text())["metrics"]["f1"]), "models/metrics.json"
        except (KeyError, ValueError):
            pass
    return None, None


def comparison_markdown(results: list[dict], best: dict, data_hash: str) -> str:
    slice_info = results[0]["slices"]
    head = (
        ["Model", "CV F1", "Test F1", "Recall", "ROC-AUC"]
        + [f"F1 {s} (n={slice_info[s][1]})" for s in slice_info]
        + ["p95 ms", "Size KB", f"Within {MAX_P95_MS:g} ms", "Selected"]
    )
    lines = [
        "# Model comparison",
        "",
        f"Data SHA-256: `{data_hash}`  ",
        f"Test set: {results[0]['n_test']} rows. Slices are small, treat slice F1 as indicative.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
    ]
    for r in results:
        m = r["metrics"]
        row = (
            [r["name"], f"{r['cv_f1']:.3f}", f"{m['f1']:.3f}", f"{m['recall']:.3f}", f"{m['roc_auc']:.3f}"]
            + [f"{r['slices'][s][0]:.3f}" if s in r["slices"] else "n/a" for s in slice_info]
            + [
                f"{r['p95_ms']:.1f}",
                f"{r['size_kb']:.0f}",
                "yes" if r["p95_ms"] <= MAX_P95_MS else "NO",
                "**yes**" if r is best else "",
            ]
        )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def set_ci_output(promoted: bool) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.write(f"promoted={'true' if promoted else 'false'}\n")


# ---main---
def main(bootstrap: bool = False) -> int:
    df = load_data()
    data_hash = sha256_file(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"train={len(X_train)} test={len(X_test)} positive_rate={df['target'].mean():.2f}")
    print(f"data sha256={data_hash[:12]}...")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    results: list[dict] = []
    for name, (estimator, params) in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        with mlflow.start_run(run_name=name) as run:
            cv_f1 = float(cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean())
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            m = compute_metrics(y_test, preds, proba)
            slices = f1_by_slice(X_test, y_test, preds)
            p95 = p95_latency_ms(pipe, X_test)
            size_kb = model_size_kb(pipe)

            # params + data hash
            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_params(
                {
                    "seed": SEED,
                    "test_size": TEST_SIZE,
                    "data_sha256": data_hash,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "features": ",".join(FEATURES),
                }
            )
            # metrics
            mlflow.log_metric("cv_f1", cv_f1)
            mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
            mlflow.log_metrics({f"f1_slice_{s}": f1 for s, (f1, _) in slices.items()})
            mlflow.log_metrics({"p95_latency_ms": p95, "model_size_kb": size_kb})
            # model + signature (cloudpickle: skops in newer MLflow rejects numpy.dtype)
            info = mlflow.sklearn.log_model(
                pipe,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                signature=infer_signature(X_train, pipe.predict(X_train)),
                input_example=X_train.head(3),
            )
            results.append(
                {
                    "name": name, "run_id": run.info.run_id, "pipe": pipe, "cv_f1": cv_f1,
                    "metrics": m, "slices": slices, "p95_ms": p95, "size_kb": size_kb,
                    "model_uri": info.model_uri, "n_test": len(X_test),
                }
            )
            print(
                f"{name:20s} cv_f1={cv_f1:.3f} test_f1={m['f1']:.3f} recall={m['recall']:.3f} "
                f"p95={p95:.1f}ms size={size_kb:.0f}KB"
            )

    # Selection: best CV F1 among models inside the latency budget (CV is computed on
    # TRAIN only, so the test set never influences which model we pick).
    within_budget = [
    r for r in results
    if (
        r["p95_ms"] <= MAX_P95_MS
        and r["size_kb"] <= MAX_MODEL_SIZE_KB
    )
                   ]

    best = max(
    within_budget or results,
    key=lambda r: r["cv_f1"]
             )
    
    print(f"\nBest candidate: {best['name']} (cv_f1={best['cv_f1']:.3f})")

    report = comparison_markdown(results, best, data_hash)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    client.log_artifact(best["run_id"], str(REPORT_PATH))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report)

    # Decision: gate (quality + latency) -> better than champion? -> promote
    ok, reasons = ok, reasons = gate_checks(
    best["metrics"],
    best["p95_ms"],
    best["size_kb"],
          )
    champion_f1, source = get_champion_baseline(client)
    better = is_better(best["metrics"]["f1"], champion_f1)
    registry_empty = source != "mlflow registry"
    promote = ok and (better or (bootstrap and registry_empty))

    for r in results:
        if r is not best:
            client.set_tag(r["run_id"], "selection", "not_selected")
            client.set_tag(r["run_id"], "promotion_gate", "not_evaluated")
    client.set_tag(best["run_id"], "selection", "selected_best_cv")
    client.set_tag(best["run_id"], "promotion_gate", "passed" if ok else "failed")

    if not ok:
        client.set_tag(best["run_id"], "promotion_decision", "failed_gate")
        client.set_tag(best["run_id"], "gate_failure_reasons", "; ".join(reasons))
        print("PROMOTION GATE: FAILED")
        for reason in reasons:
            print(f"  - {reason}")
        set_ci_output(False)
        return 1

    if not promote:
        client.set_tag(best["run_id"], "promotion_decision", "kept_champion")
        print(
            f"Gate passed, but F1 {best['metrics']['f1']:.3f} is not better than "
            f"champion {champion_f1:.3f} (from {source}). Champion kept."
        )
        set_ci_output(False)
        return 0

    mv = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version)
    client.set_tag(best["run_id"], "promotion_decision", "promoted")
    print(f"PROMOTED: {REGISTERED_MODEL_NAME} v{mv.version} is now @{CHAMPION_ALIAS}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipe"], MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "model_type": best["name"],
                "mlflow_run_id": best["run_id"],
                "model_version": int(mv.version),
                "data_sha256": data_hash,
                "features": FEATURES,
                "metrics": best["metrics"],
                "f1_by_slice": {s: f1 for s, (f1, _) in best["slices"].items()},
                "p95_latency_ms": best["p95_ms"],
                "model_size_kb": best["size_kb"],
            },
            indent=2,
        )
    )
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")
    set_ci_output(True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="register the winner as @champion if the MLflow registry has no champion yet",
    )
    sys.exit(main(bootstrap=parser.parse_args().bootstrap))
