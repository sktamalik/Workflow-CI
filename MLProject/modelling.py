"""
modelling.py (MLProject)
========================
Training model untuk Workflow-CI (Kriteria 3).

- Dataset: diabetes_preprocessing.csv (sudah ter-preprocess, siap dilatih)
- Model: RandomForest dengan tuning ringan GridSearchCV
- Logging MLflow (online DagsHub bila DAGSHUB_TOKEN ada, else local)
- Output: model artifact logged + confusion matrix/ROC/feature importance

Usage:
    python modelling.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split

DATA_PATH = Path(__file__).resolve().parent / "diabetes_preprocessing.csv"
TARGET = "Outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.2

DAGSHUB_TOKEN = os.environ.get("DAGSHUB_USER_TOKEN", "") or os.environ.get("DAGSHUB_TOKEN", "")


def setup_tracking():
    """MLflow online DagsHub bila token ada, fallback local."""
    # `mlflow run` menyuntik MLFLOW_RUN_ID yang hanya valid di backend lama.
    # Hapus agar tidak RESOURCE_DOES_NOT_EXIST setelah tracking URI diganti.
    os.environ.pop("MLFLOW_RUN_ID", None)
    if DAGSHUB_TOKEN:
        import dagshub
        # dagshub client membaca app token dari env DAGSHUB_USER_TOKEN
        os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN
        dagshub.init(repo_owner="sktamalik", repo_name="diabetes-mlflow", mlflow=True)
        print("Tracking: DagsHub online")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        print("Tracking: local 127.0.0.1:5000")


def train():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    setup_tracking()
    mlflow.set_experiment("Workflow-CI")

    with mlflow.start_run(run_name="workflow-ci-training") as run:
        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Manual logging (sesuai kriteria, no autolog)
        mlflow.log_params({f"grid_{k}": v for k, v in grid.best_params_.items()})
        mlflow.log_params({"model_type": "RandomForest", "tuning": "GridSearchCV", "cv_k": 5})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_best_mean_test_score", grid.best_score_)

        # Artefak: confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title("Confusion Matrix - Workflow CI")
        plt.tight_layout()
        fig.savefig("confusion_matrix.png")
        plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png", artifact_path="artifacts")

        # Artefak: ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        ax.set_title("ROC - Workflow CI")
        plt.tight_layout()
        fig.savefig("roc_curve.png")
        plt.close(fig)
        mlflow.log_artifact("roc_curve.png", artifact_path="artifacts")

        # Artefak: feature importance
        imp = best.feature_importances_
        idx = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([X.columns[i] for i in idx][::-1], imp[idx][::-1])
        ax.set_title("Feature Importance - Workflow CI")
        plt.tight_layout()
        fig.savefig("feature_importance.png")
        plt.close(fig)
        mlflow.log_artifact("feature_importance.png", artifact_path="artifacts")

        # Log model (dengan signature via input example)
        mlflow.sklearn.log_model(
            best,
            "model",
            input_example=X_test.iloc[:1].to_dict(orient="records"),
        )

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(f"Best params: {grid.best_params_}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train()