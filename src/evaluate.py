"""Stage 4 (evaluate): score the model, write metrics, and log the run to MLflow.

DVC reads its metrics (`dvc metrics show`, `dvc exp show`) from
metrics/metrics.json, a plain file in the repo. The same params and metrics,
plus the plots, are also logged to MLflow so you can browse and compare runs in
its local UI (`mlflow ui`).

Reads:  models/model.pkl, data/features/test.pkl
Writes: metrics/metrics.json, plus MLflow tracking data (gitignored)
"""

import json
import pickle
from pathlib import Path

import mlflow
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    roc_auc_score,
)

METRICS_OUT = Path("metrics/metrics.json")


def main() -> None:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/features/test.pkl", "rb") as f:
        X_test, y_test = pickle.load(f)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": round(float(f1_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
    }

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    # Log the same run to MLflow. Swapping trackers (W&B, Neptune, ...) means
    # changing only this import and these log calls.
    mlflow.set_experiment("churn")
    with mlflow.start_run():
        mlflow.log_params(yaml.safe_load(open("params.yaml"))["train"])
        mlflow.log_metrics(metrics)
        cm = ConfusionMatrixDisplay.from_predictions(y_test, preds)
        mlflow.log_figure(cm.figure_, "confusion_matrix.png")
        roc = RocCurveDisplay.from_predictions(y_test, probs)
        mlflow.log_figure(roc.figure_, "roc_curve.png")

    print(f"evaluate: {metrics}")


if __name__ == "__main__":
    main()
