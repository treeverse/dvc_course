"""Stage 4 (evaluate): score the model on the held-out test set.

Reads:  models/model.pkl, data/features/test.pkl
Writes: metrics/metrics.json
"""

import json
import pickle
from pathlib import Path

from sklearn.metrics import f1_score, roc_auc_score

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
    print(f"evaluate: {metrics}")


if __name__ == "__main__":
    main()
