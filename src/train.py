"""Stage 3 (train): fit a classifier on the training features.

The model type and hyperparameters come from params.yaml, which is what lets
Module 3 run experiments by overriding parameters.

Reads:  data/features/train.pkl
Writes: models/model.pkl
Params: train.*
"""

import pickle
from pathlib import Path

import yaml

MODEL_OUT = Path("models/model.pkl")


def build_classifier(p: dict):
    max_depth = p["max_depth"] or None  # 0 -> None
    if p["model_type"] == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=p["n_estimators"],
            max_depth=max_depth or 6,
            learning_rate=p["learning_rate"],
            random_state=p["random_state"],
            eval_metric="logloss",
        )
    if p["model_type"] == "randomforest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=max_depth,
            random_state=p["random_state"],
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model_type: {p['model_type']}")


def main() -> None:
    p = yaml.safe_load(open("params.yaml"))["train"]
    with open("data/features/train.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)

    model = build_classifier(p)
    model.fit(X_train, y_train)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"train: fit {p['model_type']} on {X_train.shape[0]:,} rows")


if __name__ == "__main__":
    main()
