"""Train a baseline churn model.

Module 1 keeps this deliberately simple: read the raw CSV, train one scikit-learn
pipeline, and save it. No config files, no stages yet. The point of the module is
versioning the data and the model with DVC, not the modeling itself. Module 2
breaks this script into a real pipeline.

Usage:
    python src/train_baseline.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW = "data/raw/churn.csv"
MODEL_OUT = "models/model.pkl"
TARGET = "churn"
CATEGORICAL = ["country", "gender"]


def build_model() -> Pipeline:
    numeric = SimpleImputer(strategy="median")
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", numeric), ("scale", StandardScaler())]), None),
            ("cat", categorical, CATEGORICAL),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def main() -> None:
    df = pd.read_csv(RAW).drop(columns=["customer_id"])
    y = df.pop(TARGET)
    numeric_cols = [c for c in df.columns if c not in CATEGORICAL]

    model = build_model()
    # Wire the numeric column list into the ColumnTransformer now that we know it.
    model.named_steps["pre"].transformers[0] = (
        "num",
        model.named_steps["pre"].transformers[0][1],
        numeric_cols,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"F1:      {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
