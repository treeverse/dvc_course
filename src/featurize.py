"""Stage 2 (featurize): fit a preprocessor on train, transform train and test.

Impute + scale numeric columns, one-hot encode categoricals. The fitted
transformer is saved so the same features can be produced at inference time.

Reads:  data/prepared/train.csv, data/prepared/test.csv
Writes: data/features/train.pkl, data/features/test.pkl, models/preprocessor.pkl
Params: base.*
"""

import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PREPARED = Path("data/prepared")
FEATURES = Path("data/features")
PREPROCESSOR = Path("models/preprocessor.pkl")


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def main() -> None:
    base = yaml.safe_load(open("params.yaml"))["base"]
    target, categorical = base["target"], base["categorical"]

    train = pd.read_csv(PREPARED / "train.csv")
    test = pd.read_csv(PREPARED / "test.csv")

    y_train, y_test = train.pop(target), test.pop(target)
    numeric = [c for c in train.columns if c not in categorical]

    pre = build_preprocessor(numeric, categorical)
    X_train = pre.fit_transform(train)
    X_test = pre.transform(test)

    FEATURES.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES / "train.pkl", "wb") as f:
        pickle.dump((X_train, y_train.to_numpy()), f)
    with open(FEATURES / "test.pkl", "wb") as f:
        pickle.dump((X_test, y_test.to_numpy()), f)
    with open(PREPROCESSOR, "wb") as f:
        pickle.dump(pre, f)
    print(f"featurize: {X_train.shape[1]} features")


if __name__ == "__main__":
    main()
