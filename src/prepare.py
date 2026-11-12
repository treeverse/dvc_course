"""Stage 1 (prepare): clean the raw data and split it into train and test.

Reads:  data/raw/churn.csv
Writes: data/prepared/train.csv, data/prepared/test.csv
Params: base.*, prepare.*
"""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW = "data/raw/churn.csv"
OUT_DIR = Path("data/prepared")


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    base, prep = params["base"], params["prepare"]

    df = pd.read_csv(RAW)
    df = df.drop(columns=base["drop"])

    train, test = train_test_split(
        df,
        test_size=prep["test_size"],
        random_state=prep["random_state"],
        stratify=df[base["target"]],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"prepare: {len(train):,} train / {len(test):,} test rows")


if __name__ == "__main__":
    main()
