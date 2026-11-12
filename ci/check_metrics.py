"""Fail CI if the reproduced model's metrics drop below a floor.

This script does not compare dvc.lock byte for byte. A retrained model is a
binary file, and its exact bytes (and therefore its hash in dvc.lock) differ
across operating systems, library builds, and hardware. A byte-exact check
would fail whenever CI runs on a different platform than the one where
dvc.lock was generated. Comparing metric values works the same on every
platform and still catches a real regression in the pipeline.

The floors are well below the committed model's numbers (f1 0.5891,
roc_auc 0.8581). Raise them as the model improves.
"""

import json
import sys

METRICS = "metrics/metrics.json"
FLOOR = {"f1": 0.50, "roc_auc": 0.80}


def main() -> None:
    metrics = json.load(open(METRICS))
    below = {k: metrics.get(k) for k, lo in FLOOR.items() if metrics.get(k, 0) < lo}
    if below:
        sys.exit(f"Metrics below floor {FLOOR}: {below}")
    print(f"Metrics OK: {metrics}")


if __name__ == "__main__":
    main()
