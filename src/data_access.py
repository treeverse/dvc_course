"""Read DVC-tracked data programmatically with the DVC Python API.

With `dvc.api`, a training job, a notebook, or another project can load a
specific version of a dataset directly from the remote. No manual `dvc pull` is
needed. This is also how a data registry serves its consumers.

Usage:
    python src/data_access.py
    python src/data_access.py --rev main
"""

import argparse
from io import StringIO

import dvc.api
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rev",
        default=None,
        help="Git revision (branch, tag, or commit) to read the data at.",
    )
    parser.add_argument("--path", default="data/raw/churn.csv")
    args = parser.parse_args()

    # Get the remote URL for this version of the file. Nothing is downloaded.
    url = dvc.api.get_url(args.path, rev=args.rev)
    print(f"Remote URL: {url}")

    # Read the file contents at the requested revision.
    content = dvc.api.read(args.path, rev=args.rev, mode="r")
    df = pd.read_csv(StringIO(content))
    print(f"Loaded {len(df):,} rows at rev={args.rev or 'workspace'}")
    print(df["churn"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
