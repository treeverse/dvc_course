"""Migrate the churn project's data and artifacts from DVC into lakeFS.

Prerequisites:
  1. lakeFS running locally:      docker compose up
  2. lakeFS SDK installed:        uv pip install -r requirements-lakefs.txt
  3. Data present locally:        dvc pull   (or dvc repro)

What it does:
  - Creates a lakeFS repository (local block store).
  - Uploads the dataset, features, model, and metrics onto the `main` branch,
    keeping the same paths they have in the DVC project.
  - Commits, so the whole snapshot is one atomic, named version in lakeFS.

Usage:
    python src/migrate_to_lakefs.py
    python src/migrate_to_lakefs.py --repo churn --branch main
"""

import argparse
import sys
from pathlib import Path

import lakefs
from lakefs.client import Client

# Matches docker-compose.yml. Do not use these values in production.
LAKEFS_HOST = "http://localhost:8000"
ACCESS_KEY = "AKIAIOSFOLKFSSAMPLES"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# The DVC-tracked paths to move over. These are the pipeline's real outputs.
PATHS = [
    "data/raw/churn.csv",
    "data/prepared/train.csv",
    "data/prepared/test.csv",
    "models/model.pkl",
    "models/preprocessor.pkl",
    "metrics/metrics.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="churn")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args()

    client = Client(host=LAKEFS_HOST, username=ACCESS_KEY, password=SECRET_KEY)

    # Create the repository if it does not exist yet. The storage namespace is
    # where lakeFS keeps the underlying objects; `local://` uses the block store
    # configured in docker-compose.yml.
    repo = lakefs.Repository(args.repo, client=client)
    try:
        repo.create(storage_namespace=f"local://{args.repo}", exist_ok=True)
    except lakefs.exceptions.LakeFSException as e:
        # Happens when the block store still holds objects for this namespace but
        # lakeFS has no repository record for it (e.g. the repo was deleted, or an
        # old container recreated its metadata). Point the learner at the fix
        # instead of a raw traceback.
        sys.exit(
            f"Could not create repository '{args.repo}': {e}\n"
            f"The storage namespace 'local://{args.repo}' is orphaned. Reset the "
            f"local block store with `docker compose down && rm -rf lakefs/data`, "
            f"then `docker compose up` and rerun, or pass a fresh --repo name."
        )
    print(f"Repository '{args.repo}' ready")

    branch = repo.branch(args.branch)

    uploaded = 0
    for path in PATHS:
        p = Path(path)
        if not p.exists():
            print(f"  skip (missing): {path} -- run `dvc pull` first")
            continue
        with open(p, "rb") as f:
            branch.object(path).upload(data=f.read(), mode="wb")
        print(f"  uploaded: {path}")
        uploaded += 1

    if uploaded == 0:
        print("Nothing to upload. Run `dvc pull` or `dvc repro` and try again.")
        return

    try:
        ref = branch.commit(
            message="Migrate churn dataset and model from DVC",
            metadata={"source": "dvc", "pipeline": "churn"},
        )
    except lakefs.exceptions.LakeFSException as e:
        # A re-run that uploads byte-identical files produces no diff, so lakeFS
        # rejects the commit with "no changes". That means the data is already
        # migrated, which is fine.
        if "no changes" in str(e):
            print(f"{args.repo}/{args.branch} already up to date, nothing to commit.")
            return
        raise
    print(f"Committed {uploaded} objects to {args.repo}/{args.branch} @ {ref.id[:12]}")


if __name__ == "__main__":
    main()
