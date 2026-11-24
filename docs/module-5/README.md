# Module 5: From DVC to lakeFS

lakeFS is data version control at scale. It sits as a layer on top of an object store and gives you Git-like operations, branch, commit, merge, over everything in a bucket, so you can manage data the way you manage code. In this module you run lakeFS locally in a Docker container, create a repository, work on an isolated branch with `lakectl`, commit and merge back into main, and then import the churn data and model into a lakeFS repository. The three new files are fetched from `main` with `git checkout` instead of being copy-pasted.

| Start state | Check |
|-------------|-------|
| Continue from Module 4 | Tuned project with DVC pipeline |
| Or `git checkout module-5` | Then restore from your DVC remote |
| No remote | `dvc get https://github.com/treeverse/dvc_course .data/churn.csv -o data/raw/churn.csv && dvc repro` |
| Docker | Installed and running (top-level README, Requirements) |
| New files | `docker-compose.yml`, `requirements-lakefs.txt`, `src/migrate_to_lakefs.py` |

```bash
git checkout main -- docker-compose.yml requirements-lakefs.txt src/migrate_to_lakefs.py
```

| When lakeFS helps |
|-------------------|
| A data lake is one central, cheap, scalable place where all the organization's data lands. It is also a shared folder that everyone reads and writes, and DVC has no way to give that shared bucket production safety. |
| Several teams, pipelines, or repos read and write the same bucket, and each needs an isolated environment to experiment in without anyone else seeing half-finished data. |
| The data changes on its own schedule, without a matching Git commit each time. |
| Keeping a full DVC cache in every clone has gotten too expensive. |

| Tool | Use it when |
|------|-------------|
| DVC | The data version belongs together with one repo's code, params, metrics, model, and cache. |
| lakeFS | The shared bucket needs its own branches, commits, merges, and validation hooks. |
| Together | DVC can keep running the pipeline while lakeFS versions the object storage it reads from. |

| lakeFS term | Meaning |
|-------------|---------|
| Repository | The equivalent of a bucket, with committing, branching, and merging added on top |
| Storage namespace | Where the underlying objects live. Here `local://`, in production an S3, GCS, or Azure bucket |
| Branch | An isolated view of the repository, for example `main` or `experiment`. Creating one is metadata only, no data is copied |
| Uncommitted changes | Objects written to a branch and not yet committed |
| Commit | A fixed snapshot of the branch, with a message and optional metadata such as the code version |
| Merge | Promote one branch's changes into another, atomically |
| Hook | A check such as a schema or data quality validation that runs before a merge and can block it |

![lakeFS architecture](../images/lakefs-architecture.png)

![DVC cache and lakeFS branches](../images/zero-copy-branching.svg)

| DVC | lakeFS |
|-----|--------|
| Files/dirs tracked next to code | Everything in a bucket/namespace |
| Per file or directory | Per repository |
| Scales to one project | Scales to a whole data lake |
| Git branches plus `dvc checkout` | Object-storage branches |
| Shared remote plus Git | Multiple teams over shared buckets |

| Local server |
|--------------|
| A local install is the quickest way to get a look and feel for lakeFS, try the common operations, and get familiar with the UI. Compose keeps the local lakeFS data under `lakefs/data/`, which is gitignored. |

```bash
docker compose up
```

| Local login | Value |
|-------------|-------|
| URL | http://localhost:8000 |
| Access key | `AKIAIOSFOLKFSSAMPLES` |
| Secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |

The access key and secret key work like S3 credentials. Compose preconfigures this pair. On first visit lakeFS shows a User Configuration page that asks for a name and email; fill it in, then log in with the keys above. On a real installation you create an admin user on first start, and lakeFS shows the generated keys once, so keep them somewhere safe.

| Branch, commit, merge |
|-----------------------|
| `lakectl` is the lakeFS command line tool. Point it at the server, create a repository, branch off main, write an object on the branch, commit it, compare the two branches, and merge back into main. |
| `lakectl` is not a Python package: `brew install lakefs`, or download it from the [lakeFS releases](https://github.com/treeverse/lakeFS/releases). |

```bash
lakectl config      # server endpoint http://localhost:8000, and the keys above
lakectl repo list   # confirms lakectl can talk to the server
```

```bash
# Create a repository backed by the local block store
lakectl repo create lakefs://churn local://churn

# Branch off main. This is metadata only, nothing is copied.
lakectl branch create lakefs://churn/experiment --source lakefs://churn/main

# Write an object on the branch. It shows up as an uncommitted change
# on experiment, and main does not see it.
printf "trial feature set\n" | lakectl fs upload lakefs://churn/experiment/notes/feature-set.txt --source -

lakectl commit lakefs://churn/experiment -m "Try a new feature set"

# Compare the two branches, then merge experiment back into main
lakectl diff lakefs://churn/main lakefs://churn/experiment
lakectl merge lakefs://churn/experiment lakefs://churn/main
```

Open http://localhost:8000 and pick the `churn` repository. The Objects tab shows the files on whichever branch you select, the Uncommitted Changes tab shows what has been written but not committed, and the Commits, Branches, and Compare tabs cover the rest of the cycle you just ran from the command line.

| Migration |
|-----------|
| The script creates the `churn` repository if needed, uploads the files from your workspace into it, and commits them on main. It reads the workspace files rather than raw DVC cache objects, so restore or rebuild them first. |

```bash
uv pip install -r requirements-lakefs.txt
dvc pull                              # get the tracked files locally
python src/migrate_to_lakefs.py
```

| No working remote? |
|--------------------|
| Fresh checkout with `storage` still pointing at the private demo bucket: skip `dvc pull`, rebuild instead, then run the script. |

```bash
dvc get https://github.com/treeverse/dvc_course .data/churn.csv -o data/raw/churn.csv   # regenerate data/raw/churn.csv
dvc repro                             # rebuild prepared data, features, model, metrics
python src/migrate_to_lakefs.py
```

```text
Repository 'churn' ready
  uploaded: data/raw/churn.csv
  uploaded: data/prepared/train.csv
  uploaded: data/prepared/test.csv
  uploaded: models/model.pkl
  uploaded: models/preprocessor.pkl
  uploaded: metrics/metrics.json
Committed 6 objects to churn/main @ a1b2c3d4e5f6
```

From here on any S3-compatible tool can read the data through lakeFS. The only difference from plain S3 is the path, which is prefixed with the repository and the branch, so `s3a://churn/main/data/raw/churn.csv` reads the committed file on main and swapping `main` for `experiment` reads the branch instead.

| DVC | lakeFS |
|-----|--------|
| `.dvc` pointer / `dvc.lock` | a commit |
| DVC remote (S3) | the repository's storage namespace |
| `dvc checkout <rev>` | reading from a branch or commit |
| a Git branch of the project | a lakeFS branch of the data |

| Team pattern |
|--------------|
| Each job that writes data gets its own branch, so production stays isolated from work in progress. Hooks run data quality and schema checks on that branch, and the merge into main only goes through when they pass. If something is wrong you can revert or reset the branch instead. |

| Command | What it does |
|---------|------|
| [`docker compose up`](https://docs.docker.com/compose/) | Run lakeFS locally |
| [`lakectl repo create`](https://docs.lakefs.io/reference/cli.html#lakectl-repo-create) | Create a lakeFS repository |
| [`lakectl branch create`](https://docs.lakefs.io/reference/cli.html#lakectl-branch-create) | Branch data with zero copy |
| [`lakectl diff`](https://docs.lakefs.io/reference/cli.html#lakectl-diff) | Compare two branches or commits |
| [`lakectl commit`](https://docs.lakefs.io/reference/cli.html#lakectl-commit) / [`merge`](https://docs.lakefs.io/reference/cli.html#lakectl-merge) | Snapshot and merge data versions |
| `python src/migrate_to_lakefs.py` | Migrate the DVC project into lakeFS |

| Wrap up |
|---------|
| Commit the module's files, then stop the server. `lakefs/data/` keeps the repo state between `up` and `down`, so only delete it if you want to start over from nothing. |

```bash
git add docker-compose.yml requirements-lakefs.txt src/migrate_to_lakefs.py
git commit -m "Migrate the project to lakeFS"
```

```bash
docker compose down
# rm -rf lakefs/data                  # optional: wipe the local lakeFS state
```

Congratulations, you finished the course! 🎉
