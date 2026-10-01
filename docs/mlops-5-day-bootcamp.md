# MLOps in 5 Days: From Beginner to Production-Ready Engineer

> A hands-on, step-by-step curriculum. You will build **one real project** from a raw
> training script on Day 1 into a versioned, tested, containerized, deployed, and
> monitored ML service by the end of Day 5. Every concept is introduced right before
> you use it, and every day ends with a working checkpoint.

---

## Table of Contents

- [How to Use This Guide](#how-to-use-this-guide)
- [What You Will Have Built by Day 5](#what-you-will-have-built-by-day-5)
- [Prerequisites and Setup (Day 0, ~2 hours)](#prerequisites-and-setup-day-0-2-hours)
- [Day 1: Foundations, Reproducibility, and Your First Clean Training Pipeline](#day-1-foundations-reproducibility-and-your-first-clean-training-pipeline)
- [Day 2: Experiment Tracking and the Model Registry](#day-2-experiment-tracking-and-the-model-registry)
- [Day 3: Pipelines, Data Validation, Testing, and Continuous Integration](#day-3-pipelines-data-validation-testing-and-continuous-integration)
- [Day 4: Serving Models: APIs, Docker, and Deployment](#day-4-serving-models-apis-docker-and-deployment)
- [Day 5: Monitoring, Drift, Retraining, and Production Operations](#day-5-monitoring-drift-retraining-and-production-operations)
- [Capstone Checklist](#capstone-checklist)
- [What to Learn Next](#what-to-learn-next)
- [Glossary](#glossary)
- [Appendix A: Full File Reference](#appendix-a-full-file-reference)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)

---

## How to Use This Guide

**Honest expectation setting.** Five days will not make you a senior MLOps engineer. It
*will* give you the complete mental model and a working, end-to-end project that covers
every stage of the ML lifecycle. After this week you will be able to join a team, take a
notebook from a data scientist, and turn it into something that runs reliably in
production. Depth in any single tool (Kubernetes, Spark, a specific cloud) comes with
practice after this week.

**Each day is roughly 8 hours**, split into:

| Block | Time | What happens |
|---|---|---|
| Concepts | ~2 h | Read the explanations. Do not skip them. The "why" is what separates an engineer from a tool user. |
| Build | ~4 h | Type the code yourself. Do not copy-paste whole files. Typing forces you to read every line. |
| Exercises | ~1.5 h | Extend what you built. These are where real learning happens. |
| Checkpoint | ~30 min | Answer the questions out loud or in writing. If you cannot, re-read the relevant section. |

**Rules for the week:**

1. Commit to git at the end of every section. Small commits with clear messages.
2. When something breaks, read the full error message before searching. Most errors tell you exactly what is wrong.
3. Keep a `NOTES.md` in the project. Write down every "aha" moment and every question. You will reread it.
4. If you fall behind, finish the Build block and skip exercises. Do not skip Concepts.

---

## What You Will Have Built by Day 5

A project called `housing-mlops` that predicts California house prices and includes:

```
housing-mlops/
├── .github/workflows/        # CI/CD: lint, test, train smoke test, build & push Docker image
├── data/                     # Raw and processed data, versioned with DVC (not git)
├── docs/                     # Model card, runbook
├── models/                   # Trained model artifacts (DVC-tracked)
├── metrics/                  # Evaluation metrics as JSON (git-tracked, tiny)
├── monitoring/               # Drift reports and reference data
├── notebooks/                # Exploration only; never imported by production code
├── src/housing/              # The installable Python package
│   ├── data.py               #   ingest raw data
│   ├── validate.py           #   schema validation with pandera
│   ├── features.py           #   train/test split + feature engineering
│   ├── train.py              #   training with MLflow tracking
│   ├── evaluate.py           #   evaluation + metrics file
│   ├── registry.py           #   promote model in MLflow registry
│   ├── drift.py              #   Evidently drift report
│   └── api/app.py            #   FastAPI prediction service with Prometheus metrics
├── tests/                    # pytest: unit, data, model-quality, API tests
├── Dockerfile
├── docker-compose.yml        # API + MLflow + Prometheus + Grafana locally
├── dvc.yaml                  # The pipeline definition
├── params.yaml               # All hyperparameters and config
├── pyproject.toml            # Dependencies and tooling config
└── Makefile                  # One-word commands for everything
```

You will be able to run `make train` to reproduce the model, `make serve` to run the API,
`make test` to verify everything, and push to GitHub to trigger automated checks and a
container build.

---

## Prerequisites and Setup (Day 0, ~2 hours)

### What you should already know

- **Python basics**: functions, classes, imports, virtual environments, `pip`.
- **Basic ML**: what a train/test split is, what overfitting means, what a metric like
  accuracy or RMSE measures. You should have trained at least one scikit-learn model.
- **Terminal basics**: `cd`, `ls`, running commands, editing files.
- **Git basics**: `clone`, `add`, `commit`, `push`. If not, do a 30-minute git tutorial first.

You do **not** need to know Docker, cloud platforms, Kubernetes, or any MLOps tool.

### Install these tools

Run each command and confirm the version prints. All are free.

| Tool | Why | Install | Verify |
|---|---|---|---|
| Python 3.12 | Runtime | https://www.python.org/downloads/ or `brew install python@3.12` | `python3 --version` |
| `uv` | Fast dependency manager and virtualenv tool | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (Linux/macOS) or `pip install uv` | `uv --version` |
| git | Version control | https://git-scm.com | `git --version` |
| Docker Desktop (or Docker Engine on Linux) | Containers | https://docs.docker.com/get-docker/ | `docker run hello-world` |
| VS Code (or any editor) | Editing | https://code.visualstudio.com | open it |
| GitHub account | CI/CD and hosting | https://github.com | log in |
| `make` | Task runner | Preinstalled on macOS/Linux; Windows: use WSL2 | `make --version` |

**Windows users:** install WSL2 (Windows Subsystem for Linux) with Ubuntu and do
everything inside it. Native Windows works for most of this, but Docker and `make`
are far smoother in WSL2.

### Configure git once

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

### Create the GitHub repository

1. On GitHub, click **New repository**, name it `housing-mlops`, keep it empty (no README).
2. Locally:

```bash
mkdir housing-mlops && cd housing-mlops
git init
git remote add origin git@github.com:YOUR_USERNAME/housing-mlops.git
```

If you have not set up SSH keys for GitHub, follow
https://docs.github.com/en/authentication/connecting-to-github-with-ssh or use the
HTTPS URL instead.

You are ready.

---

## Day 1: Foundations, Reproducibility, and Your First Clean Training Pipeline

### Learning objectives

By the end of today you can:

- Explain what MLOps is and why ML systems fail in production.
- Describe the ML lifecycle and the three MLOps maturity levels.
- Set up a professional Python project structure for ML.
- Write a training script that is configurable, reproducible, and produces artifacts.
- Version large data files with DVC alongside code in git.

### 1.1 Concepts: What MLOps actually is

**The one-sentence definition:** MLOps is the set of practices that let you reliably
take a model from an idea to production and keep it working there.

**Why it exists.** Traditional software is deterministic: the same code produces the
same behavior. An ML system has *three* things that change independently:

1. **Code** (your training and serving logic)
2. **Data** (what the model learned from)
3. **Model** (the trained artifact, which depends on both of the above plus randomness)

Change any one and the behavior of the system changes. DevOps tooling only tracks code.
MLOps adds tracking, testing, and automation for data and models too.

**The ML lifecycle** (you will touch every stage this week):

```
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  1. Data    │───▶│ 2. Training │───▶│ 3. Deploy   │───▶│ 4. Monitor  │
  │  collect    │    │  experiment │    │  package    │    │  drift      │
  │  validate   │    │  evaluate   │    │  serve      │    │  alerts     │
  │  version    │    │  register   │    │  release    │    │  retrain    │
  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
         ▲                                                        │
         └────────────────────────────────────────────────────────┘
                              (the loop never ends)
```

**How ML projects fail in production** (memorize these, they are interview questions
and daily reality):

| Failure | What it looks like | What prevents it |
|---|---|---|
| **Cannot reproduce** | "The model on my laptop got 0.92 but the retrained one gets 0.85 and nobody knows why." | Versioned data, pinned dependencies, seeded randomness, tracked experiments |
| **Training-serving skew** | Features computed one way in the notebook, a different way in the API. | Shared feature code used by both training and serving |
| **Data drift** | The world changes, inputs stop looking like training data, accuracy silently decays. | Monitoring input distributions, retraining triggers |
| **Silent failure** | Model returns predictions but they are garbage. No error is thrown. | Prediction logging, output distribution monitoring, ground-truth feedback |
| **Hidden technical debt** | Glue code, pipeline jungles, undeclared consumers. | Clean package structure, pipelines as code, contracts between stages |
| **No rollback** | New model is worse; nobody can restore the old one quickly. | Model registry with versions and aliases |

**MLOps maturity levels** (from Google's widely-used framework):

- **Level 0: Manual.** A data scientist trains in a notebook, hands over a pickle file,
  an engineer wraps it in an API by hand. Retraining is a rare, manual, scary event.
  Most companies start here.
- **Level 1: ML pipeline automation.** Training is an automated pipeline. You can
  retrain by running one command or on a schedule. Experiments are tracked. Models are
  registered. This is what you will reach by Day 3.
- **Level 2: CI/CD pipeline automation.** Changes to the pipeline code are
  automatically tested and deployed. New models are automatically validated and
  rolled out. You will reach the core of this by Day 5.

### 1.2 Concepts: Reproducibility, the foundation of everything

A result is reproducible if someone else (or you in six months) can get the same output
from the same inputs. For ML this requires controlling:

1. **Code version**: git commit hash.
2. **Data version**: a hash of the exact dataset. Git is bad at large files, so we use DVC.
3. **Environment**: exact library versions. A lock file (`uv.lock`) pins everything.
4. **Configuration**: every hyperparameter in a file, never hard-coded.
5. **Randomness**: every random operation gets a fixed seed.

If any one of these is missing, "it worked on my machine" is your future.

### 1.3 Build: Project skeleton

Inside `housing-mlops/`, create this structure. We will fill files in gradually.

```bash
mkdir -p src/housing tests data/raw data/processed models metrics notebooks docs
touch src/housing/__init__.py tests/__init__.py
```

Create `pyproject.toml`. This single file declares the package, dependencies, and tool
settings. Read every line and its comment.

```toml
[project]
name = "housing"
version = "0.1.0"
description = "California housing price prediction, built as an MLOps learning project"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "numpy>=2.0",
    "scikit-learn>=1.5",
    "pyyaml>=6.0",
    "joblib>=1.4",
    "pyarrow>=17.0",      # parquet support for pandas
]

[project.optional-dependencies]
# Day 2+ dependencies are added later; keeping them separate keeps the
# serving image small on Day 4.
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
    "pre-commit>=3.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/housing"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]   # errors, pyflakes, import sort, bugbear, pyupgrade

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

**Why the `src/` layout?** With code inside `src/housing/`, Python can only import it
once the package is installed. That forces you to install your own code properly, which
is exactly what will happen inside Docker and CI later. Projects with code at the root
"work on my laptop" and then fail in CI because of implicit path tricks.

Create the environment and install:

```bash
uv sync --extra dev
```

This creates `.venv/`, resolves all dependencies, writes `uv.lock` (commit this file!),
and installs `housing` in editable mode. From now on, run Python through `uv run`:

```bash
uv run python -c "import housing; print('ok')"
```

> **If you prefer plain pip:** `python -m venv .venv && source .venv/bin/activate &&
> pip install -e ".[dev]"` and drop the `uv run` prefix from every command. You lose
> the lock file, which matters on Day 4. `uv` is strongly recommended.

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/

# Data and models are tracked by DVC, not git (set up later today)
/data/raw/
/data/processed/
/models/*.joblib

# MLflow local artifacts (Day 2)
mlruns/
mlartifacts/
mlflow.db

# Monitoring reports (Day 5)
/monitoring/reports/

# Editors and OS
.vscode/
.idea/
.DS_Store

# Secrets. NEVER commit these.
.env
```

First commit:

```bash
git add .
git commit -m "Project skeleton with pyproject and gitignore"
```

### 1.4 Build: Configuration file

Every number that could change goes here. We use YAML because DVC (today) and most
orchestrators read it natively.

`params.yaml`:

```yaml
data:
  raw_path: data/raw/housing.csv
  processed_dir: data/processed

prepare:
  test_size: 0.2
  random_state: 42
  target: MedHouseVal

train:
  model_path: models/model.joblib
  random_state: 42
  n_estimators: 200
  max_depth: 12
  min_samples_leaf: 2

evaluate:
  metrics_path: metrics/metrics.json
```

A tiny helper to load it. `src/housing/config.py`:

```python
"""Load project configuration from params.yaml.

Keeping all configuration in one file (instead of scattered constants) means
every experiment is fully described by (git commit, params.yaml, data version).
"""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Turn a path from params.yaml into an absolute path from the project root."""
    return PROJECT_ROOT / relative
```

### 1.5 Build: Data ingestion

We use the California Housing dataset: 20,640 rows, 8 numeric features, target is
median house value in units of $100,000. It is small, clean, built into scikit-learn,
and realistic enough to show drift later.

`src/housing/data.py`:

```python
"""Stage 1: ingest raw data.

In a real project this would pull from a warehouse, an API, or a bucket.
The important properties are the same: idempotent, writes to a known path,
and logs what it did.
"""

import logging

from sklearn.datasets import fetch_california_housing

from housing.config import load_params, resolve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def ingest() -> None:
    params = load_params()
    out_path = resolve(params["data"]["raw_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Fetching California housing dataset")
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame  # features + target column 'MedHouseVal'

    df.to_csv(out_path, index=False)
    log.info("Wrote %d rows x %d cols to %s", df.shape[0], df.shape[1], out_path)


if __name__ == "__main__":
    ingest()
```

Run it:

```bash
uv run python -m housing.data
head -3 data/raw/housing.csv
```

You should see columns: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
Latitude, Longitude, MedHouseVal`.

### 1.6 Build: Feature preparation

`src/housing/features.py`:

```python
"""Stage 2: split and prepare features.

RULE: any transformation applied here must ALSO be applied at serving time.
To guarantee that, keep feature logic in a function (`build_features`) that
both training and the API import. This is how you prevent training-serving skew.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from housing.config import load_params, resolve

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure function: raw columns in, model-ready columns out.

    Deliberately simple today. On Day 3 you will add derived features here and
    see that the API picks them up automatically because it calls this same function.
    """
    out = df[FEATURE_COLUMNS].copy()
    # Example of a guard that belongs in shared code, not in a notebook:
    out["AveOccup"] = out["AveOccup"].clip(upper=50)  # a handful of absurd outliers
    return out


def prepare() -> None:
    params = load_params()
    raw_path = resolve(params["data"]["raw_path"])
    processed_dir = resolve(params["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    target = params["prepare"]["target"]
    df = pd.read_csv(raw_path)

    train_df, test_df = train_test_split(
        df,
        test_size=params["prepare"]["test_size"],
        random_state=params["prepare"]["random_state"],
    )

    for name, split in [("train", train_df), ("test", test_df)]:
        features = build_features(split)
        features[target] = split[target].values
        path = processed_dir / f"{name}.parquet"
        features.to_parquet(path, index=False)
        log.info("Wrote %s: %d rows", path, len(features))


if __name__ == "__main__":
    prepare()
```

Run it:

```bash
uv run python -m housing.features
ls -la data/processed/
```

### 1.7 Build: Training and evaluation

`src/housing/train.py` (Day 1 version; Day 2 adds MLflow):

```python
"""Stage 3: train the model.

Reads processed training data and params.yaml, writes a model artifact.
Nothing else. A training script that also evaluates, plots, and uploads is
a script you cannot test or reuse.
"""

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from housing.config import load_params, resolve

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def train() -> None:
    params = load_params()
    cfg = params["train"]
    target = params["prepare"]["target"]

    train_df = pd.read_parquet(resolve(params["data"]["processed_dir"]) / "train.parquet")
    X = train_df.drop(columns=[target])
    y = train_df[target]

    model = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )
    log.info("Training on %d rows with params %s", len(X), cfg)
    model.fit(X, y)

    model_path = resolve(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    log.info("Saved model to %s", model_path)


if __name__ == "__main__":
    train()
```

`src/housing/evaluate.py`:

```python
"""Stage 4: evaluate the trained model on held-out data and write metrics.

Metrics are written as a small JSON file that IS committed to git, so every
commit carries its own scorecard and `git log` becomes a performance history.
"""

import json
import logging

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from housing.config import load_params, resolve

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate() -> dict[str, float]:
    params = load_params()
    target = params["prepare"]["target"]

    test_df = pd.read_parquet(resolve(params["data"]["processed_dir"]) / "test.parquet")
    X = test_df.drop(columns=[target])
    y = test_df[target]

    model = joblib.load(resolve(params["train"]["model_path"]))
    metrics = compute_metrics(y, model.predict(X))

    out = resolve(params["evaluate"]["metrics_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    evaluate()
```

Run the whole thing:

```bash
uv run python -m housing.train
uv run python -m housing.evaluate
cat metrics/metrics.json
```

Expect roughly `rmse ≈ 0.50`, `r2 ≈ 0.81`. (Target units are $100k, so RMSE 0.50 means
about $50k average error.)

**Prove reproducibility.** Delete the model and metrics, rerun, and diff:

```bash
cp metrics/metrics.json /tmp/m1.json
rm models/model.joblib metrics/metrics.json
uv run python -m housing.train && uv run python -m housing.evaluate
diff /tmp/m1.json metrics/metrics.json && echo "REPRODUCIBLE"
```

If it prints `REPRODUCIBLE`, your seeds are working. Now change `random_state` in
`params.yaml` to 7 and rerun: metrics change. Change it back. This is the entire idea
of configuration-driven training.

### 1.8 Build: A Makefile so nobody has to remember commands

`Makefile` (indentation must be a real TAB character, not spaces):

```makefile
.PHONY: install data features train evaluate pipeline test lint clean

install:
	uv sync --extra dev

data:
	uv run python -m housing.data

features:
	uv run python -m housing.features

train:
	uv run python -m housing.train

evaluate:
	uv run python -m housing.evaluate

pipeline: data features train evaluate

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

clean:
	rm -rf data/processed/* models/*.joblib metrics/*.json
```

Test: `make clean && make pipeline`.

Commit:

```bash
make format
git add .
git commit -m "Configurable, reproducible train/evaluate pipeline"
```

### 1.9 Concepts: Data versioning with DVC

Git stores every version of every file in full. A 2 GB dataset with 10 versions makes a
20 GB repo. DVC (Data Version Control) solves this:

- You run `dvc add data/raw/housing.csv`. DVC moves the file into a content-addressed
  cache (`.dvc/cache/`) and writes a tiny `data/raw/housing.csv.dvc` text file
  containing the file's MD5 hash and size.
- You commit the `.dvc` file to git. Git now tracks *which version* of the data each
  commit used, without storing the data.
- `dvc push` uploads the actual files to a remote (S3, GCS, Azure, SSH, or a local
  folder). `dvc pull` downloads whichever version your current git commit points to.

The result: `git checkout <old-commit> && dvc pull` restores the exact code **and** the
exact data from that moment.

### 1.10 Build: Set up DVC

Add DVC to your dev dependencies and install:

```bash
uv add --optional dev "dvc>=3.50"
uv sync --extra dev
```

Initialize:

```bash
uv run dvc init
git commit -m "Initialize DVC"
```

Track the raw data and model:

```bash
uv run dvc add data/raw/housing.csv
uv run dvc add models/model.joblib
```

Look at what it generated:

```bash
cat data/raw/housing.csv.dvc
```

Something like:

```yaml
outs:
- md5: 1a2b3c...
  size: 1915795
  hash: md5
  path: housing.csv
```

DVC also edited `.gitignore` files in those folders. Remove the broad entries you added
earlier for `/data/raw/` and `/models/*.joblib` from the root `.gitignore` to avoid
double-ignoring (DVC manages these now), then commit:

```bash
git add .
git commit -m "Track raw data and model with DVC"
```

Set up a remote. For learning, a local folder outside the repo is enough. (Swap for
`s3://bucket/path` or `gs://bucket/path` in a real project; the commands are identical.)

```bash
mkdir -p ~/dvc-remote-housing
uv run dvc remote add -d localremote ~/dvc-remote-housing
git commit -am "Configure DVC remote"
uv run dvc push
```

**Prove it works.** Simulate a fresh clone:

```bash
rm data/raw/housing.csv models/model.joblib
uv run dvc pull
ls data/raw models
```

Both files are back, byte-identical.

### 1.11 Exercises

1. **Add a second model type.** Add a `train.model_type` key to `params.yaml` with
   values `random_forest` or `gradient_boosting`, and make `train.py` select
   `GradientBoostingRegressor` when asked. Run both, compare `metrics.json`.
2. **Break reproducibility on purpose.** Remove `random_state` from the
   `train_test_split` call, run the pipeline twice, and observe the metrics differ.
   Put it back. Write in `NOTES.md` why the split matters as much as the model seed.
3. **Data versioning drill.** Modify `data.py` to drop rows where `Population > 10000`,
   rerun `make data`, run `dvc status`, `dvc add data/raw/housing.csv`, commit, push.
   Then `git checkout HEAD~1 -- data/raw/housing.csv.dvc && dvc checkout` and confirm
   the old data returns. Then `git checkout main -- data/raw/housing.csv.dvc && dvc checkout`.
4. **Logging.** Add the git commit hash to the training log output. Hint:
   `subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])`.

### 1.12 Checkpoint questions

1. Name the three things that can change independently in an ML system and why that
   makes it harder than normal software.
2. What five things must be fixed to make a training run reproducible?
3. Why is feature logic in a shared function instead of inline in the training script?
4. What does a `.dvc` file contain, and why is it safe to commit to git?
5. What is the difference between `dvc push` and `git push`?
6. What is MLOps maturity Level 1 and what does your project currently lack to be there?

---

## Day 2: Experiment Tracking and the Model Registry

### Learning objectives

- Explain why experiment tracking is non-negotiable in a team.
- Log parameters, metrics, artifacts, and models to MLflow.
- Compare runs in the MLflow UI and pick a winner objectively.
- Register a model, version it, and promote it with aliases.
- Load a model from the registry by alias instead of by file path.

### 2.1 Concepts: The experiment tracking problem

Yesterday you ran training four or five times with different parameters. Where are the
results? In your terminal scrollback, which is gone. In `metrics.json`, which gets
overwritten. This is how data scientists end up with `model_final_v2_REAL.pkl`.

**Experiment tracking** records, for every training run:

- **Parameters**: every hyperparameter and config value
- **Metrics**: every evaluation number, optionally over time (per epoch)
- **Artifacts**: the model file, plots, the exact `params.yaml`, sample predictions
- **Metadata**: who ran it, when, git commit, data version, hostname

You get a queryable database of every experiment ever run. Six months later, when
someone asks "why did we pick max_depth=12?", you have the answer.

**MLflow** is the most widely used open-source tool for this. It has four components;
you will use the first two heavily:

1. **Tracking**: log and query runs
2. **Model Registry**: version models, attach aliases like `champion`, record lineage
3. **Models**: a standard packaging format (`pyfunc`) that any MLflow model can be loaded with
4. **Projects**: packaging for reproducible runs (less used now; skip)

Alternatives you will hear about: Weights & Biases (polished SaaS, popular in deep
learning), Neptune, Comet, and the built-in trackers in SageMaker / Vertex AI. The
concepts transfer directly.

### 2.2 Build: Run an MLflow tracking server

```bash
uv add "mlflow>=3.0"
uv sync --extra dev
```

Start a local server in a second terminal. It stores metadata in a SQLite file and
artifacts in a folder:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Empty for now.

Add to the Makefile:

```makefile
mlflow-ui:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

**Why a server and not just `mlflow.log_*` to a local folder?** Because a server is
what a team uses. Everyone logs to the same place. In production this server runs on a
VM or managed service (Databricks, Azure ML, SageMaker all host MLflow-compatible
servers) with a Postgres backend and S3/GCS artifact store. Your local setup is the same
architecture in miniature.

### 2.3 Build: Instrument training with MLflow

Add tracking config to `params.yaml`:

```yaml
mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: housing-price
  registered_model_name: housing-price-regressor
```

Rewrite `src/housing/train.py`. Read the comments carefully; each one is a concept.

```python
"""Stage 3: train the model, tracked with MLflow.

Every run records: params, metrics on a validation split, the model (with a
signature), the config file, and git metadata. Nothing about a run is lost.
"""

import logging
import os
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split

from housing.config import PARAMS_PATH, load_params, resolve
from housing.evaluate import compute_metrics

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # not a git repo, e.g. inside some CI containers
        return "unknown"


def build_model(cfg: dict):
    model_type = cfg.get("model_type", "random_forest")
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg.get("learning_rate", 0.1),
            random_state=cfg["random_state"],
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train() -> str:
    """Train, log to MLflow, save locally. Returns the MLflow run id."""
    params = load_params()
    cfg = params["train"]
    target = params["prepare"]["target"]

    # Environment variable overrides the file so CI/Docker can point elsewhere
    # without editing params.yaml.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    df = pd.read_parquet(resolve(params["data"]["processed_dir"]) / "train.parquet")
    X = df.drop(columns=[target])
    y = df[target]

    # A validation split carved from TRAIN. The test set from features.py stays
    # untouched until final evaluation, so it remains an honest estimate.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=cfg["random_state"]
    )

    with mlflow.start_run() as run:
        # 1. Parameters: everything that defines this run
        mlflow.log_params(cfg)
        mlflow.log_param("train_rows", len(X_tr))
        mlflow.set_tags({"git_commit": git_commit(), "stage": "dev"})

        # 2. Train
        model = build_model(cfg)
        model.fit(X_tr, y_tr)

        # 3. Metrics on validation
        val_metrics = compute_metrics(y_val, model.predict(X_val))
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        log.info("Validation metrics: %s", val_metrics)

        # 4. Model with a signature. The signature records expected input columns
        #    and dtypes; MLflow will refuse malformed input at serving time.
        signature = infer_signature(X_tr, model.predict(X_tr.head()))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=X_tr.head(3),
        )

        # 5. Artifacts: the exact config used
        mlflow.log_artifact(str(PARAMS_PATH))

        # Local copy for the DVC pipeline and the API (Day 4)
        model_path = resolve(cfg["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        log.info("Run %s complete", run.info.run_id)
        return run.info.run_id


if __name__ == "__main__":
    train()
```

Note we import `compute_metrics` from `evaluate.py`. Reusing one metrics function for
validation and test guarantees they are computed identically.

Add `model_type: random_forest` under `train:` in `params.yaml`.

Run it with the server still up:

```bash
make train
```

Refresh the MLflow UI. Click the experiment, then the run. Explore every tab:
Parameters, Metrics, Artifacts (expand `model/`: notice `MLmodel`, `conda.yaml`,
`requirements.txt`, `input_example.json`). MLflow captured the environment needed to
load this model.

### 2.4 Build: Run a small experiment sweep

Nobody picks hyperparameters by hand in production. Write a sweep script that trains
several configurations and logs each as a run. `src/housing/sweep.py`:

```python
"""Run a small grid of configurations, each logged as its own MLflow run.

For real projects use Optuna or Ray Tune; the logging pattern is identical.
"""

import itertools
import logging

from housing import train as train_module
from housing.config import load_params

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GRID = {
    "model_type": ["random_forest", "gradient_boosting"],
    "n_estimators": [100, 300],
    "max_depth": [6, 12],
}


def sweep() -> None:
    base = load_params()
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        overrides = dict(zip(keys, values, strict=True))
        log.info("=== Running %s", overrides)
        # Monkeypatch config for this run. Simple and explicit for learning;
        # a real sweep tool would pass a config object instead.
        params = {**base, "train": {**base["train"], **overrides}}
        train_module.load_params = lambda p=params: p  # noqa: E731
        train_module.train()


if __name__ == "__main__":
    sweep()
```

```bash
uv run python -m housing.sweep
```

Eight runs. In the UI, select all runs, click **Compare**, and look at the parallel
coordinates plot and the scatter of `val_rmse` vs `n_estimators`. Sort the table by
`val_rmse`. This is how you pick a configuration: with evidence, not vibes.

Copy the best configuration's values back into `params.yaml`. That file is now the
source of truth for "the model we ship."

### 2.5 Concepts: The model registry

Tracking answers "what did we try?" The **registry** answers "what is in production
right now, what was before it, and how do I switch?"

A registered model is a named entity (`housing-price-regressor`) with numbered
**versions** (1, 2, 3...). Each version points at a specific run's model artifact, so
lineage is automatic: version 3 → run abc123 → params, metrics, git commit, data.

**Aliases** are mutable pointers to versions:

- `champion` → the version serving production traffic
- `challenger` → a candidate being evaluated
- `previous` → what to roll back to

Promoting a model means moving the `champion` alias. Rolling back means moving it back.
No file copying, no redeploy of code; the serving layer loads `models:/housing-price-regressor@champion`
and gets whatever that points to.

(Older MLflow used "stages" like Staging/Production. These are deprecated in favor of
aliases. You will still see them in tutorials.)

### 2.6 Build: Register and promote

`src/housing/registry.py`:

```python
"""Register a run's model and manage aliases.

Usage:
  python -m housing.registry register <run_id>
  python -m housing.registry promote <version> [--alias champion]
  python -m housing.registry show
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient

from housing.config import load_params


def _client() -> tuple[MlflowClient, str]:
    params = load_params()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]))
    return MlflowClient(), params["mlflow"]["registered_model_name"]


def register(run_id: str) -> int:
    _, name = _client()
    mv = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=name)
    print(f"Registered {name} version {mv.version}")
    return int(mv.version)


def promote(version: int, alias: str = "champion") -> None:
    client, name = _client()
    # Keep a pointer to the outgoing champion so rollback is one command.
    if alias == "champion":
        try:
            current = client.get_model_version_by_alias(name, "champion")
            client.set_registered_model_alias(name, "previous", current.version)
        except mlflow.exceptions.MlflowException:
            pass  # no champion yet
    client.set_registered_model_alias(name, alias, version)
    print(f"{name} alias '{alias}' -> version {version}")


def show() -> None:
    client, name = _client()
    for mv in client.search_model_versions(f"name='{name}'"):
        print(f"v{mv.version}  run={mv.run_id}  aliases={list(mv.aliases)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("run_id")
    pr = sub.add_parser("promote")
    pr.add_argument("version", type=int)
    pr.add_argument("--alias", default="champion")
    sub.add_parser("show")
    args = p.parse_args()

    if args.cmd == "register":
        register(args.run_id)
    elif args.cmd == "promote":
        promote(args.version, args.alias)
    else:
        show()
```

Get the best run's id from the UI (click the run; it is in the header), then:

```bash
uv run python -m housing.registry register <RUN_ID>
uv run python -m housing.registry promote 1
uv run python -m housing.registry show
```

In the UI, open **Models** in the top bar. You will see the registered model, version 1,
alias `champion`.

### 2.7 Build: Load a model by alias

This is the pattern your API will use on Day 4. Try it in a Python shell:

```bash
uv run python
```

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/housing-price-regressor@champion")
sample = pd.DataFrame([{
    "MedInc": 8.3, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
    "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23,
}])
print(model.predict(sample))   # ~ [4.3]  (i.e. ~$430k)
```

Now try passing a DataFrame with a column missing. MLflow raises a clear error because
of the signature. That is input validation you got for free.

Commit everything:

```bash
make format && git add . && git commit -m "MLflow tracking, sweep, and model registry"
```

### 2.8 Exercises

1. **Log a plot.** In `train.py`, after fitting, create a matplotlib scatter of
   predicted vs actual on the validation set, save to a temp PNG, and
   `mlflow.log_artifact` it. (`uv add matplotlib`.) Find it in the UI.
2. **Log feature importances** as a JSON artifact and as individual metrics
   (`importance_MedInc`, etc.). Which feature matters most?
3. **Autolog.** Replace manual param/metric logging with `mlflow.sklearn.autolog()`
   and compare what it captures vs what you logged by hand. Decide which you prefer
   and write why in `NOTES.md`. (Many teams use autolog plus a few manual extras.)
4. **Rollback drill.** Register a second, deliberately worse run (e.g. `max_depth: 2`)
   as version 2, promote it to `champion`, load by alias and confirm predictions are
   worse, then promote version 1 again. Confirm `previous` alias moved correctly.
5. **Search runs programmatically.** Use `mlflow.search_runs(experiment_names=[...],
   order_by=["metrics.val_rmse ASC"], max_results=1)` to find the best run without the
   UI. This is what automated promotion will use on Day 5.

### 2.9 Checkpoint questions

1. What four things does an MLflow run record, and give one example of each for your project.
2. Why do we carve a validation set from training data instead of tuning on the test set?
3. What is a model signature and what problem does it prevent?
4. Explain the difference between an experiment, a run, a registered model, and a model version.
5. What does an alias do, and why is it better than copying `best_model.pkl` to a server?
6. Where does the MLflow server store (a) metadata and (b) artifacts in your setup, and what would change for a team deployment?

---

## Day 3: Pipelines, Data Validation, Testing, and Continuous Integration

### Learning objectives

- Define the training workflow as a DAG that only reruns what changed.
- Validate data against a schema and fail fast on bad data.
- Write unit tests for feature code, data contracts, and model quality.
- Run lint and tests automatically on every push with GitHub Actions.
- Reach MLOps maturity Level 1.

### 3.1 Concepts: Why pipelines instead of scripts

Yesterday `make pipeline` ran four scripts in order. Problems:

- It reruns everything even if only `train.py` changed (slow with real data).
- Nothing records which data version produced which model.
- There is no graph: nobody can see that `evaluate` depends on `train` and `features`.

A **pipeline** is a DAG (directed acyclic graph) of stages, each with declared inputs
(dependencies), outputs, and parameters. A pipeline runner:

1. Hashes every dependency.
2. Skips stages whose inputs have not changed.
3. Records the hashes in a lock file so the exact run is reproducible.

Tools: **DVC pipelines** (lightweight, file-based, great for single-machine training;
what we use), **Prefect / Dagster / Airflow** (general orchestrators, scheduling,
retries, distributed), **Kubeflow Pipelines / Vertex Pipelines / SageMaker Pipelines**
(Kubernetes- and cloud-native, for scale). The DAG concept is identical in all of them.

### 3.2 Build: Define the pipeline in `dvc.yaml`

First, stop tracking `models/model.joblib` as a standalone DVC file. It will become a
pipeline output:

```bash
uv run dvc remove models/model.joblib.dvc
git add . && git commit -m "Model will be a pipeline output"
```

Create `dvc.yaml`:

```yaml
stages:
  ingest:
    cmd: uv run python -m housing.data
    deps:
      - src/housing/data.py
    outs:
      - data/raw/housing.csv

  validate:
    cmd: uv run python -m housing.validate
    deps:
      - src/housing/validate.py
      - data/raw/housing.csv

  prepare:
    cmd: uv run python -m housing.features
    deps:
      - src/housing/features.py
      - data/raw/housing.csv
    params:
      - prepare
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet

  train:
    cmd: uv run python -m housing.train
    deps:
      - src/housing/train.py
      - src/housing/evaluate.py
      - data/processed/train.parquet
    params:
      - train
      - mlflow
    outs:
      - models/model.joblib

  evaluate:
    cmd: uv run python -m housing.evaluate
    deps:
      - src/housing/evaluate.py
      - models/model.joblib
      - data/processed/test.parquet
    metrics:
      - metrics/metrics.json:
          cache: false       # tiny file, keep it in git so history is visible
```

Since `ingest` now produces `data/raw/housing.csv` as a pipeline output, remove the old
standalone tracking: `uv run dvc remove data/raw/housing.csv.dvc` (keep the file).

### 3.3 Concepts and Build: Data validation

**The most common production ML incident is bad input data**, not a bad model. A
column gets renamed upstream, a unit changes from meters to feet, nulls appear where
they never did. The model happily predicts garbage.

A **data contract** is an explicit schema: column names, types, ranges, nullability.
Validate at the pipeline entrance and fail loudly. We use **pandera**, which expresses
schemas as code. (Alternative: Great Expectations, heavier but with richer reporting.)

```bash
uv add "pandera>=0.20"
```

`src/housing/validate.py`:

```python
"""Stage: validate raw data against a contract.

If this stage fails, nothing downstream runs. That is the point.
"""

import logging

import pandas as pd
import pandera.pandas as pa

from housing.config import load_params, resolve

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Ranges come from knowledge of the domain (California geography, census definitions)
# plus inspection of the training data. Tighten them as you learn more.
RAW_SCHEMA = pa.DataFrameSchema(
    {
        "MedInc": pa.Column(float, pa.Check.between(0, 20)),
        "HouseAge": pa.Column(float, pa.Check.between(1, 60)),
        "AveRooms": pa.Column(float, pa.Check.gt(0)),
        "AveBedrms": pa.Column(float, pa.Check.gt(0)),
        "Population": pa.Column(float, pa.Check.ge(1)),
        "AveOccup": pa.Column(float, pa.Check.gt(0)),
        "Latitude": pa.Column(float, pa.Check.between(32, 42.5)),
        "Longitude": pa.Column(float, pa.Check.between(-125, -114)),
        "MedHouseVal": pa.Column(float, pa.Check.between(0, 5.1)),
    },
    strict=True,   # no unexpected extra columns
    coerce=True,   # cast ints to float rather than failing on dtype
)


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raises pandera.errors.SchemaErrors with a full report if anything is wrong."""
    return RAW_SCHEMA.validate(df, lazy=True)  # lazy=True collects ALL failures, not just the first


def validate() -> None:
    params = load_params()
    df = pd.read_csv(resolve(params["data"]["raw_path"]))
    validate_frame(df)
    log.info("Validation passed: %d rows, %d columns", *df.shape)


if __name__ == "__main__":
    validate()
```

Run the full pipeline:

```bash
uv run dvc repro
```

Watch it run every stage, then run `uv run dvc repro` again: every stage is skipped
because nothing changed. Now edit `params.yaml` (change `max_depth` to 10) and rerun:
only `train` and `evaluate` run. That is the DAG doing its job.

Look at `dvc.lock`: it records the hash of every dependency and output for this run.
Commit it with the code.

Useful commands:

```bash
uv run dvc dag                # draw the pipeline
uv run dvc metrics show       # current metrics
uv run dvc params diff        # params changed vs last commit
uv run dvc metrics diff       # metrics changed vs last commit
```

Update the Makefile so `pipeline` runs `uv run dvc repro`, then commit:

```bash
git add . && git commit -m "DVC pipeline with data validation stage"
uv run dvc push
```

### 3.4 Concepts: Testing ML code

ML code has three layers of tests, each catching a different class of bug:

| Layer | What it tests | Speed | Example |
|---|---|---|---|
| **Unit tests** | Pure functions: feature logic, metric computation, config loading | ms | `build_features` clips `AveOccup` at 50 |
| **Data tests** | Contracts and invariants of data | ms to s | schema rejects a negative population |
| **Model tests** | Behavior of a trained model | s to min | RMSE on test set below threshold; prediction increases when income increases |

Most teams have unit and data tests in CI on every push, and model tests on a schedule
or when training code changes (they are slower).

### 3.5 Build: Write the tests

`tests/conftest.py` (shared fixtures):

```python
import pandas as pd
import pytest


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """A tiny, valid slice of raw data. Hand-written so tests don't need the real file."""
    return pd.DataFrame(
        {
            "MedInc": [8.3252, 3.1, 1.9],
            "HouseAge": [41.0, 20.0, 52.0],
            "AveRooms": [6.98, 5.1, 4.0],
            "AveBedrms": [1.02, 1.1, 1.0],
            "Population": [322.0, 1200.0, 800.0],
            "AveOccup": [2.55, 3.0, 120.0],  # last one is an absurd outlier
            "Latitude": [37.88, 34.1, 36.5],
            "Longitude": [-122.23, -118.2, -119.0],
            "MedHouseVal": [4.526, 2.1, 0.9],
        }
    )
```

`tests/test_features.py`:

```python
from housing.features import FEATURE_COLUMNS, build_features


def test_build_features_returns_expected_columns(raw_sample):
    out = build_features(raw_sample)
    assert list(out.columns) == FEATURE_COLUMNS


def test_build_features_clips_occupancy_outliers(raw_sample):
    out = build_features(raw_sample)
    assert out["AveOccup"].max() <= 50


def test_build_features_does_not_mutate_input(raw_sample):
    before = raw_sample.copy()
    build_features(raw_sample)
    assert raw_sample.equals(before)
```

`tests/test_validate.py`:

```python
import pandera.errors
import pytest

from housing.validate import validate_frame


def test_valid_data_passes(raw_sample):
    validate_frame(raw_sample)


def test_negative_population_is_rejected(raw_sample):
    bad = raw_sample.copy()
    bad.loc[0, "Population"] = -5
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_frame(bad)


def test_unexpected_column_is_rejected(raw_sample):
    bad = raw_sample.assign(Surprise=1.0)
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_frame(bad)


def test_missing_column_is_rejected(raw_sample):
    bad = raw_sample.drop(columns=["Latitude"])
    with pytest.raises(pandera.errors.SchemaErrors):
        validate_frame(bad)
```

`tests/test_model.py` (model-quality tests, marked slow):

```python
"""Behavioral and quality tests for the trained model.

These need a trained model on disk (run `dvc repro` first) and are marked
`slow` so CI can choose to skip them on quick checks.
"""

import json

import joblib
import pandas as pd
import pytest

from housing.config import load_params, resolve

pytestmark = pytest.mark.slow

MIN_R2 = 0.75  # the floor below which we refuse to ship


@pytest.fixture(scope="module")
def model():
    path = resolve(load_params()["train"]["model_path"])
    if not path.exists():
        pytest.skip("No trained model found; run `dvc repro` first")
    return joblib.load(path)


@pytest.fixture(scope="module")
def metrics():
    path = resolve(load_params()["evaluate"]["metrics_path"])
    if not path.exists():
        pytest.skip("No metrics file found")
    with open(path) as f:
        return json.load(f)


def test_model_meets_quality_floor(metrics):
    assert metrics["r2"] >= MIN_R2, f"r2={metrics['r2']:.3f} is below floor {MIN_R2}"


def test_prediction_increases_with_income(model):
    """Directional expectation: richer neighborhoods -> pricier houses.
    A model that violates this has learned something wrong, regardless of RMSE."""
    base = pd.DataFrame([{
        "MedInc": 3.0, "HouseAge": 30.0, "AveRooms": 5.0, "AveBedrms": 1.0,
        "Population": 1000.0, "AveOccup": 3.0, "Latitude": 34.0, "Longitude": -118.0,
    }])
    richer = base.assign(MedInc=9.0)
    assert model.predict(richer)[0] > model.predict(base)[0]


def test_predictions_in_plausible_range(model):
    df = pd.read_parquet(resolve(load_params()["data"]["processed_dir"]) / "test.parquet")
    preds = model.predict(df.drop(columns=["MedHouseVal"]))
    assert preds.min() >= 0
    assert preds.max() <= 6  # target max is ~5.0; a little headroom
```

Register the marker in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["slow: tests that need a trained model"]
```

Run:

```bash
make test                     # everything
uv run pytest -m "not slow"   # fast only
```

Commit.

### 3.6 Build: Code quality automation with pre-commit

Pre-commit runs linters before every commit so bad code never enters the repo.

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-added-large-files   # stops you committing a 500MB CSV by accident
```

```bash
uv run pre-commit install
uv run pre-commit run --all-files
git add . && git commit -m "Tests and pre-commit hooks"
```

### 3.7 Concepts: Continuous Integration for ML

CI means: every push runs automated checks on a clean machine. "Clean machine" is the
key. It proves your project works from `git clone`, not just from your laptop with its
accumulated state.

For ML projects, CI typically has tiers:

1. **Every push**: lint, unit tests, data tests. Under 2 minutes.
2. **Pull requests to main**: everything above plus a *training smoke test* on a small
   data sample to prove the pipeline runs end to end.
3. **Merges to main or nightly**: full training, model tests, and (Day 5) automatic
   promotion if the model beats the champion.

### 3.8 Build: GitHub Actions workflow

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --extra dev

      - name: Lint
        run: make lint

      - name: Fast tests
        run: uv run pytest -m "not slow"

  train-smoke:
    # Proves the whole pipeline runs from a clean checkout. Uses a local
    # MLflow file store since there is no server in CI.
    runs-on: ubuntu-latest
    needs: checks
    if: github.event_name == 'pull_request'
    env:
      MLFLOW_TRACKING_URI: file:./mlruns
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --extra dev
      - name: Run pipeline
        run: uv run dvc repro
      - name: Model quality tests
        run: uv run pytest -m slow
      - name: Show metrics
        run: cat metrics/metrics.json
```

Push and watch it run:

```bash
git add . && git commit -m "GitHub Actions CI"
git push -u origin main
```

Open the **Actions** tab on GitHub. The `checks` job should go green. To see
`train-smoke` run, open a branch and a pull request:

```bash
git checkout -b experiment/deeper-trees
# change max_depth to 14 in params.yaml
uv run dvc repro && uv run dvc push
git commit -am "Try max_depth=14" && git push -u origin experiment/deeper-trees
```

Open a PR on GitHub. Both jobs run. The PR page shows exactly what would ship. Merge it
if metrics improved, close it if not.

> **Note on data in CI.** The smoke test downloads data from scikit-learn, so it works
> without a DVC remote. In a real project you would either (a) configure DVC remote
> credentials as GitHub secrets and `dvc pull`, or (b) keep a small sample dataset in
> the repo specifically for CI. Option (b) is the most common and robust pattern.

### 3.9 Exercises

1. **Add a derived feature.** In `build_features`, add `RoomsPerHousehold = AveRooms /
   AveOccup`. Add it to `FEATURE_COLUMNS`. Write a unit test for it. Run `dvc repro`
   and `dvc metrics diff`. Did it help?
2. **Make validation smarter.** Add a pandera check that `AveBedrms <= AveRooms` for
   every row (a dataframe-level check). Write a test for it.
3. **Break CI on purpose.** Push a commit that fails lint. Watch CI go red. Fix it.
   Knowing what red looks like is important.
4. **Add a CI sample dataset.** Create `tests/fixtures/housing_sample.csv` with 500
   rows. Add an environment variable `HOUSING_RAW_PATH` that `data.py` honors to skip
   downloading. Use it in the `train-smoke` job.
5. **Metric gate.** Add a step in `train-smoke` that fails if `r2 < 0.75` using `jq`
   or a tiny Python snippet. This is the seed of an automated promotion gate.

### 3.10 Checkpoint questions

1. What does `dvc repro` do differently from `make pipeline`, and what is in `dvc.lock`?
2. Why does validation sit *before* feature preparation and not after?
3. Give one example each of a unit test, a data test, and a model test from your project. Which run in CI on every push and why?
4. What is a directional (behavioral) model test and why does it matter even when RMSE is good?
5. Why is "works from a clean checkout" the real definition of CI passing?
6. You now have automated, reproducible, tested training. Which MLOps maturity level is that, and what is still missing for Level 2?

---

## Day 4: Serving Models: APIs, Docker, and Deployment

### Learning objectives

- Choose between batch, online, and streaming inference for a use case.
- Build a production-grade prediction API with FastAPI and pydantic validation.
- Containerize the service with Docker using best practices.
- Run the whole stack locally with Docker Compose.
- Deploy the container to a cloud service and automate image builds in CI.

### 4.1 Concepts: Serving patterns

| Pattern | How it works | Use when | Example |
|---|---|---|---|
| **Batch** | A scheduled job scores a whole table and writes results to a database | Predictions are not needed instantly; large volumes | Nightly churn scores for all customers |
| **Online (real-time)** | An API returns a prediction per request in milliseconds | A user or system is waiting | Price estimate on a listing page |
| **Streaming** | Model consumes events from a queue (Kafka) and emits predictions | Continuous event flows | Fraud scoring on each transaction |
| **Embedded / edge** | Model ships inside the app or device | No network, strict latency, privacy | On-phone keyboard suggestions |

Batch is cheapest and simplest; start there whenever the product allows. We build
online serving today because it exercises the most concepts; batch is a short exercise.

**Where should the API get the model?** Two options:

1. **Bake it into the container image.** Simple, immutable, no runtime dependency on
   MLflow. Deploying a new model = building a new image. Good default.
2. **Load from the registry at startup** (`models:/name@champion`). Swap models without
   rebuilding. Requires the API to reach the MLflow server, and model and code versions
   can drift apart.

We support both via an environment variable, defaulting to the baked-in model.

### 4.2 Build: The FastAPI service

```bash
uv add "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.8"
uv add --optional dev "httpx>=0.27"       # for testing the API
```

`src/housing/api/__init__.py` (empty) and `src/housing/api/schemas.py`:

```python
"""Request/response contracts for the API.

pydantic validates every request: wrong types, missing fields, and out-of-range
values are rejected with a 422 before the model ever sees them.
"""

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., ge=0, le=20, description="Median income in block, tens of thousands USD")
    HouseAge: float = Field(..., ge=1, le=60)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., ge=1)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32, le=42.5)
    Longitude: float = Field(..., ge=-125, le=-114)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
                    "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23,
                }
            ]
        }
    }


class PredictionRequest(BaseModel):
    instances: list[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
```

`src/housing/api/model_loader.py`:

```python
"""Load the model from either a local file or the MLflow registry.

Controlled by environment variables so the same image works everywhere:
  MODEL_SOURCE=local  (default)  -> load MODEL_PATH with joblib
  MODEL_SOURCE=mlflow            -> load MODEL_URI via mlflow.pyfunc
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

import joblib

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    model: Any
    version: str

    def predict(self, df):
        return self.model.predict(df)


def load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")

    if source == "mlflow":
        import mlflow

        uri = os.getenv("MODEL_URI", "models:/housing-price-regressor@champion")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        log.info("Loading model from MLflow: %s", uri)
        model = mlflow.pyfunc.load_model(uri)
        version = model.metadata.run_id[:8]
        return LoadedModel(model=model, version=f"mlflow:{version}")

    path = os.getenv("MODEL_PATH", "models/model.joblib")
    log.info("Loading model from local path: %s", path)
    version = os.getenv("MODEL_VERSION", "local")
    return LoadedModel(model=joblib.load(path), version=version)
```

`src/housing/api/app.py`:

```python
"""Prediction service.

Design decisions, each one a production lesson:
- The model loads ONCE at startup (lifespan), not per request.
- /health answers only after the model is loaded, so orchestrators don't route
  traffic to an instance that can't predict yet.
- Features go through the SAME build_features() used in training.
- Every prediction is logged in a structured way (Day 5 turns this into monitoring).
"""

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from housing.api.model_loader import LoadedModel, load_model
from housing.api.schemas import PredictionRequest, PredictionResponse
from housing.features import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

state: dict[str, LoadedModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    log.info("Model loaded, version=%s", state["model"].version)
    yield
    state.clear()


app = FastAPI(title="Housing Price API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": state["model"].version}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    start = time.perf_counter()
    raw = pd.DataFrame([i.model_dump() for i in req.instances])
    features = build_features(raw)
    preds = state["model"].predict(features)
    latency_ms = (time.perf_counter() - start) * 1000

    log.info(
        "prediction n=%d latency_ms=%.1f mean_pred=%.3f model=%s",
        len(preds), latency_ms, float(preds.mean()), state["model"].version,
    )
    return PredictionResponse(
        predictions=[float(p) for p in preds],
        model_version=state["model"].version,
    )
```

Run it:

```bash
uv run uvicorn housing.api.app:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs. FastAPI generated interactive documentation from your
pydantic schemas. Click **POST /predict → Try it out → Execute**. Then try it from the
terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"MedInc":8.3252,"HouseAge":41,"AveRooms":6.98,"AveBedrms":1.02,"Population":322,"AveOccup":2.55,"Latitude":37.88,"Longitude":-122.23}]}'
```

Try sending `"Latitude": 90`. You get a 422 with a precise explanation. The model was
never called.

Add `serve: uv run uvicorn housing.api.app:app --reload --port 8000` to the Makefile.

### 4.3 Build: API tests

`tests/test_api.py`:

```python
import pytest
from fastapi.testclient import TestClient

from housing.api.app import app

VALID = {
    "MedInc": 8.3252, "HouseAge": 41, "AveRooms": 6.98, "AveBedrms": 1.02,
    "Population": 322, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23,
}


@pytest.fixture(scope="module")
def client():
    # TestClient as a context manager triggers the lifespan (model load).
    with TestClient(app) as c:
        yield c


@pytest.mark.slow
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.slow
def test_predict_valid(client):
    r = client.post("/predict", json={"instances": [VALID]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    assert 0 <= body["predictions"][0] <= 6


@pytest.mark.slow
def test_predict_rejects_out_of_range(client):
    bad = {**VALID, "Latitude": 90}
    r = client.post("/predict", json={"instances": [bad]})
    assert r.status_code == 422


@pytest.mark.slow
def test_predict_rejects_empty(client):
    r = client.post("/predict", json={"instances": []})
    assert r.status_code == 422
```

`make test` should pass (with a model on disk). Commit.

### 4.4 Concepts: Docker in ten minutes

A **container** packages your code, its dependencies, and the exact OS libraries into
one image that runs identically on your laptop, in CI, and on a server. It is the
answer to "works on my machine."

Key terms:

- **Image**: an immutable snapshot, built from a `Dockerfile`. Like a class.
- **Container**: a running instance of an image. Like an object.
- **Layer**: each Dockerfile instruction creates a cached layer. Order instructions
  from least-changing (base OS, dependencies) to most-changing (your code) so rebuilds
  are fast.
- **Registry**: where images are stored and pulled from (Docker Hub, GitHub Container
  Registry, AWS ECR, Google Artifact Registry).

Best practices you will apply:

1. Use a slim base image.
2. Install dependencies before copying code, so code changes do not reinstall everything.
3. Do not run as root.
4. One process per container.
5. Use `.dockerignore` so `.venv`, data, and `.git` are not sent to the build.

### 4.5 Build: Dockerfile

`.dockerignore`:

```
.git
.venv
.dvc/cache
data
mlruns
mlartifacts
mlflow.db
notebooks
tests
monitoring
*.md
__pycache__
```

`Dockerfile`:

```dockerfile
# ---- Stage 1: build the virtual environment ----------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first (cached unless pyproject/lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Then the code
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- Stage 2: minimal runtime image ------------------------------------------
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY models/model.joblib /app/models/model.joblib
COPY params.yaml /app/params.yaml

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_SOURCE=local \
    MODEL_PATH=/app/models/model.joblib \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "housing.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why `--no-dev`?** The runtime image does not need pytest, ruff, DVC, or MLflow's
server components. Smaller image = faster deploys and smaller attack surface. (If you
use `MODEL_SOURCE=mlflow`, the `mlflow` client is in main dependencies already.)

Build and run:

```bash
docker build -t housing-api:dev .
docker run --rm -p 8000:8000 housing-api:dev
```

In another terminal, run the same `curl` commands as before. Check image size with
`docker images housing-api`. Aim for under 600 MB; scikit-learn and pandas are most of it.

Add a `MODEL_VERSION` build arg so the image knows what it contains:

```dockerfile
ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=${MODEL_VERSION}
```

(Add these two lines in the runtime stage, before `USER appuser`.) Build with
`docker build --build-arg MODEL_VERSION=$(git rev-parse --short HEAD) -t housing-api:dev .`
and check `/health` reports it.

Makefile additions:

```makefile
docker-build:
	docker build --build-arg MODEL_VERSION=$$(git rev-parse --short HEAD) -t housing-api:dev .

docker-run:
	docker run --rm -p 8000:8000 housing-api:dev
```

Commit.

### 4.6 Build: The full local stack with Docker Compose

Compose runs several containers together with one command. We bring up the API, an
MLflow server, and (for Day 5) Prometheus and Grafana.

`docker-compose.yml`:

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.1.0
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0 --port 5000
    ports: ["5000:5000"]
    volumes:
      - mlflow-data:/mlflow

  api:
    build:
      context: .
      args:
        MODEL_VERSION: compose-dev
    ports: ["8000:8000"]
    environment:
      MODEL_SOURCE: local
    depends_on: [mlflow]

  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    depends_on: [prometheus]

volumes:
  mlflow-data:
```

`monitoring/prometheus.yml` (Prometheus will have nothing to scrape until Day 5, but the
file must exist):

```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: housing-api
    static_configs:
      - targets: ["api:8000"]
```

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- MLflow: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

`docker compose down` to stop. Add `up: docker compose up --build` and
`down: docker compose down` to the Makefile. Commit.

### 4.7 Build: Load testing

Before deploying, know your numbers: requests per second and p95 latency. Install
`hey` (https://github.com/rakyll/hey) or use `locust`. With `hey`:

```bash
echo '{"instances":[{"MedInc":8.3,"HouseAge":41,"AveRooms":6.98,"AveBedrms":1.02,"Population":322,"AveOccup":2.55,"Latitude":37.88,"Longitude":-122.23}]}' > /tmp/body.json
hey -n 2000 -c 20 -m POST -H "Content-Type: application/json" -D /tmp/body.json http://localhost:8000/predict
```

Read the output: requests/sec, latency distribution, status code histogram. Write the
numbers in `NOTES.md`. A random forest with 300 trees is slow per prediction; this is
where you learn that model choice is also an infrastructure decision.

### 4.8 Concepts: Deployment targets

Your container can run anywhere. Common targets in order of complexity:

| Target | Complexity | Good for |
|---|---|---|
| **Serverless containers** (Google Cloud Run, AWS App Runner, Azure Container Apps) | Low | APIs with variable traffic; scale to zero; pay per request. **Best first choice.** |
| **Managed container services** (AWS ECS/Fargate) | Medium | Steady workloads, more control |
| **Kubernetes** (GKE, EKS, AKS) | High | Many services, GPU scheduling, custom autoscaling. Learn after this week. |
| **Managed ML endpoints** (SageMaker, Vertex AI Endpoints, Azure ML) | Medium | When you want the cloud to handle model-specific features (A/B, built-in monitoring) |

### 4.9 Build: Deploy to Google Cloud Run (or read along if you skip the cloud)

Cloud Run has a generous free tier. If you prefer AWS, the equivalent is App Runner;
the steps map one to one. If you do not want to create a cloud account, read this
section and do the "deploy" to your local Docker instead; everything else in the
course still works.

1. Create a GCP project, enable billing (free tier covers this), install `gcloud`.
2. Authenticate and set defaults:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

3. Create an image repository and push:

```bash
gcloud artifacts repositories create ml --repository-format=docker --location=us-central1
gcloud auth configure-docker us-central1-docker.pkg.dev

IMAGE=us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml/housing-api:$(git rev-parse --short HEAD)
docker build --build-arg MODEL_VERSION=$(git rev-parse --short HEAD) -t $IMAGE .
docker push $IMAGE
```

4. Deploy:

```bash
gcloud run deploy housing-api \
  --image $IMAGE \
  --port 8000 \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 3 \
  --memory 1Gi --cpu 1
```

5. Test the printed URL:

```bash
URL=$(gcloud run services describe housing-api --format 'value(status.url)')
curl -s $URL/health
```

You have a model in production. Note the startup time on a cold start (first request
after idle); it is the model load. `--min-instances 1` removes cold starts at a cost.

**Rollback**: `gcloud run services update-traffic housing-api --to-revisions PREVIOUS_REVISION=100`.
Cloud Run keeps every revision. List them with `gcloud run revisions list`.

### 4.10 Build: Continuous Delivery of the image

Extend CI so merges to `main` build and publish the image to GitHub Container
Registry. (Pushing to the cloud and deploying is the same idea with cloud credentials
in secrets; see the exercise.)

Add to `.github/workflows/ci.yml`:

```yaml
  build-image:
    runs-on: ubuntu-latest
    needs: checks
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    env:
      MLFLOW_TRACKING_URI: file:./mlruns
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --extra dev

      # The image needs a model. Train it here so the image is self-contained.
      - name: Train model
        run: uv run dvc repro

      - name: Model quality gate
        run: uv run pytest -m slow

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          build-args: MODEL_VERSION=${{ github.sha }}
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
```

Push to `main`. After the workflow succeeds, the image appears under **Packages** on
your GitHub profile. Anyone (with permission) can `docker pull ghcr.io/YOU/housing-mlops:latest`.

Commit everything.

### 4.11 Exercises

1. **Batch inference.** Write `src/housing/batch_predict.py` that reads a parquet of
   features, scores them with the local model, and writes `predictions.parquet` with
   a timestamp and model version column. Add it as a DVC stage. This is the pattern
   for 80% of real-world ML.
2. **Registry-backed serving.** Run the API with `MODEL_SOURCE=mlflow
   MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. Promote a different version to
   `champion`, restart the API, confirm the version changed with no rebuild.
3. **Add a `/model-info` endpoint** returning version, training date, and metrics
   (bundle `metrics.json` into the image).
4. **Deploy from CI.** Add a `deploy` job that runs after `build-image` and calls
   `gcloud run deploy` using a service-account key stored as a GitHub secret
   (`google-github-actions/auth` + `google-github-actions/deploy-cloudrun`).
5. **Shrink the image.** Try `python:3.12-slim` vs `python:3.12-alpine` (hint: alpine
   is often slower to build for numeric packages and not worth it). Measure.

### 4.12 Checkpoint questions

1. When would you choose batch over online inference? Give a concrete example of each.
2. What is training-serving skew, and which line of `app.py` prevents it?
3. Why does the model load in `lifespan` instead of inside `/predict`?
4. Why does `/health` return 503 before the model is loaded, and who consumes that signal?
5. Explain the two Docker stages and why dependencies are installed before code is copied.
6. What are the trade-offs between baking the model into the image and loading from the registry?

---

## Day 5: Monitoring, Drift, Retraining, and Production Operations

### Learning objectives

- Distinguish system monitoring from model monitoring, and know what to track in each.
- Expose Prometheus metrics from the API and view them in Grafana.
- Log predictions for later analysis.
- Detect data drift with Evidently and decide when to retrain.
- Automate retraining and conditional promotion.
- Write a model card and a runbook, and know how to handle an incident.

### 5.1 Concepts: The two kinds of monitoring

**System monitoring** (same as any service): is it up, how fast, how many errors?

- Request rate, error rate, latency percentiles (the "RED" metrics)
- CPU, memory, instance count
- Alerts: error rate > 1%, p95 latency > 500 ms, zero traffic for 10 minutes

**Model monitoring** (unique to ML): is it still *right*?

| Signal | What it measures | Delay | How |
|---|---|---|---|
| **Input data drift** | Have the feature distributions changed vs training data? | None | Compare live inputs with a reference sample (statistical tests) |
| **Prediction drift** | Has the output distribution shifted? | None | Track mean/quantiles of predictions over time |
| **Data quality** | Nulls, out-of-range, schema violations in live traffic | None | Same pandera checks as training, counted |
| **Model performance** | Actual accuracy/RMSE in production | Days to months (needs ground truth) | Join predictions with later-observed outcomes |

The painful truth: **ground truth is usually late.** You predict a house price today;
the sale closes in 60 days. So drift monitoring is the early warning; performance
monitoring is the confirmation.

### 5.2 Build: Prometheus metrics in the API

Prometheus is the standard for metrics. Your service exposes a `/metrics` endpoint;
Prometheus scrapes it every 15 seconds and stores time series; Grafana draws graphs
and fires alerts.

```bash
uv add "prometheus-fastapi-instrumentator>=7.0" "prometheus-client>=0.20"
```

Update `src/housing/api/app.py`. Add imports:

```python
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
```

After `app = FastAPI(...)`:

```python
# Standard HTTP metrics (request count, latency by route and status) for free
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Model-specific metrics
PREDICTION_VALUE = Histogram(
    "housing_prediction_value",
    "Distribution of predicted house values (units of $100k)",
    buckets=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6],
)
PREDICTIONS_TOTAL = Counter("housing_predictions_total", "Number of predictions served")
FEATURE_MEDINC = Histogram(
    "housing_input_medinc",
    "Distribution of MedInc feature in live requests",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10, 15],
)
```

Inside `predict`, after computing `preds`:

```python
    PREDICTIONS_TOTAL.inc(len(preds))
    for p in preds:
        PREDICTION_VALUE.observe(float(p))
    for v in raw["MedInc"]:
        FEATURE_MEDINC.observe(float(v))
```

Restart the stack (`make down && make up`), send some traffic with `hey`, then:

- http://localhost:8000/metrics shows raw metrics.
- http://localhost:9090 → Status → Targets shows `housing-api` as UP.
- In Prometheus, query `rate(housing_predictions_total[1m])` and
  `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))`.

**Grafana dashboard.** In Grafana (localhost:3000), add Prometheus as a data source
(URL `http://prometheus:9090`), create a dashboard with panels for:

1. Requests per second: `rate(http_requests_total[1m])`
2. p95 latency: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))`
3. Error rate: `rate(http_requests_total{status=~"5.."}[5m])`
4. Mean prediction: `rate(housing_prediction_value_sum[5m]) / rate(housing_prediction_value_count[5m])`
5. Mean MedInc input: same pattern on `housing_input_medinc`

Panel 4 and 5 are your first model monitors. If the mean prediction suddenly jumps
while mean input income does not, something is wrong with the model or the code.

Export the dashboard JSON (Dashboard settings → JSON model) into
`monitoring/grafana-dashboard.json` and commit it. Dashboards are code too.

### 5.3 Build: Prediction logging

Metrics give aggregates; for drift analysis and debugging you need the actual rows.
Log every request's features and prediction as structured JSON lines. In production
this goes to a log pipeline or directly to a warehouse table; locally, a file.

`src/housing/api/prediction_log.py`:

```python
"""Append-only structured log of predictions.

In production, replace the file with a message queue or warehouse writer.
NEVER log personally identifiable information here without a retention policy.
"""

import json
import os
import threading
import time
from pathlib import Path

_lock = threading.Lock()
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "monitoring/predictions.jsonl"))


def log_predictions(features, predictions, model_version: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    records = features.to_dict(orient="records")
    with _lock, open(LOG_PATH, "a") as f:
        for row, pred in zip(records, predictions, strict=True):
            f.write(json.dumps({"ts": ts, "model_version": model_version, **row, "prediction": float(pred)}) + "\n")
```

Call it from `predict` after computing `preds`:

```python
    log_predictions(features, preds, state["model"].version)
```

Mount a volume in compose so the file persists: under the `api` service add
`volumes: ["./monitoring:/app/monitoring"]` and `environment: PREDICTION_LOG_PATH: /app/monitoring/predictions.jsonl`.
(The container runs as `appuser` uid 1000; make sure `monitoring/` is writable:
`chmod 777 monitoring` locally is fine for learning.)

### 5.4 Concepts: Drift detection

Compare a **reference** dataset (what the model trained on) with a **current** window
(recent live inputs). For each feature, a statistical test asks "could these two
samples come from the same distribution?"

- Numeric features: Kolmogorov-Smirnov test, Wasserstein distance, Population
  Stability Index (PSI)
- Categorical: chi-squared, Jensen-Shannon divergence

**Evidently** packages these into reports with sensible defaults. Rules of thumb:

- PSI < 0.1: no drift. 0.1–0.25: moderate, investigate. > 0.25: significant.
- Drift in one feature out of eight is noise. Drift in half of them is a signal.
- Drift in the *most important* feature matters more than in a weak one.
- Always ask "did the world change, or did the pipeline break?" A renamed column
  looks like 100% drift.

### 5.5 Build: Drift report with Evidently

```bash
uv add --optional dev "evidently>=0.7"
```

Save a reference sample during training. Add to `train.py` after loading the data:

```python
    ref_path = resolve("monitoring/reference.parquet")
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    X.sample(n=min(5000, len(X)), random_state=0).to_parquet(ref_path, index=False)
```

And add `monitoring/reference.parquet` to the `train` stage `outs` in `dvc.yaml`.

`src/housing/drift.py`:

```python
"""Compare recent live inputs against the training reference sample.

Usage: python -m housing.drift [--current monitoring/predictions.jsonl]
Writes an HTML report and prints a JSON summary; exits 1 if drift is detected
so it can gate a retraining job.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from housing.config import resolve
from housing.features import FEATURE_COLUMNS


def load_current(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return df[FEATURE_COLUMNS]


def run_drift(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> dict:
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_html))

    result = snapshot.dict()
    # Pull out the headline numbers. Structure varies slightly by version; inspect
    # `result` once in a shell to confirm these keys for your Evidently release.
    summary = {}
    for m in result["metrics"]:
        if m["metric_id"].startswith("DriftedColumnsCount"):
            summary["drifted_columns"] = m["value"]["count"]
            summary["share_drifted"] = m["value"]["share"]
    summary["drift_detected"] = summary.get("share_drifted", 0) >= 0.5
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--current", default="monitoring/predictions.jsonl")
    p.add_argument("--out", default="monitoring/reports/drift.html")
    args = p.parse_args()

    ref = pd.read_parquet(resolve("monitoring/reference.parquet"))
    cur = load_current(resolve(args.current))
    summary = run_drift(ref, cur, resolve(args.out))
    print(json.dumps(summary, indent=2))
    sys.exit(1 if summary["drift_detected"] else 0)
```

**Simulate drift.** Send 500 normal requests, then 500 with `MedInc` tripled and
`HouseAge` set to 5 (a "new luxury development" scenario). A small script,
`scripts/simulate_traffic.py`:

```python
import random
import sys

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/predict"
drift = len(sys.argv) > 2 and sys.argv[2] == "drift"


def row():
    r = {
        "MedInc": random.uniform(1.5, 8), "HouseAge": random.uniform(5, 50),
        "AveRooms": random.uniform(3, 8), "AveBedrms": random.uniform(0.9, 1.3),
        "Population": random.uniform(300, 3000), "AveOccup": random.uniform(2, 4),
        "Latitude": random.uniform(33, 39), "Longitude": random.uniform(-123, -117),
    }
    if drift:
        r["MedInc"] *= 3
        r["HouseAge"] = 5.0
    return r


for _ in range(50):
    httpx.post(URL, json={"instances": [row() for _ in range(10)]}, timeout=30).raise_for_status()
print("done")
```

```bash
uv run python scripts/simulate_traffic.py
uv run python -m housing.drift        # no drift expected (exit 0)
uv run python scripts/simulate_traffic.py http://localhost:8000/predict drift
uv run python -m housing.drift        # drift detected (exit 1)
open monitoring/reports/drift.html    # xdg-open on Linux
```

Read the HTML report. For each feature it shows the reference vs current distribution
and the test result. This is what you would attach to an incident ticket.

Add `drift: uv run python -m housing.drift` to the Makefile. Commit.

### 5.6 Concepts: Retraining strategy

When to retrain:

| Trigger | Pros | Cons |
|---|---|---|
| **Scheduled** (weekly, monthly) | Simple, predictable | May retrain needlessly or too late |
| **Drift-triggered** | Responds to actual change | Drift does not always hurt performance; can thrash |
| **Performance-triggered** | Directly tied to what matters | Needs ground truth, which is delayed |
| **Data-volume-triggered** | "Every 100k new labeled rows" | Only works when labels arrive steadily |

Most teams start with **scheduled + a manual drift review**, then add automatic
drift-triggered retraining once they trust the pipeline.

**The promotion gate.** Retraining produces a *candidate*. It becomes champion only if:

1. It passes all model tests (quality floor, behavioral tests).
2. Its test-set metric beats the current champion by a margin (or at least does not
   regress beyond a tolerance).
3. (Optionally) it wins a shadow or canary comparison on live traffic.

Never auto-promote without a gate. A pipeline that silently ships a worse model is
worse than no pipeline.

### 5.7 Build: Automated retraining with a promotion gate

`src/housing/promote.py`:

```python
"""Compare the latest run against the current champion and promote if better.

Run after `dvc repro`. Reads the new metrics from metrics.json and the
champion's metrics from MLflow. Promotes only on improvement beyond a tolerance.
"""

import json
import os
import sys

import mlflow
from mlflow import MlflowClient

from housing.config import load_params, resolve
from housing.registry import promote, register

TOLERANCE = 0.005  # new r2 must beat champion by at least this


def latest_run_id(experiment_name: str) -> str:
    runs = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"], max_results=1)
    return runs.iloc[0]["run_id"]


def champion_r2(client: MlflowClient, name: str) -> float | None:
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("test_r2")


def main() -> int:
    params = load_params()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    name = params["mlflow"]["registered_model_name"]

    with open(resolve(params["evaluate"]["metrics_path"])) as f:
        new_r2 = json.load(f)["r2"]

    run_id = latest_run_id(params["mlflow"]["experiment_name"])
    # Attach the test metric to the run so future comparisons can read it
    client.log_metric(run_id, "test_r2", new_r2)

    current = champion_r2(client, name)
    print(f"candidate test_r2={new_r2:.4f}  champion test_r2={current}")

    if current is not None and new_r2 < current + TOLERANCE:
        print("Candidate does not beat champion; not promoting.")
        return 0

    version = register(run_id)
    promote(version, "champion")
    print(f"Promoted version {version} to champion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`.github/workflows/retrain.yml`:

```yaml
name: Scheduled retraining

on:
  schedule:
    - cron: "0 3 * * 1"      # every Monday 03:00 UTC
  workflow_dispatch:         # manual "Run workflow" button

jobs:
  retrain:
    runs-on: ubuntu-latest
    env:
      # In a real deployment this points at your hosted MLflow server and the
      # secrets below hold its credentials and the DVC remote credentials.
      MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI || 'file:./mlruns' }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv python install 3.12
      - run: uv sync --extra dev

      - name: Train
        run: uv run dvc repro

      - name: Quality gate
        run: uv run pytest -m slow

      - name: Compare and promote
        run: uv run python -m housing.promote

      - name: Upload metrics and reports
        uses: actions/upload-artifact@v4
        with:
          name: retrain-${{ github.run_id }}
          path: |
            metrics/metrics.json
            monitoring/reports/
```

Trigger it manually from the Actions tab (workflow_dispatch) to see it run. With a
file-based MLflow store in CI the registry does not persist between runs; that is fine
for learning. The exercise below connects it to a real server.

### 5.8 Concepts: Feature stores (know what they are)

A **feature store** is a system that computes features once and serves them to both
training (as historical, point-in-time-correct tables) and online inference (as a
low-latency key-value lookup). It solves training-serving skew at scale and lets teams
share features.

You do not need one today. You need one when:

- Multiple models share features (e.g. "customer 30-day spend")
- Features need expensive aggregation that cannot run per request
- Point-in-time correctness matters (no leaking future data into training)

Tools: Feast (open source), Tecton, Databricks Feature Store, Vertex AI Feature Store,
SageMaker Feature Store. Your `build_features` function is the zero-infrastructure
version of the same idea: one definition, used everywhere.

### 5.9 Build: Documentation that production requires

**Model card** (`docs/model_card.md`). Every shipped model should have one. Fill it in
for your model:

```markdown
# Model Card: housing-price-regressor

## Model details
- Type: RandomForestRegressor (scikit-learn)
- Version: <registry version> / git <commit>
- Trained: <date>, by: <you>
- Training pipeline: dvc.yaml in this repo

## Intended use
- Predict median house value for a California census block from 8 features.
- Intended consumers: internal pricing tool.
- NOT intended for: individual property appraisal, lending decisions.

## Training data
- California Housing (1990 census), 20,640 rows. Version: <DVC hash>.
- Known limitations: 1990 data; no features for condition, renovations, or schools.

## Evaluation
- Hold-out test set (20%, seed 42): RMSE <x>, MAE <y>, R² <z>.
- Behavioral checks: prediction increases with income (passes).

## Monitoring
- Input drift on 8 features vs training reference (Evidently, weekly).
- Prediction distribution via Prometheus histogram.
- Ground truth: not available; performance monitoring is not possible. Rely on drift.

## Ethical considerations and risks
- Location features can proxy for protected attributes. Do not use for any
  decision affecting individuals' access to housing or credit.
```

**Runbook** (`docs/runbook.md`). What the on-call person does at 3 a.m.:

```markdown
# Runbook: housing-api

## Service
- Image: ghcr.io/<you>/housing-mlops
- Endpoints: /health, /predict, /metrics, /docs
- Dashboards: Grafana "Housing API"

## Alerts and responses
| Alert | First check | Likely cause | Action |
|---|---|---|---|
| Error rate > 1% | Logs for 5xx; /health | Model failed to load; bad deploy | Roll back to previous image/revision |
| p95 latency > 500ms | Instance CPU; request batch sizes | Traffic spike; large batches | Scale up; enforce max_length on instances |
| Prediction mean shifted > 20% | Compare input histograms | Upstream data change or drift | Run `make drift`; if pipeline bug, fix upstream; if real drift, trigger retrain |
| Zero traffic 15 min | Upstream service health | Caller outage; DNS | Escalate to caller team |

## Rollback
1. Container: redeploy previous image tag (`:<previous sha>`).
2. Model only (registry-backed deploy): `python -m housing.registry promote <previous version>`; restart.

## Retraining
- Scheduled weekly (retrain.yml). Manual: Actions -> Scheduled retraining -> Run workflow.
- Promotion requires r2 improvement > 0.005 and all slow tests passing.
```

### 5.10 Concepts: Security and governance basics

- **Secrets** (API keys, cloud credentials, DB passwords) live in environment variables
  injected by the platform or CI secrets, never in code, never in images, never in
  MLflow params. Add `.env` to `.gitignore` (already done).
- **PII**: do not log raw personal data in prediction logs. Hash or drop identifiers.
  Know your retention policy.
- **Access control**: who can promote a model to champion? In MLflow this is managed
  by the hosting platform (Databricks, etc.). Document it.
- **Lineage**: for any prediction, you should be able to answer: which model version,
  which training run, which data version, which code commit. You built all of this.
  Trace it once end to end and write the chain in `NOTES.md`.
- **Dependency hygiene**: `uv.lock` pins everything; run `uv lock --upgrade` on a
  schedule, not ad hoc. Scan images (`docker scout cves housing-api:dev` or Trivy).

### 5.11 Build: Simulate an incident (the capstone drill)

Do this end to end without looking at the answers above:

1. Bring up the stack with `make up`.
2. Send normal traffic, then drifted traffic.
3. Notice the shift in Grafana (mean MedInc input panel).
4. Run the drift report. Confirm and read it.
5. Decide: is this a pipeline bug or real drift? (Here: real drift, by construction.)
6. Trigger retraining (locally: `make pipeline && uv run python -m housing.promote`
   with the MLflow server running). Observe whether it promotes.
7. Rebuild and redeploy the image with the new model.
8. Confirm `/health` reports the new version.
9. Write a 10-line incident report in `docs/incidents/2026-xx-xx-drift.md`: what
   happened, how it was detected, what was done, what should be automated next.

If you can do all nine steps, you have operated an ML system in production.

### 5.12 Exercises

1. **Alert rule.** Add a Prometheus alerting rule (`monitoring/alerts.yml`) that fires
   when `rate(http_requests_total{status=~"5.."}[5m]) > 0.01`. Wire it into
   `prometheus.yml`. Trigger it by stopping the model (e.g. delete the model file in
   the container) and watch it fire in Prometheus → Alerts.
2. **Hosted MLflow.** Run the MLflow server from your compose stack and point
   `retrain.yml` at it via an ngrok tunnel or a small VM, so the promotion gate
   actually persists across runs. (Or skip the network and run the retrain job
   locally against the compose MLflow.)
3. **Shadow deployment.** Add a `SHADOW_MODEL_PATH` option to the API that loads a
   second model, predicts with both, logs both, but returns only the champion's
   answer. This is how you evaluate a challenger on live traffic with zero risk.
4. **Ground truth join.** Fake a "sales" file with true values for 200 logged
   predictions. Write `src/housing/performance.py` that joins by a request id (add
   one to the prediction log) and computes live RMSE. Add it to the runbook.
5. **Data quality counter.** Count requests that would fail the pandera schema (run
   it in the API but do not block) and expose `housing_invalid_inputs_total`.

### 5.13 Checkpoint questions

1. Name three system metrics and three model metrics you would put on a dashboard for this service.
2. Why is prediction drift detectable immediately but model performance degradation usually not?
3. What is PSI, and roughly what values indicate moderate and significant drift?
4. Give two reasons a drift alert could be a false alarm.
5. What conditions must a retrained candidate meet before becoming champion in your pipeline? Why is a tolerance used?
6. Trace the lineage of one prediction from your API all the way back to the data version. List every artifact in the chain.
7. What does a model card contain that a README does not, and who is it for?

---

## Capstone Checklist

Tick every box. Each one corresponds to a production capability.

**Reproducibility**
- [ ] All config in `params.yaml`; no magic numbers in code
- [ ] Seeds set for every random operation
- [ ] `uv.lock` committed; `uv sync --frozen` works from a clean clone
- [ ] Data and model versioned with DVC; `dvc pull` restores them
- [ ] `dvc repro` reproduces identical metrics

**Experimentation**
- [ ] Every training run logged to MLflow with params, metrics, model, signature, config
- [ ] At least one sweep compared in the UI
- [ ] Best model registered; `champion` and `previous` aliases set
- [ ] Model loadable by alias

**Pipeline and quality**
- [ ] `dvc.yaml` defines ingest → validate → prepare → train → evaluate
- [ ] pandera schema rejects bad data with a full report
- [ ] Unit tests for features, data tests for schema, model tests for quality and behavior
- [ ] pre-commit with ruff
- [ ] CI runs lint + fast tests on every push, pipeline smoke test on PRs

**Serving**
- [ ] FastAPI service with pydantic validation, `/health`, `/predict`, `/docs`
- [ ] Shared `build_features` used by training and serving
- [ ] Multi-stage Dockerfile, non-root user, healthcheck
- [ ] `docker compose up` brings up API + MLflow + Prometheus + Grafana
- [ ] Image built and pushed to a registry from CI on merge to main
- [ ] Deployed to a cloud service (or documented why not)

**Operations**
- [ ] `/metrics` exposes HTTP and model metrics; Grafana dashboard committed
- [ ] Predictions logged as structured records
- [ ] Drift report runs against a training reference and exits non-zero on drift
- [ ] Scheduled retraining workflow with a promotion gate
- [ ] Model card and runbook written
- [ ] Incident drill completed and written up

---

## What to Learn Next

Ordered by typical impact for a new MLOps engineer:

1. **Kubernetes fundamentals** (pods, deployments, services, ingress, HPA). Deploy
   your API to a local cluster with `kind` or `minikube`, then to a managed one.
   Then look at **KServe** or **Seldon** for model-specific serving on Kubernetes.
2. **One cloud ML platform deeply**: SageMaker, Vertex AI, or Azure ML. Map every
   piece you built this week to its managed equivalent (pipelines, registry,
   endpoints, monitoring). Interviewers ask for this mapping.
3. **An orchestrator**: Prefect or Dagster for Python-native; Airflow because
   everyone has it. Rebuild your DVC pipeline as a scheduled flow with retries.
4. **Infrastructure as code**: Terraform. Define your cloud resources (bucket,
   registry, Cloud Run service) in code so environments are reproducible too.
5. **Data engineering basics**: SQL fluency, a warehouse (BigQuery/Snowflake), and
   Spark or Polars for data larger than memory. Most ML pipeline work is data work.
6. **Deep learning serving**: GPU containers, ONNX export, TorchServe / Triton,
   batching and quantization for latency.
7. **LLMOps**: prompt and dataset versioning, evaluation harnesses, tracing (MLflow
   and others now cover this), cost and latency monitoring, guardrails. The same
   lifecycle with new artifacts.
8. **Feature stores**: build a small Feast project; understand point-in-time joins.
9. **Observability depth**: OpenTelemetry tracing, structured logging to a central
   system, SLOs and error budgets.

Reading that pays off:

- *Designing Machine Learning Systems*, Chip Huyen
- *Reliable Machine Learning*, Chen et al. (O'Reilly, Google SRE perspective)
- Google's "MLOps: Continuous delivery and automation pipelines in machine learning" (the maturity-level paper)
- "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015), the paper that started the field
- The documentation of MLflow, DVC, Evidently, FastAPI. Reading docs end to end is an underrated skill.

---

## Glossary

| Term | Meaning |
|---|---|
| **Artifact** | Any file produced by a run: model, plot, config, report |
| **Alias** | A named, movable pointer to a model version in a registry (e.g. `champion`) |
| **Canary** | Sending a small share of live traffic to a new version before full rollout |
| **CI / CD** | Continuous Integration (auto-test every change) / Continuous Delivery or Deployment (auto-build and release) |
| **DAG** | Directed acyclic graph: stages with dependencies and no cycles; the shape of every pipeline |
| **Data contract** | An explicit, enforced schema for a dataset |
| **Drift** | A change over time in the distribution of inputs (data drift), outputs (prediction drift), or the input→output relationship (concept drift) |
| **DVC** | Data Version Control: git-like versioning for large files plus a pipeline runner |
| **Experiment tracking** | Recording params, metrics, and artifacts for every training run |
| **Feature store** | A system for defining, computing, storing, and serving features consistently for training and inference |
| **Ground truth** | The actual outcome that a prediction tried to anticipate |
| **Idempotent** | Running it twice has the same effect as running it once |
| **Lineage** | The chain from a prediction back through model version, run, code commit, and data version |
| **Lock file** | A file pinning exact versions of every dependency (`uv.lock`, `dvc.lock`) |
| **Model card** | A standard document describing a model's purpose, data, performance, and limits |
| **Model registry** | A versioned catalog of trained models with metadata and aliases |
| **PSI** | Population Stability Index: a drift metric comparing two distributions |
| **Reference data** | A sample of training-time data kept for later drift comparison |
| **Runbook** | Step-by-step operational instructions for running and fixing a service |
| **Shadow deployment** | A new model receives live traffic and its outputs are logged but not returned |
| **Signature (MLflow)** | The recorded input/output schema of a model |
| **Training-serving skew** | Differences between how features are computed in training vs in production |

---

## Appendix A: Full File Reference

Final `params.yaml`:

```yaml
data:
  raw_path: data/raw/housing.csv
  processed_dir: data/processed

prepare:
  test_size: 0.2
  random_state: 42
  target: MedHouseVal

train:
  model_type: random_forest
  model_path: models/model.joblib
  random_state: 42
  n_estimators: 300
  max_depth: 12
  min_samples_leaf: 2
  learning_rate: 0.1     # used only by gradient_boosting

evaluate:
  metrics_path: metrics/metrics.json

mlflow:
  tracking_uri: http://127.0.0.1:5000
  experiment_name: housing-price
  registered_model_name: housing-price-regressor
```

Final `pyproject.toml` dependencies section:

```toml
dependencies = [
    "pandas>=2.2",
    "numpy>=2.0",
    "scikit-learn>=1.5",
    "pyyaml>=6.0",
    "joblib>=1.4",
    "pyarrow>=17.0",
    "mlflow>=3.0",
    "pandera>=0.20",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
    "prometheus-fastapi-instrumentator>=7.0",
    "prometheus-client>=0.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
    "pre-commit>=3.8",
    "dvc>=3.50",
    "httpx>=0.27",
    "evidently>=0.7",
    "matplotlib>=3.9",
]
```

Final `Makefile`:

```makefile
.PHONY: install data features train evaluate pipeline test lint format clean \
        mlflow-ui serve docker-build docker-run up down drift

install:
	uv sync --extra dev

data:
	uv run python -m housing.data

features:
	uv run python -m housing.features

train:
	uv run python -m housing.train

evaluate:
	uv run python -m housing.evaluate

pipeline:
	uv run dvc repro

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

mlflow-ui:
	uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

serve:
	uv run uvicorn housing.api.app:app --reload --port 8000

docker-build:
	docker build --build-arg MODEL_VERSION=$$(git rev-parse --short HEAD) -t housing-api:dev .

docker-run:
	docker run --rm -p 8000:8000 housing-api:dev

up:
	docker compose up --build

down:
	docker compose down

drift:
	uv run python -m housing.drift

clean:
	rm -rf data/processed/* models/*.joblib metrics/*.json monitoring/reports
```

---

## Appendix B: Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: housing` | Package not installed in the active env | `uv sync` and always run via `uv run` |
| `make: *** missing separator` | Makefile recipe lines use spaces | Replace leading spaces with a TAB |
| `dvc repro` reruns everything every time | Outputs declared but a stage writes something else, or timestamps-only change | Check `dvc status`; ensure `outs` match actual paths exactly |
| MLflow `Connection refused` | Server not running | `make mlflow-ui` in another terminal, or set `MLFLOW_TRACKING_URI=file:./mlruns` |
| MLflow model load fails on missing columns | Signature enforcement | Good. Send all feature columns with correct names |
| pandera `SchemaErrors` on real data | Schema too strict or data genuinely bad | Read the failure table; widen a range only if the domain justifies it |
| Docker build is slow on every change | Code copied before dependency install | Follow the two-stage order in the Dockerfile |
| Container exits immediately | Model file not in image, or import error | `docker run -it housing-api:dev bash` then `ls models && python -c "import housing.api.app"` |
| `/health` 503 in Cloud Run | Model load exceeds startup probe time | Increase `--timeout`, use `--min-instances 1`, or use a smaller model |
| Prometheus target DOWN | Service name/port wrong in `prometheus.yml` | Inside compose, use the service name `api` and container port `8000` |
| Evidently `KeyError` in `run_drift` | Report dict structure differs by version | `print(json.dumps(snapshot.dict(), indent=2)[:3000])` and adjust the keys |
| Permission denied writing `monitoring/predictions.jsonl` in container | Non-root user vs host volume permissions | `chmod 777 monitoring` locally, or write logs to stdout instead |
| GitHub Actions cannot push to GHCR | Missing `packages: write` permission | Add the `permissions` block to the job |

---

*End of curriculum. Keep the project; it is your portfolio. Every interview question
about MLOps has a concrete answer in this repo.*
