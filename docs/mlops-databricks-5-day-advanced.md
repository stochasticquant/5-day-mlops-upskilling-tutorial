# MLOps on Databricks in 5 Days: Intermediate to Advanced

> The follow-up to [the beginner bootcamp](./mlops-5-day-bootcamp.md). You already know
> what experiment tracking, registries, pipelines, containers, and drift are. This week
> you implement **one complete production use case on the Databricks Data Intelligence
> Platform**: real-time transaction fraud detection, from raw event files to a monitored,
> auto-retraining, multi-environment system. Every tool choice is explained in terms of
> the problem it solves, and every day ends with a running, deployable checkpoint.

---

## Table of Contents

- [How This Week Differs From the First](#how-this-week-differs-from-the-first)
- [The Use Case](#the-use-case)
- [Target Architecture](#target-architecture)
- [Prerequisites and Setup (Day 0, ~3 hours)](#prerequisites-and-setup-day-0-3-hours)
- [Day 1: Lakehouse Foundations, Streaming Ingestion, and Declarative Pipelines](#day-1-lakehouse-foundations-streaming-ingestion-and-declarative-pipelines)
- [Day 2: Feature Engineering in Unity Catalog with Point-in-Time Correctness](#day-2-feature-engineering-in-unity-catalog-with-point-in-time-correctness)
- [Day 3: Training at Scale, Tuning, Evaluation, and Governed Promotion](#day-3-training-at-scale-tuning-evaluation-and-governed-promotion)
- [Day 4: Batch and Real-Time Serving with Automatic Feature Lookup](#day-4-batch-and-real-time-serving-with-automatic-feature-lookup)
- [Day 5: Monitoring, Retraining Loops, CI/CD with Asset Bundles, and Governance](#day-5-monitoring-retraining-loops-cicd-with-asset-bundles-and-governance)
- [Capstone Checklist](#capstone-checklist)
- [Mapping Databricks to What You Learned in Week 1](#mapping-databricks-to-what-you-learned-in-week-1)
- [What to Learn Next](#what-to-learn-next)
- [Glossary](#glossary)
- [Appendix A: Full Bundle Reference](#appendix-a-full-bundle-reference)
- [Appendix B: Troubleshooting](#appendix-b-troubleshooting)

---

## How This Week Differs From the First

Last week you assembled a platform from parts: DVC, MLflow, FastAPI, Docker, Prometheus,
Evidently, GitHub Actions. That taught you what each part *is*. This week you use a
platform where those parts are integrated and governed by a single catalog, and the
engineering problems shift:

| Week 1 problem | Week 2 problem |
|---|---|
| Make one model reproducible | Make features reusable across models and teams with no leakage |
| Serve one model from a container | Serve with automatic feature lookup at low latency, with traffic splitting |
| Detect drift with a script | Monitor every inference, join delayed labels, alert, and retrain automatically |
| CI for one repo on a laptop | Promote the same bundle through dev, staging, and prod with service principals |
| Data fits in pandas | Data arrives continuously; compute is distributed; cost is a design constraint |

Expect to read more documentation this week. Databricks ships features fast and renames
things (Delta Live Tables became Lakeflow Declarative Pipelines; Workflows became
Lakeflow Jobs; online tables are moving to Lakebase-backed synced tables). Where an API
is in flux this guide says so and gives you the stable concept plus the current call.
**When a call fails with "unknown argument," check the docs page for that feature before
anything else.** That habit is part of being advanced.

**Daily rhythm** is the same as week one: concepts (~2 h), build (~4 h), exercises
(~1.5 h), checkpoint (~30 min). Build blocks are longer this week; if you fall behind,
skip exercises, never concepts.

---

## The Use Case

**Company:** a payments provider processing card transactions.
**Goal:** flag fraudulent transactions in real time so they can be held for review, while
keeping false positives low enough that legitimate customers are not blocked.

**Business constraints that shape the design:**

- A decision is needed in **under 100 ms** at authorization time (online serving).
- Fraud is rare (**about 1.5%** of transactions), so accuracy is meaningless; we optimize
  precision-recall and a cost function.
- Ground truth arrives **late**: chargebacks take days to weeks. Monitoring must handle
  delayed labels.
- Features depend on **customer history** (spend in the last hour, day, week). Those must
  be computed identically for training (historical, point-in-time correct) and serving
  (latest values, milliseconds).
- A nightly **batch score** of all transactions feeds analyst dashboards and the labeling
  team's queue.
- Regulators want **lineage**: for any decision, which model, which features, which data.

Every one of these constraints maps to a Databricks capability you will use.

**Data you will generate** (synthetic, so there are no downloads or licenses):

- `transactions`: ~1M events over 60 days, JSON files landing in a volume every "hour"
- `customers`: 20k profiles with home country, signup date, baseline spend
- `labels`: fraud outcomes per transaction, arriving with a 3 to 14 day delay

---

## Target Architecture

```
  Raw JSON files            Lakeflow Declarative Pipeline             Feature Engineering in UC
 (UC Volume)                 (Auto Loader + expectations)
 ┌──────────┐   ┌────────┐   ┌────────┐   ┌───────────────┐   ┌──────────────────────────┐
 │ txn/*.json├──▶│ bronze ├──▶│ silver ├──▶│ gold (marts)  │   │ customer_txn_features    │
 │ lbl/*.json│   │        │   │        │   │               │   │   (time-series, PIT)     │
 └──────────┘   └────────┘   └───┬────┘   └───────────────┘   │ customer_profile_features│
                                 │                             │ feature functions (UDFs) │
                                 └────────────────────────────▶└───────────┬──────────────┘
                                                                           │
                       ┌───────────────────────────────────────────────────┤
                       ▼                                                   ▼
            Lakeflow Job: training                                 Online feature store
   ┌──────────────┬──────────┬───────────────────┐                (synced / online tables)
   │ build_features│ train   │ compare_and_promote│                         │
   └──────────────┴────┬─────┴───────────────────┘                         │
                       │  MLflow (UC models, aliases)                      │
                       ▼                                                   ▼
          ┌──────────────────────┐                        ┌───────────────────────────┐
          │ main.fraud.          │  champion alias        │ Model Serving endpoint    │
          │ fraud_classifier v3  ├───────────────────────▶│ auto feature lookup       │
          └──────────────────────┘                        │ AI Gateway inference table│
                       │                                  └─────────────┬─────────────┘
                       ▼                                                │
          Lakeflow Job: batch scoring (nightly)                         ▼
                       │                                  Lakehouse Monitoring (inference
                       ▼                                  profile + delayed labels)
              gold.transaction_scores                                   │
                                                                        ▼
                                                      SQL alert ──▶ retraining job trigger

  Everything above is defined in a Databricks Asset Bundle, deployed to dev/staging/prod
  by GitHub Actions under service principals.
```

---

## Prerequisites and Setup (Day 0, ~3 hours)

### What you should already have

- Completed week one (or equivalent): you can explain MLflow runs vs registered models,
  DAG pipelines, Docker, drift, and CI.
- Comfortable PySpark or willing to learn it on the fly (DataFrame API, `groupBy`,
  `join`, window functions). SQL fluency.
- A GitHub account and the repo from week one.

### Get a Databricks workspace

Options, in order of recommendation:

1. **Databricks Free Edition** (https://www.databricks.com/learn/free-edition). Free
   forever, serverless only, Unity Catalog enabled. Covers roughly 80% of this week.
   Some features (online tables, model serving at scale, Lakehouse Monitoring, multiple
   workspaces) are limited or unavailable. Where that happens, this guide shows what to
   do and marks the section with **[Full workspace]**.
2. **14-day trial** of a full workspace on AWS, Azure, or GCP. Everything works. You pay
   the cloud provider for compute; keep clusters small and use serverless.
3. **Your employer's workspace** if you have one with Unity Catalog and permission to
   create catalogs/schemas, serving endpoints, and jobs.

Record your workspace URL (looks like `https://dbc-xxxx.cloud.databricks.com` or
`https://adb-xxxx.azuredatabricks.net`).

### Install and authenticate the CLI

The Databricks CLI (v0.2xx+, the Go version, not the old Python one) is how you deploy
bundles and automate everything.

```bash
# macOS
brew tap databricks/tap && brew install databricks
# Linux
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

databricks --version      # must be >= 0.230
```

Authenticate with OAuth user-to-machine (opens a browser):

```bash
databricks auth login --host https://YOUR-WORKSPACE-URL --profile fraud
databricks current-user me --profile fraud     # prints your user JSON
```

The profile is stored in `~/.databrickscfg`. Set `export DATABRICKS_CONFIG_PROFILE=fraud`
in your shell so you do not need `--profile` on every command.

### Unity Catalog: the one concept to internalize before starting

**Unity Catalog (UC)** is the governance layer for *everything*: tables, volumes (files),
functions, models, and feature tables all live in a three-level namespace
`catalog.schema.object`. Permissions, lineage, audit, and discovery are uniform across
all of them. In week one, your data was in DVC, your model in MLflow, your features in a
Python function; three systems, three permission models. Here it is one.

Create your working namespace. Use the SQL editor in the workspace or the CLI:

```sql
-- Use an existing catalog (Free Edition: `workspace`; trial/enterprise: `main` or create one)
CREATE CATALOG IF NOT EXISTS main;
CREATE SCHEMA IF NOT EXISTS main.fraud_dev COMMENT 'Fraud detection, dev environment';
CREATE VOLUME IF NOT EXISTS main.fraud_dev.raw COMMENT 'Landing zone for raw event files';
```

> If `CREATE CATALOG` fails with a permissions error, use the catalog your admin gives
> you and substitute it everywhere this guide says `main`. **Throughout, we parameterize
> catalog and schema so that dev, staging, and prod are the same code pointed at
> different namespaces.**

### Local project setup

Create a fresh repo (or a folder in last week's repo) called `fraud-databricks`:

```bash
mkdir fraud-databricks && cd fraud-databricks && git init
uv init --no-workspace --python 3.12
uv add "databricks-sdk>=0.40" "databricks-feature-engineering>=0.8" "mlflow>=3.0" \
       "pandas>=2.2" "scikit-learn>=1.5" "optuna>=3.6" "pyyaml>=6.0"
uv add --dev "pytest>=8" "ruff>=0.6" "databricks-connect>=16.0"
```

> **Databricks Connect** lets your local Python run Spark code on a Databricks cluster or
> serverless compute. It makes local tests and IDE development possible. Match its major
> version to your workspace runtime (16.x for DBR 16, 17.x for DBR 17). If you only use
> notebooks, you can skip it this week; the guide shows both paths.

Create the layout. It is a **Databricks Asset Bundle** from day one, because retrofitting
a bundle later is painful and this is how production projects are structured:

```
fraud-databricks/
├── databricks.yml              # bundle definition: name, targets (dev/staging/prod), variables
├── resources/                  # one YAML per deployable thing
│   ├── pipeline.yml            #   Lakeflow Declarative Pipeline (bronze/silver/gold)
│   ├── training_job.yml        #   training + promotion job
│   ├── scoring_job.yml         #   nightly batch scoring job
│   ├── monitoring_job.yml      #   label join + monitor refresh + retrain trigger
│   └── serving.yml             #   model serving endpoint
├── src/
│   ├── fraud/                  # pure Python package: feature logic, evaluation, utilities
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── features.py
│   │   ├── evaluation.py
│   │   └── promotion.py
│   ├── pipelines/
│   │   └── fraud_pipeline.py   # declarative pipeline source
│   └── notebooks/              # thin notebooks that call the package
│       ├── 00_generate_data.py
│       ├── 01_explore.py
│       ├── 02_build_features.py
│       ├── 03_train.py
│       ├── 04_promote.py
│       ├── 05_batch_score.py
│       ├── 06_deploy_serving.py
│       ├── 07_unpack_inference.py
│       └── 08_monitor.py
├── tests/
├── .github/workflows/
├── pyproject.toml
└── README.md
```

```bash
mkdir -p resources src/fraud src/pipelines src/notebooks tests .github/workflows
touch src/fraud/__init__.py
```

### The bundle skeleton

`databricks.yml`:

```yaml
bundle:
  name: fraud-mlops

include:
  - resources/*.yml

variables:
  catalog:
    description: Unity Catalog catalog for all objects
    default: main
  schema:
    description: Schema (environment-specific)
    default: fraud_dev
  model_name:
    default: fraud_classifier
  endpoint_name:
    default: fraud-classifier

targets:
  dev:
    mode: development          # resources get a [dev yourname] prefix; jobs are paused
    default: true
    workspace:
      host: https://YOUR-WORKSPACE-URL

  staging:
    mode: production
    workspace:
      host: https://YOUR-WORKSPACE-URL
      root_path: /Workspace/Shared/.bundle/${bundle.name}/${bundle.target}
    variables:
      schema: fraud_staging
    run_as:
      service_principal_name: REPLACE_WITH_SP_APPLICATION_ID   # Day 5

  prod:
    mode: production
    workspace:
      host: https://YOUR-WORKSPACE-URL
      root_path: /Workspace/Shared/.bundle/${bundle.name}/${bundle.target}
    variables:
      schema: fraud
    run_as:
      service_principal_name: REPLACE_WITH_SP_APPLICATION_ID
```

> **Development mode** does three useful things: prefixes every resource with
> `[dev your.name]` so teammates do not collide, pauses schedules so dev jobs do not run
> on their own, and deploys to your user folder. **Production mode** removes the prefix,
> enforces that schedules are set and `run_as` is a service principal, and deploys to a
> shared path.

Validate (it will complain about no resources yet; that is fine):

```bash
databricks bundle validate
```

### Notebook-to-package configuration pattern

Every notebook starts with the same cell, so the same notebook works in every environment:

`src/fraud/config.py`:

```python
"""Resolve the Unity Catalog namespace for this run.

Notebooks receive catalog/schema as job parameters (widgets). When run interactively
without widgets, they fall back to dev defaults. Nothing is hard-coded anywhere else.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Namespace:
    catalog: str
    schema: str

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def raw_volume(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/raw"


def from_widgets(dbutils, default_catalog="main", default_schema="fraud_dev") -> Namespace:
    try:
        dbutils.widgets.text("catalog", default_catalog)
        dbutils.widgets.text("schema", default_schema)
        return Namespace(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
    except Exception:
        return Namespace(default_catalog, default_schema)
```

Each notebook begins:

```python
# Databricks notebook source
# MAGIC %pip install -q optuna>=3.6
# COMMAND ----------
import sys
sys.path.append("../")              # bundle deploys src/ alongside notebooks
from fraud.config import from_widgets
ns = from_widgets(dbutils)
print(ns)
```

> Notebooks in a bundle are plain `.py` files with `# Databricks notebook source` on
> line one and `# COMMAND ----------` separating cells. The workspace renders them as
> notebooks; git sees text. Edit them locally in your IDE or in the workspace via a
> **Git folder** (Repos). Set up the Git folder now: Workspace → your user → Create →
> Git folder → paste your repo URL. Commit from either side; pull on the other.

Commit the skeleton. You are ready.

---

## Day 1: Lakehouse Foundations, Streaming Ingestion, and Declarative Pipelines

### Learning objectives

- Explain Delta Lake's guarantees and use time travel, MERGE, and liquid clustering.
- Land raw files in a Unity Catalog volume and ingest them incrementally with Auto Loader.
- Build a bronze/silver/gold medallion pipeline as a Lakeflow Declarative Pipeline with
  data quality expectations.
- Deploy the pipeline from your bundle and run it on a schedule.

### 1.1 Concepts: Delta Lake and the lakehouse

A **lakehouse** stores data as open files (Parquet) in cloud object storage but adds the
properties of a database: ACID transactions, schema enforcement, versioning, and indexing.
**Delta Lake** is the table format that provides this. Every Databricks table you create
is a Delta table unless you say otherwise.

What you get, and why ML cares:

| Property | Mechanism | ML consequence |
|---|---|---|
| **ACID transactions** | A JSON transaction log (`_delta_log/`) records every commit | A training job never reads a half-written table |
| **Time travel** | Every version is retained until `VACUUM` | `SELECT * FROM t VERSION AS OF 42` reproduces the exact training snapshot. This replaces DVC. |
| **Schema enforcement and evolution** | Writes with mismatched schema fail unless evolution is enabled | Upstream column renames break loudly, not silently |
| **MERGE (upsert)** | `MERGE INTO target USING source ON key WHEN MATCHED ... WHEN NOT MATCHED ...` | Feature tables and label tables update incrementally |
| **Change Data Feed** | Row-level change log per version | Incremental feature recomputation and audit |
| **Liquid clustering** | Adaptive data layout by chosen columns | Fast point-in-time lookups by customer and timestamp |

Try it with a scratch table in the SQL editor to feel the semantics:

```sql
USE CATALOG main; USE SCHEMA fraud_dev;

CREATE OR REPLACE TABLE scratch (id INT, v STRING) CLUSTER BY (id);
INSERT INTO scratch VALUES (1, 'a'), (2, 'b');
INSERT INTO scratch VALUES (3, 'c');
DESCRIBE HISTORY scratch;                     -- three versions: create, insert, insert
SELECT * FROM scratch VERSION AS OF 1;        -- only ids 1,2
MERGE INTO scratch t USING (SELECT 2 AS id, 'B' AS v) s ON t.id = s.id
  WHEN MATCHED THEN UPDATE SET v = s.v
  WHEN NOT MATCHED THEN INSERT *;
SELECT * FROM scratch;                        -- id 2 is now 'B'
RESTORE TABLE scratch TO VERSION AS OF 2;     -- undo the merge
DROP TABLE scratch;
```

**For ML reproducibility, record the Delta version (or a timestamp) of every input table
in the MLflow run.** You will do this on Day 3. It is the lakehouse equivalent of the
DVC hash from week one, with no extra tool.

### 1.2 Build: Generate the synthetic data feed

A generator that writes JSON files into the raw volume in hourly batches, so Auto Loader
has something incremental to ingest and you can simulate "new data arrived" at will.

`src/fraud/datagen.py`:

```python
"""Synthetic card-transaction generator with planted fraud patterns.

Fraud is planted with structure the model can learn (and that drifts later):
- bursts of several transactions within minutes
- amounts far above the customer's baseline
- transactions from a country other than the customer's home country
- a higher rate on 'web' device type at night
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTRIES = ["US", "GB", "DE", "FR", "ES", "NL", "BR", "IN"]
CATEGORIES = ["grocery", "fuel", "restaurant", "electronics", "travel", "fashion", "gaming", "other"]
DEVICES = ["pos", "web", "mobile"]


def make_customers(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"c_{i:06d}" for i in range(n)],
            "home_country": rng.choice(COUNTRIES, size=n, p=[0.35, 0.15, 0.12, 0.1, 0.08, 0.07, 0.07, 0.06]),
            "signup_date": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
            "baseline_amount": np.round(rng.lognormal(mean=3.3, sigma=0.6, size=n), 2),
        }
    )


def make_transactions(
    customers: pd.DataFrame,
    start: pd.Timestamp,
    hours: int,
    txn_per_hour: int,
    fraud_rate: float = 0.015,
    seed: int = 0,
    drift: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (transactions, labels). Labels are separate because they arrive later."""
    rng = np.random.default_rng(seed)
    n = hours * txn_per_hour
    cust = customers.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    ts = start + pd.to_timedelta(rng.uniform(0, hours * 3600, n), unit="s")
    ts = pd.Series(ts).sort_values().reset_index(drop=True)

    amount = np.round(cust["baseline_amount"].values * rng.lognormal(0, 0.5, n), 2)
    country = cust["home_country"].values.copy()
    device = rng.choice(DEVICES, size=n, p=[0.5, 0.3, 0.2])
    category = rng.choice(CATEGORIES, size=n)

    is_fraud = rng.random(n) < fraud_rate
    # Planted patterns
    amount[is_fraud] = np.round(amount[is_fraud] * rng.uniform(3, 12, is_fraud.sum()), 2)
    foreign = is_fraud & (rng.random(n) < 0.6)
    country[foreign] = rng.choice(COUNTRIES, size=foreign.sum())
    device[is_fraud & (rng.random(n) < 0.7)] = "web"
    category[is_fraud & (rng.random(n) < 0.5)] = rng.choice(["electronics", "gaming"], size=(is_fraud & (rng.random(n) < 0.5)).sum())

    if drift:
        # The world changes: fraudsters move to mobile and smaller amounts.
        device[is_fraud] = "mobile"
        amount[is_fraud] = np.round(amount[is_fraud] * 0.3, 2)

    txn = pd.DataFrame(
        {
            "transaction_id": [f"t_{seed}_{i:08d}" for i in range(n)],
            "customer_id": cust["customer_id"].values,
            "event_ts": ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z",
            "amount": amount,
            "currency": rng.choice(["USD", "EUR", "GBP"], size=n, p=[0.6, 0.3, 0.1]),
            "country": country,
            "merchant_id": [f"m_{m:05d}" for m in rng.integers(0, 3000, n)],
            "merchant_category": category,
            "device_type": device,
        }
    )
    # A little dirt, like real data
    dirty = rng.random(n) < 0.002
    txn.loc[dirty, "amount"] = -1.0
    txn.loc[rng.random(n) < 0.001, "customer_id"] = None

    delay_days = rng.integers(3, 15, n)
    labels = pd.DataFrame(
        {
            "transaction_id": txn["transaction_id"],
            "is_fraud": is_fraud.astype(int),
            "labeled_at": (ts + pd.to_timedelta(delay_days, unit="D")).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    return txn, labels
```

`src/notebooks/00_generate_data.py`:

```python
# Databricks notebook source
# MAGIC %md # Generate synthetic transaction feed
# MAGIC Writes customers to a Delta table and transactions/labels as JSON files into the raw volume.
# MAGIC Run once with `days=60`; later run with `days=1` to simulate new arrivals (and `drift=true` for Day 5).

# COMMAND ----------
import sys
sys.path.append("../")
import pandas as pd
from fraud.config import from_widgets
from fraud.datagen import make_customers, make_transactions

ns = from_widgets(dbutils)
dbutils.widgets.text("days", "60")
dbutils.widgets.text("start", "2026-07-01")
dbutils.widgets.text("drift", "false")
days = int(dbutils.widgets.get("days"))
start = pd.Timestamp(dbutils.widgets.get("start"))
drift = dbutils.widgets.get("drift").lower() == "true"

# COMMAND ----------
customers = make_customers(20_000)
spark.createDataFrame(customers).write.mode("overwrite").saveAsTable(ns.table("customers"))
print("customers written")

# COMMAND ----------
# One JSON file per simulated day, so Auto Loader sees many files.
for d in range(days):
    day_start = start + pd.Timedelta(days=d)
    txn, labels = make_transactions(customers, day_start, hours=24, txn_per_hour=700, seed=int(day_start.strftime("%Y%m%d")), drift=drift)
    stamp = day_start.strftime("%Y%m%d")
    (spark.createDataFrame(txn).coalesce(1).write.mode("append").json(f"{ns.raw_volume}/transactions/dt={stamp}"))
    (spark.createDataFrame(labels).coalesce(1).write.mode("append").json(f"{ns.raw_volume}/labels/dt={stamp}"))
    if d % 10 == 0:
        print(f"day {d} written")
print("done")
```

Run it in the workspace (open the notebook from your Git folder, attach to serverless,
run all). Sixty days at 700/hour is about 1M transactions. Inspect the volume in
Catalog Explorer: `main.fraud_dev.raw` → `transactions/dt=20260701/`.

### 1.3 Concepts: Auto Loader and the medallion architecture

**Auto Loader** (`cloudFiles` source) incrementally ingests new files from a storage
location. It tracks which files it has already processed (in a checkpoint), infers and
evolves schema, and scales to millions of files. It is the standard way to turn a landing
zone into a table. You never write "list files, diff against processed list" code again.

**Medallion architecture** is a naming convention for data quality tiers:

- **Bronze**: raw, as-landed, append-only, with ingestion metadata. Never modified. Your
  audit trail and your replay source.
- **Silver**: cleaned, deduplicated, typed, validated. One row per real-world event. What
  features and models consume.
- **Gold**: aggregated or business-shaped marts: daily merchant stats, customer summaries,
  the scoring output table.

### 1.4 Concepts: Lakeflow Declarative Pipelines

A declarative pipeline lets you *define* tables as functions of other tables and let the
engine figure out the DAG, incremental processing, retries, and orchestration. You wrote
a DAG by hand in `dvc.yaml` last week; here you write the table definitions and the DAG
is inferred from the reads.

Three table kinds:

- **Streaming table**: processes each input row exactly once; append-heavy; backed by
  Structured Streaming. Use for bronze and silver.
- **Materialized view**: recomputed (incrementally where possible) from its sources on
  each update. Use for gold aggregates.
- **View**: not persisted; intermediate logic.

**Expectations** are declarative data quality rules attached to a table:

- `expect(name, condition)`: record violations, keep the row
- `expect_or_drop`: record and drop violating rows
- `expect_or_fail`: stop the pipeline

Metrics for every expectation are stored in the pipeline's event log; you can query and
alert on them. This is pandera from week one, but built into the pipeline and tracked
over time.

> **Naming.** This feature was called Delta Live Tables (DLT) until 2025. The Python
> module was `dlt`; newer runtimes expose it as `pyspark.pipelines`. Both work on current
> runtimes; this guide uses the newer import with the old one in a comment.

### 1.5 Build: The pipeline

`src/pipelines/fraud_pipeline.py`:

```python
"""Bronze -> silver -> gold for transactions, labels, and customers.

Catalog and schema come from pipeline configuration (set in resources/pipeline.yml),
so this file is environment-agnostic.
"""

from pyspark import pipelines as dp          # older runtimes: import dlt as dp
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

RAW = spark.conf.get("fraud.raw_volume")      # e.g. /Volumes/main/fraud_dev/raw

TXN_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType()),
        StructField("customer_id", StringType()),
        StructField("event_ts", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("country", StringType()),
        StructField("merchant_id", StringType()),
        StructField("merchant_category", StringType()),
        StructField("device_type", StringType()),
    ]
)


# ---------------------------------------------------------------- bronze
@dp.table(
    name="bronze_transactions",
    comment="Raw transaction events exactly as landed, plus ingestion metadata.",
    table_properties={"quality": "bronze"},
)
def bronze_transactions():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{RAW}/_schemas/transactions")
        .schema(TXN_SCHEMA)                      # explicit schema: fail loudly on surprises
        .load(f"{RAW}/transactions")
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.col("_metadata.file_path"))
    )


@dp.table(name="bronze_labels", comment="Fraud outcomes as reported, with reporting delay.")
def bronze_labels():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaLocation", f"{RAW}/_schemas/labels")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(f"{RAW}/labels")
        .withColumn("_ingested_at", F.current_timestamp())
    )


# ---------------------------------------------------------------- silver
@dp.table(
    name="silver_transactions",
    comment="One row per transaction: typed, validated, deduplicated.",
    table_properties={"quality": "silver", "delta.enableChangeDataFeed": "true"},
    cluster_by=["customer_id", "event_ts"],      # liquid clustering for PIT lookups
)
@dp.expect_or_drop("valid_transaction_id", "transaction_id IS NOT NULL")
@dp.expect_or_drop("valid_customer_id", "customer_id IS NOT NULL")
@dp.expect_or_drop("positive_amount", "amount > 0 AND amount < 100000")
@dp.expect("known_currency", "currency IN ('USD','EUR','GBP')")
@dp.expect_or_fail("timestamp_parsed", "event_ts IS NOT NULL")
def silver_transactions():
    return (
        spark.readStream.table("bronze_transactions")     # older runtimes: "LIVE.bronze_transactions"
        .withColumn("event_ts", F.to_timestamp("event_ts"))
        .withWatermark("event_ts", "2 hours")
        .dropDuplicates(["transaction_id"])
        .withColumn("event_date", F.to_date("event_ts"))
        .withColumn("hour_of_day", F.hour("event_ts"))
        .drop("_source_file")
    )


@dp.table(name="silver_labels", comment="Latest known label per transaction.")
@dp.expect_or_drop("valid_label", "is_fraud IN (0, 1)")
def silver_labels():
    return (
        spark.readStream.table("bronze_labels")
        .withColumn("labeled_at", F.to_timestamp("labeled_at"))
        .withWatermark("labeled_at", "1 day")
        .dropDuplicates(["transaction_id"])
    )


# ---------------------------------------------------------------- gold
@dp.materialized_view(
    name="gold_daily_merchant_stats",
    comment="Daily volume and amount per merchant category; feeds analyst dashboards.",
)
def gold_daily_merchant_stats():
    return (
        spark.read.table("silver_transactions")
        .groupBy("event_date", "merchant_category")
        .agg(
            F.count("*").alias("txn_count"),
            F.sum("amount").alias("amount_sum"),
            F.avg("amount").alias("amount_avg"),
            F.countDistinct("customer_id").alias("customers"),
        )
    )
```

`resources/pipeline.yml`:

```yaml
resources:
  pipelines:
    fraud_pipeline:
      name: "fraud-medallion"
      catalog: ${var.catalog}
      schema: ${var.schema}
      serverless: true
      continuous: false                  # triggered: runs when the job/schedule says so
      development: true                  # faster retries, keeps cluster warm in dev
      libraries:
        - file:
            path: ../src/pipelines/fraud_pipeline.py
      configuration:
        fraud.raw_volume: /Volumes/${var.catalog}/${var.schema}/raw
```

Deploy and run:

```bash
databricks bundle validate
databricks bundle deploy
databricks bundle run fraud_pipeline
```

The CLI prints a link to the pipeline UI. Watch the DAG execute: bronze tables stream
in files, silver applies expectations, gold aggregates. Click `silver_transactions` to
see the expectation metrics: a few hundred rows dropped for negative amount or null
customer, exactly the dirt the generator planted.

Query the results:

```sql
SELECT COUNT(*) FROM main.fraud_dev.silver_transactions;
SELECT * FROM main.fraud_dev.gold_daily_merchant_stats ORDER BY event_date DESC LIMIT 20;
```

**Incremental proof.** Run the generator for one more day
(`days=1, start=2026-08-30`), then `databricks bundle run fraud_pipeline` again. Only
the new file is processed; the silver count grows by about 16,800. Check
`DESCRIBE HISTORY main.fraud_dev.silver_transactions` to see the new version.

### 1.6 Build: Query the expectation metrics

The pipeline event log is a table you can query. This is how you alert on data quality.

```sql
-- Event log location is shown in pipeline settings; UC pipelines publish it as a table:
SELECT
  timestamp,
  details:flow_progress.data_quality.expectations AS expectations
FROM event_log(TABLE(main.fraud_dev.silver_transactions))
WHERE event_type = 'flow_progress'
  AND details:flow_progress.data_quality.expectations IS NOT NULL
ORDER BY timestamp DESC LIMIT 5;
```

Each row lists, per expectation, `passed_records` and `failed_records`. On Day 5 you
will put a SQL alert on `failed_records / passed_records`.

### 1.7 Exercises

1. **Schema evolution drill.** Add a `channel` column to the generator, write one new
   day, run the pipeline. Does bronze fail (explicit schema) or evolve? Change the bronze
   definition to `inferColumnTypes` with `cloudFiles.schemaEvolutionMode: addNewColumns`
   and observe the difference. Decide which policy you want in production and write
   the reasoning in `README.md`.
2. **Quarantine instead of drop.** Replace `expect_or_drop` on silver with a pattern
   that routes bad rows to a `silver_transactions_quarantine` table (hint: two tables
   from the same source with complementary filters). Why is this better for a fraud
   team?
3. **Customers as SCD type 2.** Customers' `home_country` can change. Use
   `dp.create_auto_cdc_flow` (formerly `apply_changes`) with `stored_as_scd_type=2` to
   maintain a history table from a `customer_updates` feed. This is what lets you
   compute "home country *at the time of* the transaction."
4. **Time travel for reproducibility.** Record the Delta version of
   `silver_transactions` right now. Insert another day. Query the table `VERSION AS OF`
   the recorded version and confirm the count matches the earlier value.
5. **Cost awareness.** Open the pipeline's run details and note DBUs consumed. Switch
   `development: false` and rerun; compare. Read what "enhanced autoscaling" does.

### 1.8 Checkpoint questions

1. What does the Delta transaction log give an ML training job that a folder of Parquet files does not?
2. Why keep bronze immutable even though silver is what everyone uses?
3. How does Auto Loader know which files it already processed, and what happens if you delete its checkpoint?
4. What is the difference between a streaming table and a materialized view, and why is silver one and gold the other?
5. Where do expectation metrics live, and how would you alert if drop rate exceeded 1%?
6. Explain what `withWatermark` does in the dedup step and what breaks without it.

---

## Day 2: Feature Engineering in Unity Catalog with Point-in-Time Correctness

### Learning objectives

- Explain point-in-time (PIT) correctness and label leakage, and why they are the
  central problem in feature engineering for event data.
- Build time-series feature tables with Spark window functions.
- Register feature tables in Unity Catalog and use lookups with timestamp keys.
- Write on-demand feature functions as UC Python UDFs usable in training and serving.
- Assemble a leak-free training set with full lineage.

### 2.1 Concepts: The feature store problem, properly

Last week `build_features` was one function imported by training and serving. That works
when features depend only on the request payload. It breaks the moment a feature depends
on **history**: "how many transactions did this customer make in the last 24 hours?"

At serving time that number must be looked up from a store that is kept current. At
training time it must be computed *as it would have been known at that moment*, not as
it is now. Getting the second part wrong is **label leakage** and it is the most common
way a fraud model reports 0.99 AUC offline and fails in production.

**Point-in-time correctness**: for a training example at time T, every feature value
must be computed using only data with timestamp < T. A **time-series feature table** has
a primary key (customer_id) plus a timestamp key (feature_ts). A PIT lookup for
(customer_id, T) returns the row with the largest feature_ts ≤ T. Databricks Feature
Engineering does this join for you (an "AS OF" join), which is tedious and error-prone
to write by hand at scale.

Three kinds of features, by *where* they can be computed:

| Kind | Example | Training | Serving |
|---|---|---|---|
| **Batch / precomputed** | Customer's 30-day average spend | PIT lookup from feature table | Lookup from online store |
| **Streaming** | Transactions in the last 10 minutes | PIT lookup (table updated by a stream) | Lookup from online store, updated continuously |
| **On-demand** | Ratio of this amount to the 30-day average; is this country foreign | Computed by a function during training set creation | Same function executed inside the serving endpoint |

The **Feature Engineering in Unity Catalog** client handles all three uniformly: lookups
with PIT joins, functions attached to the training set, and packaging all of it *into the
model* so the serving endpoint knows what to look up and compute. You send only the
raw request; the endpoint does the rest. That is the end of training-serving skew.

### 2.2 Build: Shared feature logic as a package

Window functions over customer history. Read the comments; the `rangeBetween` bounds are
where leakage is prevented.

`src/fraud/features.py`:

```python
"""Customer behavior features computed per transaction event.

Each output row is keyed by (customer_id, feature_ts) where feature_ts is the event
time. Window bounds end at -1 second so the CURRENT transaction is never included:
the row at time T describes the customer's state just BEFORE T. A PIT lookup at T
therefore returns leak-free values.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

HOUR, DAY = 3600, 86400


def customer_txn_features(txn: DataFrame) -> DataFrame:
    t = txn.withColumn("_ts", F.col("event_ts").cast("long"))
    by_cust = Window.partitionBy("customer_id").orderBy("_ts")

    w1h = by_cust.rangeBetween(-1 * HOUR, -1)
    w24h = by_cust.rangeBetween(-1 * DAY, -1)
    w7d = by_cust.rangeBetween(-7 * DAY, -1)
    w30d = by_cust.rangeBetween(-30 * DAY, -1)

    out = (
        t.withColumn("txn_count_1h", F.count("*").over(w1h))
        .withColumn("txn_count_24h", F.count("*").over(w24h))
        .withColumn("amount_sum_24h", F.coalesce(F.sum("amount").over(w24h), F.lit(0.0)))
        .withColumn("distinct_merchants_24h", F.size(F.collect_set("merchant_id").over(w24h)))
        .withColumn("distinct_countries_7d", F.size(F.collect_set("country").over(w7d)))
        .withColumn("txn_count_7d", F.count("*").over(w7d))
        .withColumn("avg_amount_30d", F.avg("amount").over(w30d))
        .withColumn("max_amount_30d", F.max("amount").over(w30d))
        .withColumn(
            "seconds_since_last_txn",
            F.col("_ts") - F.lag("_ts").over(by_cust),
        )
        .select(
            "customer_id",
            F.col("event_ts").alias("feature_ts"),
            "txn_count_1h",
            "txn_count_24h",
            "amount_sum_24h",
            "distinct_merchants_24h",
            "distinct_countries_7d",
            "txn_count_7d",
            "avg_amount_30d",
            "max_amount_30d",
            "seconds_since_last_txn",
        )
        # Primary key must be unique; two events in the same millisecond collapse to one.
        .dropDuplicates(["customer_id", "feature_ts"])
    )
    return out


def customer_profile_features(customers: DataFrame) -> DataFrame:
    """Slowly changing attributes keyed by customer only (no timestamp key)."""
    return customers.select(
        "customer_id",
        "home_country",
        F.col("signup_date").cast("date").alias("signup_date"),
        "baseline_amount",
    )
```

**Unit-test it locally** with Databricks Connect (or on a tiny Spark session). This is
the kind of test that catches leakage before it costs you a quarter.

`tests/test_features.py`:

```python
import pytest
from pyspark.sql import SparkSession

from fraud.features import customer_txn_features


@pytest.fixture(scope="session")
def spark():
    # Databricks Connect picks up DATABRICKS_CONFIG_PROFILE; falls back to local Spark if installed.
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.serverless().getOrCreate()
    except Exception:
        return SparkSession.builder.master("local[2]").getOrCreate()


def test_current_transaction_is_excluded_from_its_own_window(spark):
    rows = [
        ("c1", "2026-01-01T10:00:00", 10.0, "m1", "US"),
        ("c1", "2026-01-01T10:30:00", 20.0, "m2", "US"),
        ("c1", "2026-01-01T12:00:00", 30.0, "m3", "US"),
    ]
    df = spark.createDataFrame(rows, ["customer_id", "event_ts", "amount", "merchant_id", "country"])
    df = df.withColumn("event_ts", df.event_ts.cast("timestamp"))

    out = {r.feature_ts.isoformat(): r for r in customer_txn_features(df).collect()}
    assert out["2026-01-01T10:00:00"].txn_count_1h == 0          # first event: nothing before it
    assert out["2026-01-01T10:30:00"].txn_count_1h == 1
    assert out["2026-01-01T10:30:00"].amount_sum_24h == 10.0     # excludes its own 20.0
    assert out["2026-01-01T12:00:00"].txn_count_1h == 0          # 10:30 is 90 min earlier
    assert out["2026-01-01T12:00:00"].txn_count_24h == 2
```

```bash
uv run pytest tests/test_features.py -q
```

### 2.3 Build: Create and populate feature tables

`src/notebooks/02_build_features.py`:

```python
# Databricks notebook source
# MAGIC %md # Build and publish feature tables to Unity Catalog

# COMMAND ----------
import sys
sys.path.append("../")
from databricks.feature_engineering import FeatureEngineeringClient
from fraud.config import from_widgets
from fraud.features import customer_profile_features, customer_txn_features

ns = from_widgets(dbutils)
fe = FeatureEngineeringClient()

TXN_FEATURES = ns.table("customer_txn_features")
PROFILE_FEATURES = ns.table("customer_profile_features")

# COMMAND ----------
txn = spark.table(ns.table("silver_transactions"))
txn_feats = customer_txn_features(txn)

if not spark.catalog.tableExists(TXN_FEATURES):
    fe.create_table(
        name=TXN_FEATURES,
        primary_keys=["customer_id", "feature_ts"],
        timeseries_columns=["feature_ts"],          # enables point-in-time lookups
        df=txn_feats,
        description="Customer transaction-behavior features as of just before each event. "
                    "Windows exclude the current event (no leakage).",
        tags={"domain": "fraud", "owner": "ml-platform"},
    )
else:
    fe.write_table(name=TXN_FEATURES, df=txn_feats, mode="merge")

# COMMAND ----------
profile = customer_profile_features(spark.table(ns.table("customers")))
if not spark.catalog.tableExists(PROFILE_FEATURES):
    fe.create_table(
        name=PROFILE_FEATURES,
        primary_keys=["customer_id"],
        df=profile,
        description="Slowly-changing customer attributes.",
    )
else:
    fe.write_table(name=PROFILE_FEATURES, df=profile, mode="merge")

# COMMAND ----------
# Liquid clustering on the lookup keys makes PIT joins and online sync fast.
spark.sql(f"ALTER TABLE {TXN_FEATURES} CLUSTER BY (customer_id, feature_ts)")
spark.sql(f"OPTIMIZE {TXN_FEATURES}")
display(spark.table(TXN_FEATURES).orderBy("customer_id", "feature_ts").limit(10))
```

> Older versions of the client use `timestamp_keys=` instead of `timeseries_columns=`.
> If one is rejected, use the other.

Run it. Then open Catalog Explorer → `main.fraud_dev.customer_txn_features`. Note the
**Lineage** tab: it already shows `silver_transactions` as upstream. After Day 3 it will
show which models consume it. That lineage is automatic and is what regulators ask for.

### 2.4 Build: On-demand feature functions

Some features need the *request* (the amount of the transaction being scored) combined
with a *looked-up* feature (the customer's 30-day average). They cannot be precomputed.
A UC Python function is the solution: defined once, executed in training set creation
and inside the serving endpoint.

Run in the SQL editor (or `spark.sql` in a notebook):

```sql
USE CATALOG main; USE SCHEMA fraud_dev;

CREATE OR REPLACE FUNCTION amount_ratio(amount DOUBLE, avg_amount_30d DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Transaction amount relative to the customer 30-day average. 1.0 when no history.'
AS $$
if avg_amount_30d is None or avg_amount_30d <= 0:
    return 1.0
return float(amount) / float(avg_amount_30d)
$$;

CREATE OR REPLACE FUNCTION is_foreign(country STRING, home_country STRING)
RETURNS INT
LANGUAGE PYTHON
COMMENT '1 if the transaction country differs from the customer home country.'
AS $$
if country is None or home_country is None:
    return 0
return int(country != home_country)
$$;

CREATE OR REPLACE FUNCTION account_age_days(event_ts TIMESTAMP, signup_date DATE)
RETURNS INT
LANGUAGE PYTHON
COMMENT 'Days between signup and the event; computed on demand so it is PIT-correct.'
AS $$
if signup_date is None or event_ts is None:
    return 0
return (event_ts.date() - signup_date).days
$$;

SELECT amount_ratio(500.0, 50.0), is_foreign('DE', 'US'), account_age_days(TIMESTAMP'2026-08-01', DATE'2024-01-01');
```

Keep these in `src/sql/feature_functions.sql` in the repo and apply them from a job task
(Day 5) so they are versioned like everything else.

### 2.5 Build: Assemble the training set

`src/notebooks/02b_training_set_demo.py` (exploratory; Day 3 puts this in the training
notebook):

```python
# Databricks notebook source
import sys
sys.path.append("../")
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from pyspark.sql import functions as F
from fraud.config import from_widgets

ns = from_widgets(dbutils)
fe = FeatureEngineeringClient()

# COMMAND ----------
# The "spine": one row per labeled event with the raw request-time columns and the label.
labels = (
    spark.table(ns.table("silver_transactions")).alias("t")
    .join(spark.table(ns.table("silver_labels")).alias("l"), "transaction_id")
    .select(
        "transaction_id", "customer_id", "event_ts",
        "amount", "country", "merchant_category", "device_type", "hour_of_day",
        F.col("is_fraud").cast("int").alias("is_fraud"),
    )
)
print(labels.count(), "labeled rows;", "fraud rate:", labels.agg(F.avg("is_fraud")).first()[0])

# COMMAND ----------
feature_spec = [
    FeatureLookup(
        table_name=ns.table("customer_txn_features"),
        lookup_key="customer_id",
        timestamp_lookup_key="event_ts",        # <- the PIT join
    ),
    FeatureLookup(
        table_name=ns.table("customer_profile_features"),
        lookup_key="customer_id",
    ),
    FeatureFunction(
        udf_name=ns.table("amount_ratio"),
        input_bindings={"amount": "amount", "avg_amount_30d": "avg_amount_30d"},
        output_name="amount_ratio",
    ),
    FeatureFunction(
        udf_name=ns.table("is_foreign"),
        input_bindings={"country": "country", "home_country": "home_country"},
        output_name="is_foreign",
    ),
    FeatureFunction(
        udf_name=ns.table("account_age_days"),
        input_bindings={"event_ts": "event_ts", "signup_date": "signup_date"},
        output_name="account_age_days",
    ),
]

training_set = fe.create_training_set(
    df=labels,
    feature_lookups=feature_spec,
    label="is_fraud",
    exclude_columns=["transaction_id", "event_ts", "signup_date", "home_country"],
)
train_df = training_set.load_df()
display(train_df.limit(10))
train_df.printSchema()
```

Inspect the output. For every labeled transaction you now have: request-time columns,
PIT-correct behavior features, profile features, and three computed features. The
feature-table lookup used the **AS OF** semantics automatically.

**Prove PIT correctness.** Pick one customer with many transactions and compare
`txn_count_24h` from `train_df` for their 5th transaction with a manual count of their
events in the preceding 24 hours in `silver_transactions`. They must match exactly.

### 2.6 Concepts: Feature table hygiene in a team

- **One table per entity and grain.** `customer_txn_features` (customer, event time),
  `customer_profile_features` (customer), later `merchant_features` (merchant, day).
  Do not create "model_X_features" tables; they are not reusable.
- **Descriptions and tags are mandatory.** Discovery is the point of a catalog.
- **Owners and permissions.** `GRANT SELECT ON TABLE ... TO `data-scientists``; only the
  feature pipeline's service principal gets `MODIFY`.
- **Backfill vs incremental.** Today's notebook recomputes all history (fine at 1M rows).
  At scale, compute only for new events using Change Data Feed on silver and `merge`.
  Exercise 3 below.
- **Freshness contract.** Document how fresh each table is (hourly? streaming?) because
  serving uses the latest row and a stale table silently degrades the model.

### 2.7 Exercises

1. **Merchant features.** Add `merchant_features` keyed by (merchant_id, feature_ts) with
   `merchant_txn_count_24h` and `merchant_fraud_rate_30d` (from labels; careful: labels
   arrive late, so compute using `labeled_at <= feature_ts`, not `event_ts`). Add it to
   the training set. This is the classic leakage trap; write in README why the naive
   version leaks.
2. **Test the function.** Write a pytest that calls `amount_ratio` via
   `spark.sql("SELECT main.fraud_dev.amount_ratio(10, 0)")` and asserts 1.0.
3. **Incremental feature build.** Rewrite `02_build_features` to read only rows from
   `silver_transactions` with `_ingested_at` after the max `feature_ts` currently in
   the feature table (plus a 30-day lookback for the windows), then `write_table(mode="merge")`.
   Confirm it processes only the new day after a generator run.
4. **Streaming feature.** Define `txn_count_10m` as a streaming aggregation written to
   the feature table with `fe.write_table(..., mode="merge")` from a streaming DataFrame
   (`checkpoint_location` required). Observe it updating as new files land.
5. **Lineage walk.** In Catalog Explorer, start at `bronze_transactions` and follow
   lineage down to `customer_txn_features`. Screenshot it for your README.

### 2.8 Checkpoint questions

1. Define point-in-time correctness and give the specific line of `features.py` that enforces it.
2. Why do on-demand features exist? Give one that cannot be precomputed and explain why.
3. What is the difference between `lookup_key` and `timestamp_lookup_key`, and what happens if you omit the latter on a time-series table?
4. How does the feature client eliminate training-serving skew for lookups *and* for computed features?
5. Why should merchant fraud rate be computed from `labeled_at` rather than `event_ts`?
6. What does lineage in Unity Catalog give you that MLflow tags alone did not?

---

## Day 3: Training at Scale, Tuning, Evaluation, and Governed Promotion

### Learning objectives

- Use MLflow on Databricks with Models in Unity Catalog and record full data lineage.
- Handle severe class imbalance and choose a decision threshold with a cost function.
- Tune hyperparameters with Optuna, logged as nested runs.
- Package the model with its feature specification so serving needs only raw inputs.
- Implement champion/challenger comparison and promotion as a multi-task Lakeflow Job.
- Know when and how to go distributed (pandas UDFs, Spark ML, Ray).

### 3.1 Concepts: MLflow on Databricks, what is different

Everything from week one applies. What changes:

- **The tracking server is built in.** No server to run; experiments live in the
  workspace and are permissioned like notebooks. `mlflow.set_experiment("/Users/you/fraud")`.
- **Models in Unity Catalog** replace the workspace registry. A model is
  `catalog.schema.name`; versions and aliases work as before; permissions are UC grants;
  lineage links the version to the tables and features it was trained on. Set
  `mlflow.set_registry_uri("databricks-uc")` (default on recent runtimes).
- **`fe.log_model`** replaces `mlflow.sklearn.log_model` when features come from the
  feature client. It stores the feature spec with the model, which is what enables
  automatic lookup at serving time and batch scoring with `fe.score_batch`.
- **Autologging** is on by default for sklearn, XGBoost, LightGBM, PyTorch Lightning.
  Explicit logging still wins for the things you care about.

### 3.2 Concepts: Imbalanced classification done right

With 1.5% positives:

- **Do not use accuracy.** A model that says "never fraud" scores 98.5%.
- **Primary metric: PR-AUC** (area under precision-recall curve), sometimes called
  average precision. It focuses on the positive class. ROC-AUC is fine as a secondary
  metric but inflates under imbalance.
- **The threshold is a business decision**, not 0.5. Compute expected cost:
  `cost = FP × review_cost + FN × avg_fraud_loss`. Pick the threshold that minimizes it,
  log it as a parameter, and serve probabilities so the threshold can change without
  retraining.
- **Class weights** (`class_weight="balanced"` or a positive-class weight) are the
  cheapest effective fix. Oversampling (SMOTE) is rarely worth it with tree models.
- **Time-based split, not random.** Train on the first 45 days, validate on days 46-52,
  test on days 53-60. A random split leaks temporal patterns and overstates performance.

### 3.3 Build: Evaluation utilities

`src/fraud/evaluation.py`:

```python
"""Metrics and threshold selection for imbalanced fraud detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

REVIEW_COST = 5.0  # USD to manually review one flagged transaction


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def choose_threshold(y_true, y_prob, amounts, review_cost: float = REVIEW_COST) -> tuple[float, dict]:
    """Pick the probability cutoff minimizing expected cost.

    FP cost: one review. FN cost: the transaction amount (we lose it).
    Returns (threshold, metrics_at_threshold).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amounts = np.asarray(amounts)
    best = (0.5, np.inf, {})
    for thr in np.linspace(0.02, 0.98, 97):
        pred = y_prob >= thr
        fp = pred & (y_true == 0)
        fn = (~pred) & (y_true == 1)
        tp = pred & (y_true == 1)
        cost = fp.sum() * review_cost + amounts[fn].sum()
        if cost < best[1]:
            precision = tp.sum() / max(pred.sum(), 1)
            recall = tp.sum() / max((y_true == 1).sum(), 1)
            best = (float(thr), float(cost), {
                "threshold": float(thr),
                "expected_cost": float(cost),
                "precision_at_thr": float(precision),
                "recall_at_thr": float(recall),
                "flag_rate": float(pred.mean()),
            })
    return best[0], best[2]


def recall_at_precision(y_true, y_prob, target_precision: float = 0.9) -> float:
    p, r, _ = precision_recall_curve(y_true, y_prob)
    ok = p >= target_precision
    return float(r[ok].max()) if ok.any() else 0.0
```

### 3.4 Build: The training notebook

`src/notebooks/03_train.py`:

```python
# Databricks notebook source
# MAGIC %pip install -q optuna>=3.6
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import sys
sys.path.append("../")
import json
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction, FeatureLookup
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud.config import from_widgets
from fraud.evaluation import choose_threshold, classification_metrics, recall_at_precision

ns = from_widgets(dbutils)
dbutils.widgets.text("n_trials", "15")
N_TRIALS = int(dbutils.widgets.get("n_trials"))

fe = FeatureEngineeringClient()
mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/fraud-{ns.schema}")
MODEL_NAME = ns.table("fraud_classifier")

# COMMAND ----------
# MAGIC %md ## 1. Spine with time-based split

# COMMAND ----------
labels = (
    spark.table(ns.table("silver_transactions")).alias("t")
    .join(spark.table(ns.table("silver_labels")).alias("l"), "transaction_id")
    .select("transaction_id", "customer_id", "event_ts", "amount", "country",
            "merchant_category", "device_type", "hour_of_day",
            F.col("is_fraud").cast("int").alias("is_fraud"))
)
bounds = labels.agg(F.min("event_ts").alias("lo"), F.max("event_ts").alias("hi")).first()
span = bounds.hi - bounds.lo
val_start = bounds.lo + span * 0.75
test_start = bounds.lo + span * 0.875
labels = labels.withColumn(
    "split",
    F.when(F.col("event_ts") < F.lit(val_start), "train")
     .when(F.col("event_ts") < F.lit(test_start), "val")
     .otherwise("test"),
)

# Record input data versions for lineage. This is the DVC hash of week one.
def delta_version(table: str) -> int:
    return spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()["version"]

data_versions = {
    t: delta_version(ns.table(t))
    for t in ["silver_transactions", "silver_labels", "customer_txn_features", "customer_profile_features"]
}
print(data_versions)

# COMMAND ----------
# MAGIC %md ## 2. Training set via feature client

# COMMAND ----------
feature_spec = [
    FeatureLookup(table_name=ns.table("customer_txn_features"), lookup_key="customer_id", timestamp_lookup_key="event_ts"),
    FeatureLookup(table_name=ns.table("customer_profile_features"), lookup_key="customer_id"),
    FeatureFunction(udf_name=ns.table("amount_ratio"), input_bindings={"amount": "amount", "avg_amount_30d": "avg_amount_30d"}, output_name="amount_ratio"),
    FeatureFunction(udf_name=ns.table("is_foreign"), input_bindings={"country": "country", "home_country": "home_country"}, output_name="is_foreign"),
    FeatureFunction(udf_name=ns.table("account_age_days"), input_bindings={"event_ts": "event_ts", "signup_date": "signup_date"}, output_name="account_age_days"),
]
# Two training-set objects over the same spine and spec:
#  - `training_set` excludes bookkeeping columns and is what gets packaged with the model
#    (its schema is the serving contract).
#  - `pdf` keeps transaction_id/event_ts/split so we can do the time-based split locally.
# ~1M rows x 20 columns fits comfortably in pandas on a single node.
training_set = fe.create_training_set(
    df=labels,
    feature_lookups=feature_spec,
    label="is_fraud",
    exclude_columns=["transaction_id", "event_ts", "signup_date", "home_country", "split"],
)
pdf = (
    fe.create_training_set(df=labels, feature_lookups=feature_spec, label="is_fraud",
                           exclude_columns=["signup_date", "home_country"])
    .load_df()
    .toPandas()
    .sort_values("event_ts")
)
train = pdf[pdf.split == "train"]; val = pdf[pdf.split == "val"]; test = pdf[pdf.split == "test"]
print(len(train), len(val), len(test), "fraud rates:", train.is_fraud.mean(), val.is_fraud.mean(), test.is_fraud.mean())

CATEGORICAL = ["country", "merchant_category", "device_type"]
DROP = ["transaction_id", "event_ts", "split", "is_fraud"]
FEATURES = [c for c in pdf.columns if c not in DROP]

def xy(df):
    return df[FEATURES], df["is_fraud"].values

X_tr, y_tr = xy(train); X_val, y_val = xy(val); X_te, y_te = xy(test)

# COMMAND ----------
# MAGIC %md ## 3. Model factory

# COMMAND ----------
def make_pipeline(params: dict) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL)],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        max_iter=params["max_iter"],
        l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])

# COMMAND ----------
# MAGIC %md ## 4. Tuning: Optuna, each trial a nested MLflow run

# COMMAND ----------
mlflow.sklearn.autolog(disable=True)  # we log explicitly; autolog would double-log per trial

def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 600),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
    }
    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
        model = make_pipeline(params).fit(X_tr, y_tr)
        prob = model.predict_proba(X_val)[:, 1]
        m = classification_metrics(y_val, prob)
        mlflow.log_params(params)
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
        return m["pr_auc"]

with mlflow.start_run(run_name="fraud-training") as parent:
    mlflow.set_tags({"stage": "training", "data_versions": json.dumps(data_versions)})
    mlflow.log_params({"n_trials": N_TRIALS, "split": "time-based 75/12.5/12.5", **{f"version_{k}": v for k, v in data_versions.items()}})

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params
    mlflow.log_params({f"best_{k}": v for k, v in best.items()})

    # COMMAND ----------
    # MAGIC %md ## 5. Final fit on train+val, evaluate on test, choose threshold

    # COMMAND ----------
    X_fit = pd.concat([X_tr, X_val]); y_fit = np.concatenate([y_tr, y_val])
    model = make_pipeline(best).fit(X_fit, y_fit)

    prob_te = model.predict_proba(X_te)[:, 1]
    test_metrics = classification_metrics(y_te, prob_te)
    threshold, thr_metrics = choose_threshold(y_te, prob_te, X_te["amount"].values)
    test_metrics["recall_at_p90"] = recall_at_precision(y_te, prob_te, 0.9)
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in thr_metrics.items()})
    mlflow.log_param("decision_threshold", threshold)
    print("TEST:", test_metrics, thr_metrics)

    # Baseline snapshot of training features for drift monitoring on Day 5
    baseline = X_fit.sample(n=min(20_000, len(X_fit)), random_state=0).copy()
    baseline["model_version"] = "pending"
    spark.createDataFrame(baseline).write.mode("overwrite").saveAsTable(ns.table("training_baseline"))

    # COMMAND ----------
    # MAGIC %md ## 6. Log with the feature spec and register in UC

    # COMMAND ----------
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
        input_example=X_tr.head(3),
    )
    run_id = parent.info.run_id

# COMMAND ----------
# Tag the new version with its threshold so serving consumers can read it.
from mlflow import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
version = max(int(v.version) for v in versions)
client.set_model_version_tag(MODEL_NAME, str(version), "decision_threshold", str(threshold))
client.set_model_version_tag(MODEL_NAME, str(version), "test_pr_auc", f"{test_metrics['pr_auc']:.4f}")
client.set_registered_model_alias(MODEL_NAME, "challenger", version)
dbutils.jobs.taskValues.set("model_version", version)   # hand off to the next task
print(f"Registered {MODEL_NAME} v{version} as challenger (threshold={threshold:.2f})")
```

Run it (set `n_trials` to 5 the first time; each trial is ~30 seconds on serverless).
Then:

- Open **Experiments** in the left nav → your experiment. You see the parent run with
  15 nested trial runs. The chart view plots `val_pr_auc` across trials.
- Open **Catalog** → `main.fraud_dev.fraud_classifier`. Version 1 exists, alias
  `challenger`, tags with threshold and PR-AUC. Click the version → **Lineage** shows the
  four tables and the notebook that produced it. That is the audit trail regulators want.
- Expand the run's artifacts: the `model/` folder contains the feature spec alongside the
  sklearn model. Open `MLmodel`; note the `databricks.feature_store` section listing
  lookups and functions.

Expect test PR-AUC around 0.6 to 0.8 on this synthetic data (planted patterns are
learnable but noisy). The absolute number matters less than the pipeline.

### 3.5 Build: Champion/challenger promotion

`src/fraud/promotion.py`:

```python
"""Decide whether a challenger model version should become champion."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MIN_IMPROVEMENT = 0.01    # PR-AUC points the challenger must gain
MAX_REGRESSION = 0.0      # never promote something worse


@dataclass
class Decision:
    promote: bool
    reason: str
    champion_version: int | None
    challenger_version: int


def current_alias_version(client: MlflowClient, name: str, alias: str) -> int | None:
    try:
        return int(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def metric_for_version(client: MlflowClient, name: str, version: int, metric: str) -> float | None:
    mv = client.get_model_version(name, str(version))
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(metric)


def decide(client: MlflowClient, name: str, challenger: int, metric: str = "test_pr_auc") -> Decision:
    champion = current_alias_version(client, name, "champion")
    if champion is None:
        return Decision(True, "no champion exists", None, challenger)
    if champion == challenger:
        return Decision(False, "challenger is already champion", champion, challenger)

    c_score = metric_for_version(client, name, champion, metric)
    n_score = metric_for_version(client, name, challenger, metric)
    if c_score is None or n_score is None:
        return Decision(False, f"missing {metric} on one of the versions", champion, challenger)

    delta = n_score - c_score
    if delta >= MIN_IMPROVEMENT:
        return Decision(True, f"{metric} improved by {delta:.4f}", champion, challenger)
    return Decision(False, f"{metric} delta {delta:.4f} below {MIN_IMPROVEMENT}", champion, challenger)


def apply(client: MlflowClient, name: str, decision: Decision) -> None:
    if not decision.promote:
        return
    if decision.champion_version is not None:
        client.set_registered_model_alias(name, "previous", decision.champion_version)
    client.set_registered_model_alias(name, "champion", decision.challenger_version)
    client.delete_registered_model_alias(name, "challenger")
```

> **Important caveat on comparing metrics across versions.** Champion and challenger were
> evaluated on *different* test windows (the data grew). A fair comparison scores both
> on the same, newest holdout. Exercise 2 fixes this; the version above is the simple
> first cut, and is what many teams actually run.

`src/notebooks/04_promote.py`:

```python
# Databricks notebook source
import sys
sys.path.append("../")
import mlflow
from mlflow import MlflowClient
from fraud.config import from_widgets
from fraud.promotion import apply, decide

ns = from_widgets(dbutils)
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
name = ns.table("fraud_classifier")

# Version comes from the train task; fall back to the challenger alias when run alone.
try:
    version = int(dbutils.jobs.taskValues.get(taskKey="train", key="model_version"))
except Exception:
    version = int(client.get_model_version_by_alias(name, "challenger").version)

decision = decide(client, name, version)
print(decision)
apply(client, name, decision)
dbutils.jobs.taskValues.set("promoted", decision.promote)
```

Run it. Version 1 becomes `champion` (no champion existed). Run `03_train` again with
a different `n_trials`, then `04_promote`: it compares and probably declines. Open the
model in Catalog Explorer and watch the aliases move.

### 3.6 Build: The training job

`resources/training_job.yml`:

```yaml
resources:
  jobs:
    fraud_training:
      name: "fraud-training"
      description: Rebuild features, train a challenger, promote if better.

      schedule:
        quartz_cron_expression: "0 0 3 ? * MON"     # Mondays 03:00
        timezone_id: UTC
        pause_status: PAUSED                        # Day 5 unpauses in prod

      parameters:
        - name: catalog
          default: ${var.catalog}
        - name: schema
          default: ${var.schema}

      tasks:
        - task_key: refresh_pipeline
          pipeline_task:
            pipeline_id: ${resources.pipelines.fraud_pipeline.id}

        - task_key: feature_functions
          depends_on: [{ task_key: refresh_pipeline }]
          sql_task:
            warehouse_id: ${var.warehouse_id}
            file:
              path: ../src/sql/feature_functions.sql

        - task_key: build_features
          depends_on: [{ task_key: feature_functions }]
          notebook_task:
            notebook_path: ../src/notebooks/02_build_features.py

        - task_key: train
          depends_on: [{ task_key: build_features }]
          notebook_task:
            notebook_path: ../src/notebooks/03_train.py
            base_parameters:
              n_trials: "15"

        - task_key: promote
          depends_on: [{ task_key: train }]
          notebook_task:
            notebook_path: ../src/notebooks/04_promote.py

      email_notifications:
        on_failure: [REPLACE_WITH_YOUR_EMAIL]
```

Add `warehouse_id` to `variables` in `databricks.yml` (find it under SQL Warehouses →
your warehouse → Connection details). Notebook tasks with no cluster specified run on
serverless compute.

Make the SQL file environment-aware: `src/sql/feature_functions.sql` should start with
`USE CATALOG IDENTIFIER(:catalog); USE SCHEMA IDENTIFIER(:schema);` and the task gets
`parameters: {catalog: ${var.catalog}, schema: ${var.schema}}`. (Alternatively, apply
functions from a small notebook task; both patterns are common.)

```bash
databricks bundle deploy
databricks bundle run fraud_training
```

The job UI shows the five tasks as a DAG, each with its own logs. A failed task can be
**repaired** (rerun from the failure point) without rerunning the pipeline refresh.

### 3.7 Concepts: When and how to go distributed

Your data is ~1M rows; pandas on a single serverless node is right. Know the next steps
for when it is not:

| Situation | Approach | Databricks mechanism |
|---|---|---|
| Features too big for pandas, model fits | Compute features in Spark (done), `toPandas()` a sample or the result | What you did today |
| Many independent models (per country, per merchant) | Train in parallel with grouped pandas UDFs | `df.groupBy("country").applyInPandas(train_fn, schema)` |
| Hyperparameter trials in parallel | Optuna with a distributed storage, or Ray Tune on a Ray-on-Spark cluster | `ray.util.spark.setup_ray_cluster` |
| Single model on 100M+ rows, tabular | Spark ML (`GBTClassifier`), or XGBoost/LightGBM Spark estimators | `xgboost.spark.SparkXGBClassifier` |
| Deep learning | Single-node multi-GPU, or TorchDistributor | `pyspark.ml.torch.distributor.TorchDistributor` |
| Scoring 1B rows | Spark-native inference via `fe.score_batch` or `mlflow.pyfunc.spark_udf` | Day 4 |

The pattern across all of them: **the orchestration, tracking, registry, and serving
stay identical.** Only the training function's internals change.

### 3.8 Exercises

1. **Threshold as a model artifact.** Wrap the sklearn pipeline in an `mlflow.pyfunc.PythonModel`
   whose `predict` returns both `probability` and `is_fraud` using the stored threshold,
   and log that with `flavor=mlflow.pyfunc`. Decide whether you prefer this or the tag
   approach and record why.
2. **Fair comparison.** Change `decide` to re-score both champion and challenger on the
   latest test window using `fe.score_batch` (Day 4) and compare PR-AUC on identical rows.
3. **Per-country models.** Use `applyInPandas` to train one model per country and log
   each as a nested run. Compare against the global model on the test set per country.
4. **Model validation tests.** Add a `validate` task between `train` and `promote` that
   fails if: recall at 90% precision < 0.3; flag rate > 5%; or the prediction
   distribution on a fixed probe set differs from champion's by more than a threshold.
5. **Experiment comparison without the UI.** Use `mlflow.search_runs` to produce a
   DataFrame of all parent runs with test metrics and data versions, sorted by
   `test_pr_auc`. This becomes a dashboard query.

### 3.9 Checkpoint questions

1. Why is PR-AUC the primary metric here and what would ROC-AUC hide?
2. Explain the cost function used to choose the threshold. Who should own the `REVIEW_COST` number?
3. Why a time-based split? What would a random split overstate, specifically for the `txn_count_24h` feature?
4. What does `fe.log_model` store that `mlflow.sklearn.log_model` does not, and what does it enable?
5. What are the three lineage links recorded for version 1 of the model, and how did each get there?
6. What is wrong with comparing champion and challenger on their own test sets, and how do you fix it?

---

## Day 4: Batch and Real-Time Serving with Automatic Feature Lookup

### Learning objectives

- Run scheduled batch scoring over Spark with feature lookups handled automatically.
- Publish feature tables to an online store for millisecond lookups.
- Deploy a Model Serving endpoint that performs feature lookup and on-demand
  computation from a raw request.
- Capture every request and response in an inference table.
- Split traffic between versions, measure latency, and roll back.

### 4.1 Build: Batch scoring job

`src/notebooks/05_batch_score.py`:

```python
# Databricks notebook source
# MAGIC %md # Nightly batch scoring of recent transactions with the champion model

# COMMAND ----------
import sys
sys.path.append("../")
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow import MlflowClient
from pyspark.sql import functions as F
from fraud.config import from_widgets

ns = from_widgets(dbutils)
dbutils.widgets.text("lookback_hours", "24")
LOOKBACK = int(dbutils.widgets.get("lookback_hours"))

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()
client = MlflowClient()
name = ns.table("fraud_classifier")
mv = client.get_model_version_by_alias(name, "champion")
threshold = float(mv.tags.get("decision_threshold", "0.5"))
print(f"Scoring with {name} v{mv.version} threshold={threshold}")

# COMMAND ----------
# Only the raw request-time columns. Lookups and functions are resolved from the model's feature spec.
recent = (
    spark.table(ns.table("silver_transactions"))
    .where(F.col("event_ts") >= F.current_timestamp() - F.expr(f"INTERVAL {LOOKBACK} HOURS"))
    .select("transaction_id", "customer_id", "event_ts", "amount", "country",
            "merchant_category", "device_type", "hour_of_day")
)
# In a demo with synthetic timestamps in the past, fall back to the latest day of data:
if recent.isEmpty():
    max_ts = spark.table(ns.table("silver_transactions")).agg(F.max("event_ts")).first()[0]
    recent = (spark.table(ns.table("silver_transactions"))
              .where(F.col("event_ts") >= F.lit(max_ts) - F.expr(f"INTERVAL {LOOKBACK} HOURS"))
              .select("transaction_id", "customer_id", "event_ts", "amount", "country",
                      "merchant_category", "device_type", "hour_of_day"))

scored = fe.score_batch(model_uri=f"models:/{name}@champion", df=recent, result_type="double")
# score_batch returns a `prediction` column; for sklearn classifiers it is the class.
# To get probabilities, score with the predict_proba wrapper from Day 3 exercise 1, or:
```

> `fe.score_batch` calls the model's `predict`. For a probability, log the model with a
> pyfunc wrapper that returns `predict_proba[:, 1]` (Day 3, exercise 1). The rest of
> this notebook assumes the wrapper, so `prediction` is a probability. If you have not
> done that exercise, treat `prediction` as the 0/1 class and skip the threshold line.

```python
# COMMAND ----------
out = (
    scored.withColumn("fraud_probability", F.col("prediction").cast("double"))
    .withColumn("is_flagged", (F.col("fraud_probability") >= F.lit(threshold)).cast("int"))
    .withColumn("model_name", F.lit(name))
    .withColumn("model_version", F.lit(int(mv.version)))
    .withColumn("scored_at", F.current_timestamp())
    .select("transaction_id", "customer_id", "event_ts", "amount", "fraud_probability",
            "is_flagged", "model_name", "model_version", "scored_at")
)

target = ns.table("gold_transaction_scores")
if spark.catalog.tableExists(target):
    out.createOrReplaceTempView("new_scores")
    spark.sql(f"""
        MERGE INTO {target} t USING new_scores s ON t.transaction_id = s.transaction_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
else:
    out.write.clusterBy("event_ts").saveAsTable(target)

display(spark.table(target).agg(F.count("*"), F.avg("is_flagged"), F.max("scored_at")))
```

`resources/scoring_job.yml`:

```yaml
resources:
  jobs:
    fraud_batch_scoring:
      name: "fraud-batch-scoring"
      schedule:
        quartz_cron_expression: "0 30 1 * * ?"
        timezone_id: UTC
        pause_status: PAUSED
      parameters:
        - { name: catalog, default: "${var.catalog}" }
        - { name: schema, default: "${var.schema}" }
      tasks:
        - task_key: refresh_pipeline
          pipeline_task:
            pipeline_id: ${resources.pipelines.fraud_pipeline.id}
        - task_key: build_features
          depends_on: [{ task_key: refresh_pipeline }]
          notebook_task:
            notebook_path: ../src/notebooks/02_build_features.py
        - task_key: score
          depends_on: [{ task_key: build_features }]
          notebook_task:
            notebook_path: ../src/notebooks/05_batch_score.py
            base_parameters: { lookback_hours: "24" }
```

```bash
databricks bundle deploy && databricks bundle run fraud_batch_scoring
```

The scoring table now feeds a dashboard (Databricks SQL → Dashboards → new dashboard on
`gold_transaction_scores`: flagged count per hour, flag rate, top flagged merchants).
Build a three-panel dashboard now; you will add monitoring panels on Day 5.

### 4.2 Concepts: Online feature stores

Batch scoring reads features from Delta: fine at minutes of latency. The authorization
path needs the customer's features in **under 10 ms**. Delta on object storage cannot do
that. An **online store** is a low-latency key-value copy of a feature table, kept in
sync with the Delta source.

On Databricks this is an **online table** (classic) or a **Lakebase synced table**
(newer, Postgres-based; the direction the platform is going). Both are created from a UC
feature table, both are discovered automatically by Model Serving when the model was
logged with `fe.log_model`, and both support time-series tables by serving the **latest**
row per primary key.

Decision rule: use what your workspace offers in Catalog Explorer → table → **Create** →
*Online table* or *Synced table*. The concept and the serving integration are identical.

**[Full workspace]** Free Edition may not expose online stores. If so, do 4.3 by reading,
and in 4.4 deploy the endpoint without feature lookup using the "request contains all
features" variant shown there.

### 4.3 Build: Publish feature tables online **[Full workspace]**

Via UI: Catalog → `customer_txn_features` → Create → Online table (or Synced table) →
primary key `customer_id`, timeseries key `feature_ts`, sync mode *Triggered* → Create.
Repeat for `customer_profile_features` (no timeseries key).

Via SDK, classic online tables:

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient()

def publish(source: str, pk: list[str], ts_key: str | None):
    spec = OnlineTableSpec(
        primary_key_columns=pk,
        timeseries_key=ts_key,
        source_table_full_name=source,
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    )
    return w.online_tables.create_and_wait(table=OnlineTable(name=f"{source}_online", spec=spec))

publish(ns.table("customer_txn_features"), ["customer_id"], "feature_ts")
publish(ns.table("customer_profile_features"), ["customer_id"], None)
```

> On workspaces that have moved to Lakebase, the client exposes
> `fe.create_online_store(...)` and `fe.publish_table(...)` instead. The docs page
> "Online feature serving" is authoritative; the rest of this day is unchanged.

Triggered sync means a refresh happens when you request it (or after the feature job
runs). Add a refresh step to the batch scoring and training jobs in the exercises.

### 4.4 Build: Deploy the serving endpoint

`src/notebooks/06_deploy_serving.py`:

```python
# Databricks notebook source
# MAGIC %md # Create or update the real-time endpoint for the champion model

# COMMAND ----------
import sys
sys.path.append("../")
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig, AiGatewayInferenceTableConfig, AiGatewayRateLimit, AiGatewayRateLimitKey, AiGatewayRateLimitRenewalPeriod,
    EndpointCoreConfigInput, Route, ServedEntityInput, TrafficConfig,
)
from mlflow import MlflowClient
from fraud.config import from_widgets

ns = from_widgets(dbutils)
dbutils.widgets.text("endpoint_name", "fraud-classifier")
dbutils.widgets.text("challenger_traffic", "0")
ENDPOINT = dbutils.widgets.get("endpoint_name")
CHALLENGER_PCT = int(dbutils.widgets.get("challenger_traffic"))

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
w = WorkspaceClient()
name = ns.table("fraud_classifier")

champion = client.get_model_version_by_alias(name, "champion").version
served = [
    ServedEntityInput(
        name="champion",
        entity_name=name,
        entity_version=str(champion),
        workload_size="Small",
        scale_to_zero_enabled=True,          # dev: save money; prod: False to avoid cold starts
    )
]
routes = [Route(served_model_name="champion", traffic_percentage=100 - CHALLENGER_PCT)]

if CHALLENGER_PCT > 0:
    challenger = client.get_model_version_by_alias(name, "challenger").version
    served.append(ServedEntityInput(name="challenger", entity_name=name, entity_version=str(challenger),
                                    workload_size="Small", scale_to_zero_enabled=True))
    routes.append(Route(served_model_name="challenger", traffic_percentage=CHALLENGER_PCT))

config = EndpointCoreConfigInput(served_entities=served, traffic_config=TrafficConfig(routes=routes))
gateway = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        catalog_name=ns.catalog, schema_name=ns.schema, table_name_prefix="fraud_serving", enabled=True
    ),
    rate_limits=[AiGatewayRateLimit(calls=1000, key=AiGatewayRateLimitKey.ENDPOINT,
                                    renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE)],
)

# COMMAND ----------
existing = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT in existing:
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT, served_entities=served, traffic_config=TrafficConfig(routes=routes))
    w.serving_endpoints.put_ai_gateway(name=ENDPOINT, inference_table_config=gateway.inference_table_config, rate_limits=gateway.rate_limits)
else:
    w.serving_endpoints.create_and_wait(name=ENDPOINT, config=config, ai_gateway=gateway)
print(w.serving_endpoints.get(ENDPOINT).state)
```

Run it. Creation takes 5 to 10 minutes (container build + model load). Watch it under
**Serving** in the left nav. When **Ready**, test from the notebook:

```python
resp = w.serving_endpoints.query(
    name=ENDPOINT,
    dataframe_records=[{
        "customer_id": "c_000123",
        "amount": 640.0,
        "country": "BR",
        "merchant_category": "electronics",
        "device_type": "web",
        "hour_of_day": 2,
    }],
)
print(resp.predictions)
```

Read the request body again. It contains **only the raw request fields plus the lookup
key.** The endpoint looked up 9 behavior features and 3 profile features from the online
store, executed the 3 feature functions, assembled the exact training-time feature frame,
and scored it. There is no feature code in the serving path for you to keep in sync.

Try `"country": "US"` with `amount: 20.0` for the same customer. The probability should
drop sharply.

**Without online tables [Free Edition variant]:** log a plain `mlflow.sklearn` model
(not `fe.log_model`) on Day 3 and send every feature in the request. You lose automatic
lookup but keep everything else (inference tables, traffic split, monitoring).

### 4.5 Build: Call it like an application would

From your laptop, with a personal access token or OAuth token:

```bash
export DATABRICKS_TOKEN=$(databricks auth token --profile fraud | jq -r .access_token)
export HOST=https://YOUR-WORKSPACE-URL

curl -s -X POST "$HOST/serving-endpoints/fraud-classifier/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"customer_id":"c_000123","amount":640.0,"country":"BR","merchant_category":"electronics","device_type":"web","hour_of_day":2}]}'
```

Load test with `hey` as in week one (same body in a file). Record p50/p95/p99 and
requests per second for `Small` workload. Then set `workload_size="Medium"`, redeploy,
measure again. Write the numbers in README. This is capacity planning.

A real authorization system would call this from its service, with a **service
principal** token (Day 5), a timeout of ~80 ms, and a fallback rule when the endpoint
does not answer in time. Add the fallback logic to your README's design notes.

### 4.6 Concepts: Inference tables

Because you enabled the AI Gateway inference table, every request and response is
being written to `main.fraud_dev.fraud_serving_payload` (name varies by version:
look in the schema). Columns include `request_time`, `request` (JSON), `response`
(JSON), `served_entity_id`, `status_code`, and latency.

This table is:

- your **prediction log** from week one, with zero code;
- the **input to monitoring** on Day 5;
- the evidence for any **dispute** ("why was my card blocked at 14:02?").

Query it after the load test:

```sql
SELECT request_time, status_code, execution_duration_ms, served_entity_id
FROM main.fraud_dev.fraud_serving_payload
ORDER BY request_time DESC LIMIT 20;
```

The JSON needs unpacking into columns before monitoring; Day 5 does that.

### 4.7 Build: Traffic split and rollback

Register a second version (rerun `03_train` with different `n_trials`; it becomes
`challenger`). Then run `06_deploy_serving` with `challenger_traffic=10`. The endpoint
now routes 10% of traffic to the challenger and the inference table records which
`served_entity_id` answered each request. On Day 5 you compare them.

Rollback in two commands:

```python
# Model-level: point champion back to previous and redeploy
client.set_registered_model_alias(name, "champion", client.get_model_version_by_alias(name, "previous").version)
# then rerun 06_deploy_serving with challenger_traffic=0
```

or endpoint-level in the UI: Serving → endpoint → **Edit** → set the previous version to
100%. Both take under a minute; the endpoint keeps serving during the swap.

### 4.8 Exercises

1. **Online refresh in jobs.** Add a task after `build_features` in both jobs that
   triggers the online table refresh (`w.online_tables` / pipeline refresh for the
   synced table). Verify freshness by checking a customer's `txn_count_24h` online
   after generating a new day of data.
2. **Latency budget.** Instrument a client script to measure end-to-end latency
   including network. Break it down: endpoint `execution_duration_ms` vs total. Where
   does the time go? What would you change to hit a 50 ms p99?
3. **Provisioned throughput.** Set `scale_to_zero_enabled=False` and compare the first
   request after 15 minutes idle. Compute the monthly cost difference from the pricing
   page. Write the recommendation for prod.
4. **Shadow scoring.** Instead of a traffic split, call both versions for every request
   from a batch job over yesterday's inference table and compare their flag rates and
   agreement. Which approach (split vs shadow) is safer for fraud and why?
5. **Streaming scoring.** Use Structured Streaming to read `silver_transactions` as a
   stream and score each micro-batch with `fe.score_batch` inside `foreachBatch`,
   writing to `gold_transaction_scores`. This is the middle ground between nightly
   batch and per-request serving.

### 4.9 Checkpoint questions

1. What does `fe.score_batch` do with the model's feature spec that a plain `predict` does not?
2. Why can Delta not serve features at authorization time, and what does an online table change?
3. List exactly what the serving endpoint does between receiving `{"customer_id": ..., "amount": ...}` and returning a score.
4. Where are prediction logs, who writes them, and what three things are they used for?
5. Describe two ways to roll back a bad model and the trade-off between them.
6. When would you choose streaming scoring over a real-time endpoint?

---

## Day 5: Monitoring, Retraining Loops, CI/CD with Asset Bundles, and Governance

### Learning objectives

- Turn the inference table into a monitored, labeled scoring log.
- Create a Lakehouse Monitoring inference profile with drift and quality metrics.
- Alert on drift and trigger retraining automatically.
- Promote the bundle through dev, staging, and prod with GitHub Actions and service
  principals.
- Apply governance: permissions, lineage, cost controls, runbooks.
- Run the end-to-end incident drill.

### 5.1 Build: Unpack and label the inference log

`src/notebooks/07_unpack_inference.py`:

```python
# Databricks notebook source
# MAGIC %md # Flatten the serving inference table and join delayed labels

# COMMAND ----------
import sys
sys.path.append("../")
from pyspark.sql import functions as F, types as T
from fraud.config import from_widgets

ns = from_widgets(dbutils)
payload = ns.table("fraud_serving_payload")       # check the exact name in the schema
scored = ns.table("serving_scored")

# COMMAND ----------
req_schema = T.StructType([T.StructField("dataframe_records", T.ArrayType(T.MapType(T.StringType(), T.StringType())))])
resp_schema = T.StructType([T.StructField("predictions", T.ArrayType(T.DoubleType()))])

raw = spark.table(payload).where("status_code = 200")
flat = (
    raw.withColumn("req", F.from_json("request", req_schema))
       .withColumn("resp", F.from_json("response", resp_schema))
       .withColumn("pair", F.arrays_zip("req.dataframe_records", "resp.predictions"))
       .withColumn("pair", F.explode("pair"))
       .select(
           F.col("databricks_request_id").alias("request_id"),
           F.col("request_time").alias("request_ts"),
           F.col("served_entity_id"),
           F.col("pair.dataframe_records").alias("rec"),
           F.col("pair.predictions").alias("prediction"),
       )
       .select(
           "request_id", "request_ts", "served_entity_id", "prediction",
           F.col("rec.customer_id").alias("customer_id"),
           F.col("rec.amount").cast("double").alias("amount"),
           F.col("rec.country").alias("country"),
           F.col("rec.merchant_category").alias("merchant_category"),
           F.col("rec.device_type").alias("device_type"),
           F.col("rec.hour_of_day").cast("int").alias("hour_of_day"),
           F.col("rec.transaction_id").alias("transaction_id"),   # include it in requests from your app
       )
)

# Map served_entity_id -> model version so the monitor can slice by version
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
ep = w.serving_endpoints.get("fraud-classifier")
version_map = {e.name: e.entity_version for e in ep.config.served_entities}
flat = flat.withColumn("model_version", F.col("served_entity_id"))
for k, v in version_map.items():
    flat = flat.withColumn("model_version", F.when(F.col("served_entity_id") == k, F.lit(v)).otherwise(F.col("model_version")))

# COMMAND ----------
# Join whatever labels have arrived so far. Most recent rows will have NULL is_fraud; that is expected.
labels = spark.table(ns.table("silver_labels")).select("transaction_id", "is_fraud", "labeled_at")
joined = flat.join(labels, "transaction_id", "left")

joined.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(scored)
spark.sql(f"ALTER TABLE {scored} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
display(spark.table(scored).agg(F.count("*"), F.count("is_fraud").alias("labeled"), F.avg("prediction")))
```

> **Make your app send `transaction_id` in the request.** It is ignored by the model
> (not a feature) but it is the join key that makes performance monitoring possible.
> Add it to the test requests from Day 4 and re-send traffic before continuing.
> If the model signature rejects extra columns, log the model with `transaction_id`
> included in the training spine and in `exclude_columns`; the feature client then
> passes it through.

### 5.2 Concepts: Lakehouse Monitoring

Instead of Evidently scripts and Prometheus histograms, Databricks has a managed
monitor that you attach to a table. For an **inference profile** you tell it which
columns are timestamp, prediction, label, and model id. It then computes, per time
window and per model version:

- **Profile metrics**: per-column statistics (nulls, mean, quantiles, distinct counts)
- **Drift metrics**: statistical distance of each column vs a baseline table and vs the
  previous window (KS, Wasserstein, chi-square, JS divergence)
- **Model quality metrics**: precision, recall, F1, accuracy, confusion counts, on the
  rows where the label is present

Output is two Delta tables (`<table>_profile_metrics`, `<table>_drift_metrics`) and an
auto-generated dashboard. Because the outputs are tables, alerts are just SQL alerts.

> In the UI this now lives under **Data Quality Monitoring** on a table's Quality tab.
> The SDK call below is `quality_monitors`. **[Full workspace]**: Free Edition may not
> include it; if so, run the manual drift SQL in 5.4 instead and read this section.

### 5.3 Build: Create the monitor **[Full workspace]**

`src/notebooks/08_monitor.py`:

```python
# Databricks notebook source
import sys
sys.path.append("../")
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
from fraud.config import from_widgets

ns = from_widgets(dbutils)
w = WorkspaceClient()
user = spark.sql("SELECT current_user()").first()[0]
table = ns.table("serving_scored")

# COMMAND ----------
try:
    w.quality_monitors.get(table_name=table)
    w.quality_monitors.run_refresh(table_name=table)
    print("monitor exists; refresh triggered")
except Exception:
    w.quality_monitors.create(
        table_name=table,
        assets_dir=f"/Workspace/Users/{user}/monitoring/{ns.schema}",
        output_schema_name=f"{ns.catalog}.{ns.schema}",
        baseline_table_name=ns.table("training_baseline"),
        inference_log=MonitorInferenceLog(
            timestamp_col="request_ts",
            model_id_col="model_version",
            prediction_col="prediction",
            label_col="is_fraud",
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,  # prediction is a probability; use CLASSIFICATION if you serve 0/1
            granularities=["1 hour", "1 day"],
        ),
        slicing_exprs=["country", "device_type"],
    )
    print("monitor created; first refresh running")
```

> The baseline table from Day 3 needs the same column names as the monitored table
> for the drift comparison to apply (the monitor matches by name). Rename or add
> columns in `training_baseline` to match `serving_scored`; a quick `CREATE OR REPLACE
> TABLE ... AS SELECT` does it.

After the first refresh (a few minutes), open the table in Catalog Explorer → Quality
tab → **View dashboard**. Explore: prediction distribution over time per model version,
drift scores per feature, and (where labels exist) precision and recall.

### 5.4 Build: Simulate drift, detect it, and alert

Generate drifted traffic. Run `00_generate_data` with `days=2, start=2026-09-01,
drift=true`, run the pipeline and feature job, then send 2,000 requests sampled from
those rows to the endpoint (adapt `simulate_traffic.py` from week one to read from
`silver_transactions` and post to the endpoint). Run `07_unpack_inference` and refresh
the monitor.

The drift dashboard should light up on `device_type` (fraud moved to mobile) and
`amount` (smaller). Prediction distribution shifts too.

**SQL alert** (works on any workspace, with or without the managed monitor). If you
have the monitor, alert on its drift table:

```sql
SELECT column_name, drift_type, window.start AS window_start, js_distance, ks_test.pvalue
FROM main.fraud_dev.serving_scored_drift_metrics
WHERE drift_type = 'BASELINE'
  AND window.start >= current_timestamp() - INTERVAL 1 DAY
  AND column_name IN ('amount','device_type','country','prediction')
  AND (js_distance > 0.1 OR ks_test.pvalue < 0.01)
```

If you do not, compute PSI by hand on the scored table vs the baseline:

```sql
WITH b AS (
  SELECT width_bucket(prediction, 0, 1, 10) AS bin, COUNT(*)/SUM(COUNT(*)) OVER () AS p
  FROM main.fraud_dev.serving_scored WHERE request_ts < current_timestamp() - INTERVAL 7 DAYS GROUP BY 1),
c AS (
  SELECT width_bucket(prediction, 0, 1, 10) AS bin, COUNT(*)/SUM(COUNT(*)) OVER () AS p
  FROM main.fraud_dev.serving_scored WHERE request_ts >= current_timestamp() - INTERVAL 1 DAY GROUP BY 1)
SELECT SUM((c.p - b.p) * LN(c.p / b.p)) AS psi
FROM b JOIN c USING (bin);
```

Create the alert: SQL → Alerts → New alert → paste the query → condition "row count > 0"
(or `psi > 0.2`) → schedule hourly → notification to your email. Add a second alert on
**label-based quality**: recall over the last 7 labeled days below 0.4 from
`serving_scored_profile_metrics` (column `recall`), or computed directly from
`serving_scored`.

### 5.5 Build: Close the loop, retraining on drift

A monitoring job that unpacks inference, refreshes the monitor, evaluates the drift
condition, and runs the training job if it fires. The "if" uses a job **condition
task**, so the orchestration is visible in the DAG rather than buried in code.

`resources/monitoring_job.yml`:

```yaml
resources:
  jobs:
    fraud_monitoring:
      name: "fraud-monitoring"
      schedule:
        quartz_cron_expression: "0 0 */6 * * ?"      # every 6 hours
        timezone_id: UTC
        pause_status: PAUSED
      parameters:
        - { name: catalog, default: "${var.catalog}" }
        - { name: schema, default: "${var.schema}" }
      tasks:
        - task_key: unpack
          notebook_task:
            notebook_path: ../src/notebooks/07_unpack_inference.py

        - task_key: refresh_monitor
          depends_on: [{ task_key: unpack }]
          notebook_task:
            notebook_path: ../src/notebooks/08_monitor.py

        - task_key: check_drift
          depends_on: [{ task_key: refresh_monitor }]
          notebook_task:
            notebook_path: ../src/notebooks/09_check_drift.py   # sets taskValue drift_detected = "true"/"false"

        - task_key: drift_gate
          depends_on: [{ task_key: check_drift }]
          condition_task:
            op: EQUAL_TO
            left: "{{tasks.check_drift.values.drift_detected}}"
            right: "true"

        - task_key: retrain
          depends_on: [{ task_key: drift_gate, outcome: "true" }]
          run_job_task:
            job_id: ${resources.jobs.fraud_training.id}

        - task_key: redeploy
          depends_on: [{ task_key: retrain }]
          notebook_task:
            notebook_path: ../src/notebooks/06_deploy_serving.py
            base_parameters: { challenger_traffic: "0" }
```

`src/notebooks/09_check_drift.py` runs the drift SQL from 5.4 and sets
`dbutils.jobs.taskValues.set("drift_detected", "true" if rows else "false")`.

Deploy and run it manually. With drifted traffic in the log, the gate opens, the
training job runs as a child run (visible in the UI as a nested job), promotion
decides, and the endpoint is updated to whatever `champion` is. Then run the generator
without drift, send normal traffic, and run the job again: the gate stays closed.

> **Safety valve.** Auto-redeploy is appropriate only because promotion has a gate and
> rollback is one command. Many teams stop the automation at "open a ticket with the
> comparison report" and keep a human in the final step. Decide for your use case and
> document it in the runbook.

### 5.6 Concepts: Environments and the release process

You have been deploying to `dev` as yourself. Production needs:

- **Separate namespaces** per environment (`fraud_dev`, `fraud_staging`, `fraud`), the
  same code, different `schema` variable. The catalog can also differ
  (`dev`/`staging`/`prod` catalogs are common at larger companies).
- **Service principals** (SPs): non-human identities that own and run production jobs
  and endpoints. Nothing in prod runs as a person. Create one per environment (Settings
  → Identity and access → Service principals), grant it `USE CATALOG`, `USE SCHEMA`,
  `CREATE TABLE`, `CREATE MODEL`, `EXECUTE` on functions, and `CAN MANAGE` on the
  endpoint. Generate an OAuth secret for CI.
- **Promotion = redeploy the same bundle to the next target.** Not "copy the model."
  Code, jobs, pipeline, and endpoint definition all move together, so staging is a true
  rehearsal of prod.
- **Models cross environments through the registry, not re-training.** A common pattern:
  staging trains and registers in `staging` schema; after validation, the version is
  copied to prod with `mlflow.register_model("models:/main.fraud_staging.fraud_classifier/7", "main.fraud.fraud_classifier")`
  (UC supports cross-schema registration with lineage preserved), then prod's endpoint
  is updated. Alternatively prod retrains on prod data on its own schedule. Both are
  valid; the second is simpler and what this guide's jobs do.

### 5.7 Build: CI/CD with GitHub Actions

Secrets to add in the GitHub repo (Settings → Secrets → Actions):
`DATABRICKS_HOST`, `SP_STAGING_CLIENT_ID`, `SP_STAGING_CLIENT_SECRET`,
`SP_PROD_CLIENT_ID`, `SP_PROD_CLIENT_SECRET`.

`.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --dev
      - run: uv run ruff check . && uv run ruff format --check .
      - name: Unit tests (local Spark fallback)
        run: uv run pytest -q tests/test_features.py tests/test_evaluation.py

  validate-bundle:
    runs-on: ubuntu-latest
    needs: unit
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle validate -t staging
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_CLIENT_ID: ${{ secrets.SP_STAGING_CLIENT_ID }}
          DATABRICKS_CLIENT_SECRET: ${{ secrets.SP_STAGING_CLIENT_SECRET }}

  deploy-staging:
    runs-on: ubuntu-latest
    needs: validate-bundle
    if: github.event_name == 'pull_request'
    environment: staging
    env:
      DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
      DATABRICKS_CLIENT_ID: ${{ secrets.SP_STAGING_CLIENT_ID }}
      DATABRICKS_CLIENT_SECRET: ${{ secrets.SP_STAGING_CLIENT_SECRET }}
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle deploy -t staging
      - name: Integration test, full training on staging data
        run: databricks bundle run fraud_training -t staging
      - name: Smoke-score with the staging champion
        run: databricks bundle run fraud_batch_scoring -t staging
```

`.github/workflows/release.yml`:

```yaml
name: Release to prod

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy-prod:
    runs-on: ubuntu-latest
    environment: production          # configure "required reviewers" on this GitHub environment
    env:
      DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
      DATABRICKS_CLIENT_ID: ${{ secrets.SP_PROD_CLIENT_ID }}
      DATABRICKS_CLIENT_SECRET: ${{ secrets.SP_PROD_CLIENT_SECRET }}
    steps:
      - uses: actions/checkout@v4
      - uses: databricks/setup-cli@main
      - run: databricks bundle validate -t prod
      - run: databricks bundle deploy -t prod
      - name: Unpause schedules (prod mode requires explicit schedules; they deploy as defined)
        run: echo "Schedules defined in resources/*.yml are live in prod mode"
```

Set `pause_status: UNPAUSED` under a `targets.prod.resources.jobs.<name>.schedule`
override in `databricks.yml` so dev stays paused but prod runs. Example:

```yaml
targets:
  prod:
    # ...
    resources:
      jobs:
        fraud_training:
          schedule: { quartz_cron_expression: "0 0 3 ? * MON", timezone_id: UTC, pause_status: UNPAUSED }
        fraud_batch_scoring:
          schedule: { quartz_cron_expression: "0 30 1 * * ?", timezone_id: UTC, pause_status: UNPAUSED }
        fraud_monitoring:
          schedule: { quartz_cron_expression: "0 0 */6 * * ?", timezone_id: UTC, pause_status: UNPAUSED }
```

Open a PR → CI deploys and runs on staging → review → merge → prod deploy behind a
required reviewer. That is MLOps maturity Level 2 on a managed platform.

### 5.8 Build: Serving endpoint as a bundle resource

Endpoints can be bundle resources too, so prod gets its endpoint from YAML rather
than from the notebook:

`resources/serving.yml`:

```yaml
resources:
  model_serving_endpoints:
    fraud_endpoint:
      name: ${var.endpoint_name}
      config:
        served_entities:
          - name: champion
            entity_name: ${var.catalog}.${var.schema}.${var.model_name}
            entity_version: "1"          # bump via PR, or keep the notebook-driven path for alias-based updates
            workload_size: Small
            scale_to_zero_enabled: true
        traffic_config:
          routes:
            - served_model_name: champion
              traffic_percentage: 100
      ai_gateway:
        inference_table_config:
          catalog_name: ${var.catalog}
          schema_name: ${var.schema}
          table_name_prefix: fraud_serving
          enabled: true
```

Pinning `entity_version` in YAML gives you a reviewable, auditable change for every
model release (a PR that bumps "3" to "4"). The notebook path (`06_deploy_serving`,
alias-driven) gives you automation. Pick one per environment and write it down:
a common choice is notebook-driven in dev/staging, YAML-pinned with PR review in prod.

### 5.9 Build: Governance and cost checklist

Run through these in your workspace and tick them off:

**Permissions (Unity Catalog)**

```sql
GRANT USE CATALOG ON CATALOG main TO `fraud-analysts`;
GRANT USE SCHEMA, SELECT ON SCHEMA main.fraud TO `fraud-analysts`;
GRANT SELECT ON TABLE main.fraud.gold_transaction_scores TO `fraud-analysts`;
GRANT EXECUTE ON FUNCTION main.fraud.amount_ratio TO `sp-fraud-prod`;
-- Models
GRANT EXECUTE ON MODEL main.fraud.fraud_classifier TO `sp-fraud-prod`;
-- Nobody but the prod SP writes feature tables
REVOKE MODIFY ON TABLE main.fraud.customer_txn_features FROM `data-scientists`;
```

**Lineage and audit.** Catalog Explorer → model version → Lineage: tables, functions,
notebook, job. System tables (`system.access.audit`, `system.billing.usage`,
`system.lakeflow.job_run_timeline`) answer "who ran what when" and "what did it cost."

**Cost controls**

- Serverless everywhere this guide used it; for classic clusters, enforce **cluster
  policies** (max workers, autotermination, allowed instance types).
- Scale-to-zero on non-prod endpoints; provisioned concurrency sized from the Day 4
  load test in prod.
- Query `system.billing.usage` grouped by `usage_metadata.job_id` weekly; put it on
  your dashboard.
- Liquid clustering + `OPTIMIZE` on large tables; `VACUUM` with a retention that
  still permits the time travel you need for reproducibility (default 7 days; set
  `delta.deletedFileRetentionDuration` longer on training tables).

**Documentation.** Write `docs/model_card.md` and `docs/runbook.md` as in week one,
now including: endpoint name, inference table, monitor dashboard link, alert names,
the retraining gate, the rollback commands, the service principals, and the cost
baseline.

### 5.10 Build: The incident drill (capstone)

Without looking back at the sections:

1. Generate two days of drifted data and send it through the pipeline and the endpoint.
2. Confirm the alert fires (or the drift query returns rows).
3. Investigate with the monitor dashboard: which features drifted, by how much, and did
   quality degrade on labeled rows?
4. Decide: pipeline bug or real drift? Record the evidence.
5. Run the monitoring job and watch the gate open, the retrain run, and the promotion
   decide.
6. If promoted, confirm the endpoint serves the new version. If not, explain why from
   the promotion logs.
7. Roll back deliberately to `previous`, verify, then roll forward again.
8. Write `docs/incidents/<date>-mobile-fraud-shift.md`: timeline, detection, decision,
   actions, what to automate next.

If you can do all eight, you can run an ML system on Databricks in production.

### 5.11 Exercises

1. **Slice monitoring.** Use the `slicing_exprs` output to find the country where
   recall is worst. Propose a fix (per-country threshold? per-country model from Day
   3?) and implement the threshold version via a small config table read by the
   scoring job.
2. **Integration test as a bundle job.** Add `fraud_integration_test`: a job that runs
   the full chain on a 1-day sample schema and asserts row counts, PR-AUC above a floor,
   and a successful endpoint query. Run it in `deploy-staging`.
3. **Cross-environment model promotion.** Implement the register-from-staging pattern
   from 5.6 as a `promote_to_prod` workflow with a manual approval step.
4. **Cost report.** A SQL query on `system.billing.usage` that shows this project's
   spend per job and per endpoint for the last 30 days. Add it to the dashboard.
5. **Access review.** Produce the list of principals with any privilege on
   `main.fraud.*` from `system.information_schema.table_privileges`. Is anything
   surprising?

### 5.12 Checkpoint questions

1. What four things does an inference profile monitor compute, and which require labels?
2. Why do you need `transaction_id` in the serving request, and what breaks without it?
3. Why does the monitoring job use a condition task instead of an `if` in a notebook?
4. What is the argument for and against automatic redeploy after retraining in a fraud system?
5. Explain how the same bundle deploys to three environments. What changes, and what must not?
6. What is a service principal, and name three things in your project that should run as one.
7. Trace a single blocked transaction from the alert to the training data version. List every table and object on the path.

---

## Capstone Checklist

**Lakehouse and ingestion**
- [ ] Raw files land in a UC volume; Auto Loader ingests incrementally
- [ ] Bronze/silver/gold declarative pipeline with expectations; event log queried
- [ ] Silver tables liquid-clustered on lookup keys; CDF enabled
- [ ] Delta versions of all training inputs recorded in each MLflow run

**Features**
- [ ] Time-series feature table with PIT semantics; leakage unit test passes
- [ ] Profile feature table; three on-demand feature functions in UC
- [ ] Training set assembled with lookups and functions; PIT correctness proven by hand
- [ ] Lineage visible from bronze to features to model

**Training**
- [ ] Time-based split; PR-AUC primary; cost-based threshold logged and tagged
- [ ] Optuna trials as nested runs; parent run with best params and test metrics
- [ ] Model logged with feature spec and registered in UC with aliases
- [ ] Champion/challenger decision code with unit tests; promotion task in a job
- [ ] Five-task training job deployed from the bundle

**Serving**
- [ ] Batch scoring job writes a merged scores table; dashboard built on it
- [ ] Feature tables published online; endpoint deployed with automatic lookup
- [ ] Request contains only raw fields plus key; inference table capturing traffic
- [ ] Load test numbers recorded; traffic split and rollback exercised

**Operations**
- [ ] Inference table unpacked and joined with delayed labels
- [ ] Monitor (managed or SQL) with drift and quality; alerts configured
- [ ] Monitoring job with condition-task gate triggering retraining and redeploy
- [ ] Bundle targets for dev/staging/prod; GitHub Actions deploy with service principals
- [ ] UC grants reviewed; cost query in place; model card, runbook, incident report written
- [ ] Incident drill completed

---

## Mapping Databricks to What You Learned in Week 1

| Week 1 tool | Databricks equivalent | What changed |
|---|---|---|
| DVC data versioning | Delta time travel + logged table versions | No separate tool; versioning is a property of every table |
| DVC pipeline / Makefile | Lakeflow Declarative Pipelines + Lakeflow Jobs | DAG inferred from table reads; orchestration, retries, and repair built in |
| pandera schema | Pipeline expectations | Declarative, tracked over time in the event log |
| `build_features` function | Feature tables + feature functions in UC | PIT joins, online sync, and packaging into the model |
| Local MLflow server | Workspace MLflow + Models in UC | Permissions, lineage, cross-workspace sharing |
| `train.py` + `registry.py` | Training notebook + promotion task | Same logic, now a governed job |
| FastAPI + Docker | Model Serving endpoint | No container to build; automatic feature lookup; traffic splits |
| Prediction log file | AI Gateway inference table | Automatic, queryable, permissioned |
| Evidently + Prometheus | Lakehouse Monitoring + SQL alerts | Managed metrics tables and dashboard |
| GitHub Actions + GHCR | GitHub Actions + Asset Bundles | Deploys jobs, pipelines, and endpoints, not just an image |
| Runbook / model card | Same | Plus UC lineage and system tables as evidence |

The *concepts* are identical. What a platform buys you is integration and governance;
what it costs is vendor-specific APIs and the discipline to keep logic in plain Python
packages so it stays portable.

---

## What to Learn Next

1. **Streaming end to end.** Make the feature pipeline continuous (`continuous: true`),
   add a Kafka or Event Hubs source with Lakeflow Connect, and measure feature
   freshness at the endpoint. Fraud teams live here.
2. **Lakebase and synced tables** in depth: transactional serving, reverse ETL, and how
   online features become part of an OLTP system.
3. **Mosaic AI for LLM workloads.** Agent Framework, Vector Search, AI Gateway with
   external models, and MLflow evaluation and tracing for GenAI. Same lifecycle, new
   artifacts (prompts, retrieval indexes, evals).
4. **Multi-workspace and multi-cloud governance.** Delta Sharing, catalog binding,
   metastore design, and attribute-based access control.
5. **Terraform for Databricks.** Bundles manage project resources; Terraform manages the
   workspace itself (catalogs, groups, SPs, policies, networking). Together they are
   full infrastructure as code.
6. **Performance engineering.** Photon, predictive optimization, Spark UI diagnosis,
   skew handling, and when `applyInPandas` beats Spark-native operations.
7. **The Databricks certifications.** Machine Learning Associate then Professional map
   closely to this week's content; the Data Engineer Associate covers Day 1 in depth.

Reading: Databricks' "The Big Book of MLOps" (free PDF; the reference architecture it
describes is what you built), the Delta Lake Definitive Guide, and the MLflow 3
documentation on Models in Unity Catalog.

---

## Glossary

| Term | Meaning |
|---|---|
| **AS OF join** | A join that, for each left row at time T, picks the right row with the greatest timestamp ≤ T |
| **Asset Bundle (DAB)** | YAML + code describing every deployable resource of a project; deployed per target |
| **Auto Loader** | Incremental file ingestion source (`cloudFiles`) with checkpointing and schema evolution |
| **Champion / challenger** | Aliases for the serving model and its candidate replacement |
| **Condition task** | A job task that branches the DAG on a boolean expression |
| **Delta Lake** | Open table format with ACID transactions, time travel, and schema enforcement |
| **Expectation** | A declarative data quality rule on a pipeline table |
| **Feature function** | A UC Python UDF computed on demand during training-set creation and serving |
| **Feature table** | A Delta table with declared primary (and optional timestamp) keys, registered for lookups |
| **Inference table** | Automatic log of endpoint requests and responses as a Delta table |
| **Lakeflow Declarative Pipelines** | Managed framework for defining tables as functions of other tables (formerly Delta Live Tables) |
| **Lakeflow Jobs** | Orchestrator for multi-task DAGs of notebooks, pipelines, SQL, and other jobs (formerly Workflows) |
| **Lakehouse Monitoring** | Managed profiling, drift, and quality metrics on a table, with a dashboard |
| **Liquid clustering** | Adaptive data layout by chosen columns, replacing partitioning and Z-order |
| **Medallion** | Bronze (raw) → silver (clean) → gold (business) data tiers |
| **Models in Unity Catalog** | The registry: models as `catalog.schema.name` with versions, aliases, grants, and lineage |
| **Online table / synced table** | Low-latency copy of a feature table for serving-time lookups |
| **Point-in-time (PIT)** | Computing a feature using only data available at the example's timestamp |
| **Service principal** | A non-human identity that owns and runs production resources |
| **Serverless** | Compute managed entirely by Databricks; no clusters to configure |
| **Task value** | A small value passed from one job task to the next |
| **Time travel** | Querying a Delta table as of a prior version or timestamp |
| **Unity Catalog** | The governance layer: three-level namespace, permissions, lineage, audit for all objects |
| **Volume** | A UC-governed location for non-tabular files |

---

## Appendix A: Full Bundle Reference

Final `databricks.yml` (variables and targets consolidated):

```yaml
bundle:
  name: fraud-mlops

include:
  - resources/*.yml

variables:
  catalog: { default: main }
  schema: { default: fraud_dev }
  model_name: { default: fraud_classifier }
  endpoint_name: { default: fraud-classifier }
  warehouse_id: { description: SQL warehouse for SQL tasks }

targets:
  dev:
    mode: development
    default: true
    workspace: { host: https://YOUR-WORKSPACE-URL }
    variables: { warehouse_id: REPLACE }

  staging:
    mode: production
    workspace:
      host: https://YOUR-WORKSPACE-URL
      root_path: /Workspace/Shared/.bundle/${bundle.name}/${bundle.target}
    variables: { schema: fraud_staging, endpoint_name: fraud-classifier-staging, warehouse_id: REPLACE }
    run_as: { service_principal_name: REPLACE_STAGING_SP }

  prod:
    mode: production
    workspace:
      host: https://YOUR-WORKSPACE-URL
      root_path: /Workspace/Shared/.bundle/${bundle.name}/${bundle.target}
    variables: { schema: fraud, warehouse_id: REPLACE }
    run_as: { service_principal_name: REPLACE_PROD_SP }
    resources:
      jobs:
        fraud_training:
          schedule: { quartz_cron_expression: "0 0 3 ? * MON", timezone_id: UTC, pause_status: UNPAUSED }
        fraud_batch_scoring:
          schedule: { quartz_cron_expression: "0 30 1 * * ?", timezone_id: UTC, pause_status: UNPAUSED }
        fraud_monitoring:
          schedule: { quartz_cron_expression: "0 0 */6 * * ?", timezone_id: UTC, pause_status: UNPAUSED }
      model_serving_endpoints:
        fraud_endpoint:
          config:
            served_entities:
              - name: champion
                entity_name: main.fraud.fraud_classifier
                entity_version: "1"
                workload_size: Medium
                scale_to_zero_enabled: false
```

Useful CLI commands:

```bash
databricks bundle validate -t <target>
databricks bundle deploy   -t <target>
databricks bundle run <resource_key> -t <target>
databricks bundle summary  -t <target>           # what is deployed and where
databricks bundle destroy  -t dev                # tear down your dev copy
databricks jobs list --output json | jq '.[].settings.name'
databricks serving-endpoints get fraud-classifier
```

Suggested `tests/test_evaluation.py` and `tests/test_promotion.py` cover:
`choose_threshold` returns a threshold minimizing cost on a toy array; `decide` promotes
when no champion exists, declines on small deltas, and promotes on large ones (mock
`MlflowClient` with `unittest.mock`).

---

## Appendix B: Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PERMISSION_DENIED` creating catalog/schema | No metastore admin or `CREATE CATALOG` grant | Use an existing catalog; ask an admin for `CREATE SCHEMA` on it |
| Bundle deploy: "run_as is required in production mode" | Staging/prod target lacks `run_as` | Add the service principal's application ID |
| Pipeline fails: `cloudFiles` schema mismatch | Explicit schema differs from files | Fix the schema or use inference with `schemaEvolutionMode` |
| `dropDuplicates` on streaming errors about state | Missing watermark | `withWatermark` before `dropDuplicates` |
| `create_table` rejects `timeseries_columns` | Older client version | Use `timestamp_keys=`; or upgrade `databricks-feature-engineering` |
| Training set has nulls for all behavior features | Lookup key mismatch or feature_ts > event_ts for every row | Check dtypes of `customer_id` and that `feature_ts` is the event time |
| PR-AUC suspiciously high (> 0.95) | Leakage: window includes current event, or merchant fraud rate uses `event_ts` | Re-check `rangeBetween(..., -1)` and label-time logic |
| `fe.log_model` fails with "function not found" | Feature function in a different schema than expected | Use fully qualified `catalog.schema.function` in `udf_name` |
| Endpoint stuck in `UPDATE_FAILED` | Model dependencies missing, or online table not found | Check build logs; confirm online tables exist for every lookup table |
| Endpoint returns 400 "missing column" | Request lacks a lookup key or raw feature | Compare the request with the model signature in the MLmodel file |
| Inference table empty after requests | AI Gateway config not applied, or lag | Wait a few minutes; check `put_ai_gateway` succeeded; confirm table name in schema |
| Monitor create fails on baseline | Column names differ between baseline and monitored table | Recreate `training_baseline` with matching names |
| Condition task never opens | Task value not set or wrong type | Set a string `"true"`; reference `{{tasks.<key>.values.<name>}}` exactly |
| GitHub Action: `default auth: cannot configure` | Env vars missing in that job | Add `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` at job level |
| Serverless job fails importing `fraud` | `sys.path.append("../")` wrong relative to bundle root | Check the deployed path in `bundle summary`; append the `src` directory explicitly |

---

*End of curriculum. Combined with week one you now have two portfolio projects: one
built from open-source parts, one built on a governed platform. Being able to explain
the mapping between them is exactly what senior MLOps interviews probe.*
