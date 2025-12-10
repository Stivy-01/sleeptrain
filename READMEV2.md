# SleepTrain v2 — End-to-End Guide

This guide explains setup, configuration, testing, benchmarks, and the new training/evaluation workflow so anyone can reproduce results and extend the framework.

## 1) Prerequisites
- Python 3.11+ recommended.
- Postgres running and reachable (local or remote). Optional: `pgvector` extension for faster vector search.
- git (optional), PowerShell/CMD or bash.

## 2) Install Python dependencies
```bash
pip install psycopg2-binary sentence-transformers torch certifi
# Optional:
# pip install wandb
```

## 3) Environment variables
Required:
- `COCOINDEX_DB_URL` — e.g. `postgresql://sleeptrain_user:yourpassword@localhost:5432/sleeptrain`
- `COCOINDEX_EMBED_MODEL` — default `all-MiniLM-L6-v2`
- `REQUESTS_CA_BUNDLE` — set to a valid CA bundle (recommended: `python -c "import certifi; print(certifi.where())"`)

Optional:
- `COCOINDEX_EMBED_DIM` — default 384; must match the embedding model dim.
- `WANDB_*` — if using WandB logging.

## 4) Postgres setup (local example)
In psql as postgres (adjust port/user/password as needed):
```sql
CREATE DATABASE sleeptrain;
CREATE USER sleeptrain_user WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE sleeptrain TO sleeptrain_user;
\c sleeptrain
-- Optional if pgvector is installed:
-- CREATE EXTENSION IF NOT EXISTS vector;
GRANT ALL ON SCHEMA public TO sleeptrain_user;
ALTER DATABASE sleeptrain OWNER TO sleeptrain_user;
```
Test connectivity:
```bash
psql $COCOINDEX_DB_URL -c "SELECT 1;"
```

## 5) Reproducible config (JSON)
`configs/repro_config.json` contains default placeholders for:
- DB URL, embedding model, CA bundle,
- Benchmark paths,
- WandB toggles,
- Hedge parameters.
Update the values to your environment for reproducible runs.

## 6) Core components (what changed in v2)
- Persistent memory store: `scripts/memory/coco_memory_flow.py` (Postgres-backed, optional pgvector).
- Hippocampus integration: `scripts/training/hippocampus.py` uses the new memory flow.
- Hedge RL policy: `scripts/rl/hedge_hippocampus.py` and `scripts/training/train_loop.py` (STORE/REJECT/CORRECT with persistence and logging hooks).
- Benchmarks: `scripts/evaluation/benchmarks.py` (TRACE/MemoryBench/BABILong/InfiniteBench loaders + runner, optional WandB logging).
- Reporting: `scripts/analysis/generate_benchmark_report.py` (HTML from results JSON).
- Release artifacts: `docs/v2_release_notes.md`, `docs/colab_notebook_stub.md`, `results/benchmarks/leaderboard.json`, `configs/benchmark_paths.yaml`.

## 7) Running tests
General:
```bash
python -m py_compile scripts/**/*.py tests/**/*.py
```
DB-backed test (requires COCOINDEX_DB_URL and dependencies):
```bash
export COCOINDEX_DB_URL="postgresql://sleeptrain_user:yourpassword@localhost:5432/sleeptrain"
python -m pytest tests/test_coco_memory_flow.py -vv
```
Hedge tests:
```bash
python -m pytest tests/test_hedge_policy.py -vv
```
Other tests (if any) can be run similarly with pytest.

## 8) Using the memory flow
```python
from scripts.memory.coco_memory_flow import COCOIndexMemoryFlow
import os

flow = COCOIndexMemoryFlow(db_url=os.environ["COCOINDEX_DB_URL"])
flow.upsert({"person_id": "demo", "fact": "Demo fact", "importance": 7, "type": "fact"})
print(flow.query("Demo fact", top_k=3, person_id="demo"))
```
Notes:
- Real embeddings are required (sentence-transformers). No fallback.
- If pgvector is installed, it is auto-used; otherwise, Python ranking is used.

## 9) Hedge RL in a training loop
```python
from scripts.rl.hedge_hippocampus import HedgePolicy
from scripts.training.train_loop import HedgeTrainer, HedgeConfig

def action_fn(action_name: str):
    # Perform action (STORE/REJECT/CORRECT) and return metrics
    return {"delta_metric": 0.1 if action_name == "STORE" else 0.0}

def reward_fn(metrics):
    return metrics["delta_metric"]

trainer = HedgeTrainer(HedgeConfig(routing_bias="store-heavy"), action_fn, reward_fn)
log = trainer.step()
print(log)
```
Persistence: `policy.save(path)` / `HedgePolicy.load(path)` handle weights/eta/strict/reward_clip.

## 10) Benchmarks
1) Ensure dataset paths in `configs/benchmark_paths.yaml` (JSONL with at least `{"input":..., "target":...}` per line).
2) Load and run:
```python
from scripts.evaluation.benchmarks import TRACEBenchLoader, run_benchmark

loader = TRACEBenchLoader("data/benchmarks/trace.jsonl")
ds = loader.load()

def model_fn(example):  # dummy example
    return "prediction"

def metric_fn(pred, example):
    # user-defined scoring, e.g., exact match
    return float(pred == example.get("target"))

result = run_benchmark(model_fn, ds, metric_fn, wandb_log=False)
print(result)
```
3) Log to WandB: set `wandb_log=True` and ensure `wandb.init()` is active.
4) Update leaderboard: edit `results/benchmarks/leaderboard.json` with new scores.

## 11) Reports
Generate HTML from results JSON:
```bash
python -m scripts.analysis.generate_benchmark_report --results results.json --out results/benchmarks/report.html
```
Where `results.json` is a list of run dicts like:
```json
[{"benchmark": "TRACE", "split": "test", "avg_score": 0.72, "count": 100, "path": "data/benchmarks/trace.jsonl"}]
```

## 12) Colab usage
See `docs/colab_notebook_stub.md` for a minimal Colab cell. Set `COCOINDEX_DB_URL` to a reachable Postgres instance (via tunnel/public endpoint). Ensure `REQUESTS_CA_BUNDLE` is set (certifi recommended).

## 13) Release notes
See `docs/v2_release_notes.md` for highlights, setup, modules, and next steps.

## 14) Optional: pgvector install
If you want ANN speedups:
```sql
-- In the target DB, as superuser
CREATE EXTENSION IF NOT EXISTS vector;
```
No code changes required; the flow auto-detects pgvector.

## 15) Troubleshooting
- `psql not found`: add Postgres bin to PATH, or invoke full path (e.g., `"C:\Program Files\PostgreSQL\18\bin\psql.exe"`).
- TLS/CA errors on model download: set `REQUESTS_CA_BUNDLE` to `certifi.where()`.
- Permission errors creating tables: ensure `GRANT ALL ON SCHEMA public TO sleeptrain_user` and `ALTER DATABASE sleeptrain OWNER TO sleeptrain_user`.
- Tests skipping: set `COCOINDEX_DB_URL` and install `psycopg2-binary`, `sentence-transformers`.

You now have a full, reproducible pipeline: Postgres-backed memory store, Hedge RL control, benchmark loaders/runner, reporting, release notes, and a Colab stub. Update `repro_config.json` and env vars for your environment, then run tests/benchmarks as needed.
