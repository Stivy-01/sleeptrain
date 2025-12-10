# SleepTrain v2 — End-to-End Guide

This guide explains setup, configuration, testing, benchmarks, and the new training/evaluation workflow so anyone can reproduce results and extend the framework.

## What's New in v2.1 (SOTA Extensions)

This release adds several state-of-the-art mechanisms:

| Feature | Module | Description |
|---------|--------|-------------|
| **SEAL Loop** | `scripts/training/seal_loop.py` | ReST-EM-style closed-loop self-training with reward weighting |
| **Routing Replay** | `scripts/rl/hedge_hippocampus.py` | Multiplicative weights on paths with replay buffer and probability clipping |
| **Memento Rewrite** | `scripts/memory/memento_rewrite.py` | Promote/evict/merge policy for memory consolidation |
| **SPICE Challenger** | `scripts/training/spice_challenger.py` | Corpus-grounded challenger mining hard contradictions |
| **Dual Backends** | `scripts/memory/local_backend.py` | Local/FAISS backend as Postgres alternative |
| **WandB Integration** | `scripts/analysis/wandb_leaderboard.py` | WandB logging + HTML leaderboard generation |

## 1) Prerequisites
- Python 3.11+ recommended.
- Postgres running and reachable (local or remote). Optional: `pgvector` extension for faster vector search.
- **NEW**: Local backend available (no Postgres required for development).
- git (optional), PowerShell/CMD or bash.

## 2) Install Python dependencies
```bash
pip install psycopg2-binary sentence-transformers torch certifi
# Optional:
pip install wandb  # for experiment tracking
pip install faiss-cpu  # for fast local similarity search
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
- **Local backend**: `scripts/memory/local_backend.py` (in-memory/file-backed with optional FAISS).
- Hippocampus integration: `scripts/training/hippocampus.py` uses the new memory flow.
- Hedge RL policy: `scripts/rl/hedge_hippocampus.py` and `scripts/training/train_loop.py` (STORE/REJECT/CORRECT with persistence, **routing replay**, and logging hooks).
- **SEAL Loop**: `scripts/training/seal_loop.py` (reward-weighted self-training).
- **Memento Rewrite**: `scripts/memory/memento_rewrite.py` (promote/evict/merge policy).
- **SPICE Challenger**: `scripts/training/spice_challenger.py` (hard example mining).
- Benchmarks: `scripts/evaluation/benchmarks.py` (TRACE/MemoryBench/BABILong/InfiniteBench loaders + runner, optional WandB logging).
- Reporting: `scripts/analysis/generate_benchmark_report.py` (HTML from results JSON).
- **WandB + Leaderboard**: `scripts/analysis/wandb_leaderboard.py` (experiment tracking + HTML leaderboard).
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
from scripts.rl.hedge_hippocampus import HedgePolicy, RoutingConfig
from scripts.training.train_loop import HedgeTrainer, HedgeConfig

def action_fn(action_name: str):
    # Perform action (STORE/REJECT/CORRECT) and return metrics
    return {"delta_metric": 0.1 if action_name == "STORE" else 0.0}

def reward_fn(metrics):
    return metrics["delta_metric"]

# NEW: Configure routing replay and probability clipping
config = HedgeConfig(
    routing_bias="store-heavy",
    enable_routing_replay=True,      # Enable replay buffer
    replay_every_n_steps=10,         # Replay every 10 steps
    replay_sample_size=5,            # 5 samples per replay
    clip_min=0.05,                   # Min probability floor
    clip_max=0.90,                   # Max probability ceiling
)

trainer = HedgeTrainer(config, action_fn, reward_fn)
log = trainer.step()
print(log)
print(trainer.get_replay_stats())  # NEW: replay buffer stats
```
Persistence: `policy.save(path)` / `HedgePolicy.load(path)` handle weights/eta/strict/reward_clip and **replay buffer**.

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

## 16) SEAL Loop (Self-Alignment)
Run reward-weighted self-training:
```python
from scripts.training.seal_loop import SEALLoop, SEALConfig

def model_fn(prompt):
    return "generated response"

def reward_fn(prompt, response, context):
    return 0.8  # score response

def train_fn(samples):
    # Fine-tune on filtered samples
    return {"loss": 0.1}

seal = SEALLoop(
    model_fn=model_fn,
    reward_fn=reward_fn,
    train_fn=train_fn,
    config=SEALConfig(
        num_candidates=4,           # candidates per prompt
        max_iterations=5,           # EM iterations
        reward_threshold=0.5,       # minimum reward to include
        use_reward_weighting=True,  # weight by reward in loss
    ),
)
result = seal.run(prompts=["prompt1", "prompt2"])
```

## 17) Memento Rewrite Policy
Manage memory consolidation with promote/evict/merge:
```python
from scripts.memory.memento_rewrite import MementoRewritePolicy, MementoConfig

memento = MementoRewritePolicy(config=MementoConfig(
    evict_threshold=2.0,             # evict if importance below
    promote_threshold=7.0,           # promote if above
    merge_similarity_threshold=0.85, # merge if similarity above
))

records = [{"id": "1", "fact": "...", "embedding": emb, "importance": 5, ...}]
actions = memento.batch_evaluate(records)
summary = memento.apply_rewrite(memory_flow, "person_id", actions)
```

## 18) SPICE Challenger
Mine hard examples for training:
```python
from scripts.training.spice_challenger import SPICEChallenger, ChallengerConfig, default_corpus_paths

challenger = SPICEChallenger(config=ChallengerConfig(
    corpus_paths=default_corpus_paths(),
    min_difficulty=0.3,
))
challenger.load_corpus()

# Mine contradictions from existing facts
existing_facts = [{"fact": "Obama born 1961", "person_id": "obama"}]
contradictions = challenger.mine_contradictions(existing_facts)

# Sample challenges for SEAL loop
samples = challenger.sample_challenges(num_examples=10)
seal_format = challenger.to_seal_format(samples)
```

## 19) Local Backend (No Postgres)
Use local/FAISS backend for development:
```python
from scripts.memory.local_backend import LocalMemoryStore, create_memory_backend, BackendConfig

# Option 1: Direct local store
store = LocalMemoryStore(
    persist_path="data/local_memory.json",  # optional file persistence
    use_faiss=True,                          # use FAISS if available
)
store.upsert({"person_id": "demo", "fact": "Demo fact", "importance": 7})

# Option 2: Auto-detect backend (Postgres if COCOINDEX_DB_URL set, else local)
backend = create_memory_backend(BackendConfig(
    backend_type="auto",  # "auto", "postgres", or "local"
    persist_path="data/memory.json",
))
```

## 20) WandB + Leaderboard
Log experiments and update leaderboard:
```python
from scripts.analysis.wandb_leaderboard import (
    WandBLogger, WandBConfig, LeaderboardManager, log_benchmark_result
)

# WandB logging
logger = WandBLogger(WandBConfig(project="sleeptrain"))
logger.init(run_name="experiment_1", config={"lr": 1e-4})
logger.log({"loss": 0.5, "accuracy": 0.9})
logger.finish()

# Update leaderboard
log_benchmark_result(
    benchmark="TRACE",
    model="qwen2.5-7b-lora",
    avg_score=0.85,
    count=100,
    notes="With SEAL loop",
    update_leaderboard=True,
)

# Generate HTML leaderboard
manager = LeaderboardManager()
manager.generate_html()  # -> results/benchmarks/leaderboard.html
```

---

You now have a full, reproducible pipeline: Postgres-backed or local memory store, Hedge RL control with routing replay, SEAL self-training loop, Memento memory management, SPICE challenger, benchmark loaders/runner, WandB integration, reporting, release notes, and a Colab stub. Update `repro_config.json` and env vars for your environment, then run tests/benchmarks as needed.
