# SleepTrain v2 Release Notes

## Highlights
- Production-ready COCOIndex flow backed by Postgres (optional pgvector), with logging and tests.
- Hedge RL policy integrated for hippocampus actions (STORE / REJECT / CORRECT) with persistence and logging hooks.
- Benchmark loaders and runner for TRACE, MemoryBench, BABILong, InfiniteBench, plus HTML report generator and leaderboard scaffold.

## Setup
- Env vars: `COCOINDEX_DB_URL` (Postgres), `COCOINDEX_EMBED_MODEL` (default `all-MiniLM-L6-v2`), `REQUESTS_CA_BUNDLE` (use `certifi.where()`), optional `COCOINDEX_EMBED_DIM` and pgvector if installed.
- Python deps: `psycopg2-binary`, `sentence-transformers`, `torch`, `certifi`, optional `wandb`.

## Notable Modules
- `scripts/memory/coco_memory_flow.py`: persistent memory store (upsert/query/diff/logging).
- `scripts/training/hippocampus.py`: integrated with the new memory flow.
- `scripts/rl/hedge_hippocampus.py`, `scripts/training/train_loop.py`: Hedge policy and trainer.
- `scripts/evaluation/benchmarks.py`: loaders + runner, wandb logging hook.
- `scripts/analysis/generate_benchmark_report.py`: HTML report from results JSON.

## Benchmarks
- Config paths: `configs/benchmark_paths.yaml`
- Leaderboard scaffold: `results/benchmarks/leaderboard.json`
- Generate HTML: `python -m scripts.analysis.generate_benchmark_report --results <results.json> --out results/benchmarks/report.html`
- Optional WandB: pass `wandb_log=True` (and have a `wandb.init()` run active).

## Next Steps
- Run benchmark suites and fill leaderboard.
- Optional pgvector install for faster ANN search.
- Optional Colab example and reproducible configs for public release.
