"""
Unified benchmark loaders and runners for external suites.

Supported loaders:
- TRACE
- MemoryBench
- BABILong
- InfiniteBench

Each loader reads JSONL where each line is a dict containing at least:
    {"input": ..., "target": ...}

run_benchmark(model_fn, dataset, metric_fn) provides a lightweight loop to
standardize evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class BenchmarkDataset:
    name: str
    split: str
    path: Path
    items: List[Dict[str, Any]]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
class BaseLoader:
    def __init__(self, path: str, name: str, split: str = "test") -> None:
        self.path = Path(path)
        self.name = name
        self.split = split

    def load(self) -> BenchmarkDataset:
        items = _load_jsonl(self.path)
        return BenchmarkDataset(name=self.name, split=self.split, path=self.path, items=items)


class TRACEBenchLoader(BaseLoader):
    def __init__(self, path: str, split: str = "test") -> None:
        super().__init__(path, name="TRACE", split=split)


class MemoryBenchLoader(BaseLoader):
    def __init__(self, path: str, split: str = "test") -> None:
        super().__init__(path, name="MemoryBench", split=split)


class BabilongLoader(BaseLoader):
    def __init__(self, path: str, split: str = "test") -> None:
        super().__init__(path, name="BABILong", split=split)


class InfiniteBenchLoader(BaseLoader):
    def __init__(self, path: str, split: str = "test") -> None:
        super().__init__(path, name="InfiniteBench", split=split)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
ModelFunc = Callable[[Dict[str, Any]], Any]
MetricFunc = Callable[[Any, Dict[str, Any]], float]


def run_benchmark(
    model_fn: ModelFunc,
    dataset: BenchmarkDataset,
    metric_fn: MetricFunc,
    limit: Optional[int] = None,
    wandb_log: bool = False,
    wandb_run: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute a benchmark with a provided model and metric function.

    model_fn:  takes one example dict, returns a prediction/answer
    metric_fn: takes (prediction, example) -> float score
    limit:     optional limit on number of examples
    wandb_log: if True, log avg_score and count to wandb_run (or wandb.run)
    """
    items = dataset.items if limit is None else dataset.items[:limit]
    scores: List[float] = []
    for ex in items:
        pred = model_fn(ex)
        score = metric_fn(pred, ex)
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    result = {
        "benchmark": dataset.name,
        "split": dataset.split,
        "path": str(dataset.path),
        "count": len(scores),
        "avg_score": avg,
        "scores": scores,
    }

    if wandb_log:
        try:
            import wandb  # type: ignore

            run = wandb_run or wandb.run
            if run is not None:
                run.log(
                    {
                        f"bench/{dataset.name}/{dataset.split}/avg_score": avg,
                        f"bench/{dataset.name}/{dataset.split}/count": len(scores),
                    }
                )
        except Exception:
            # Keep evaluation resilient even if wandb is not available
            pass

    return result


__all__ = [
    "BenchmarkDataset",
    "TRACEBenchLoader",
    "MemoryBenchLoader",
    "BabilongLoader",
    "InfiniteBenchLoader",
    "run_benchmark",
]
