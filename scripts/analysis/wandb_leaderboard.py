"""
WandB logging and leaderboard integration for SleepTrain experiments.

Features:
- WandB logging for training sweeps and benchmarks
- Leaderboard update script (updates results/benchmarks/leaderboard.json)
- HTML leaderboard publisher
- Integration hooks for notebooks and CLI
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


LEADERBOARD_PATH = Path("results/benchmarks/leaderboard.json")
HTML_TEMPLATE_PATH = Path("results/benchmarks/leaderboard.html")


@dataclass
class LeaderboardEntry:
    """Single entry in the leaderboard."""
    benchmark: str
    model: str
    avg_score: Optional[float]
    count: int
    notes: str = ""
    timestamp: Optional[str] = None
    run_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "avg_score": self.avg_score,
            "count": self.count,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeaderboardEntry":
        return cls(
            benchmark=data.get("benchmark", ""),
            model=data.get("model", ""),
            avg_score=data.get("avg_score"),
            count=data.get("count", 0),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp"),
            run_id=data.get("run_id"),
            config=data.get("config", {}),
        )


@dataclass
class WandBConfig:
    """Configuration for WandB logging."""
    project: str = "sleeptrain"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    group: Optional[str] = None
    job_type: Optional[str] = None
    notes: Optional[str] = None
    mode: str = "online"  # "online", "offline", "disabled"


class WandBLogger:
    """
    WandB logging wrapper for SleepTrain experiments.

    Usage:
        logger = WandBLogger(config=WandBConfig(project="sleeptrain"))
        logger.init(run_name="experiment_1", config={"lr": 1e-4})
        logger.log({"loss": 0.5, "accuracy": 0.9})
        logger.finish()
    """

    def __init__(self, config: Optional[WandBConfig] = None):
        self.config = config or WandBConfig()
        self.run: Optional[Any] = None
        self._initialized = False

    def init(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume: bool = False,
    ) -> Optional[str]:
        """
        Initialize a WandB run.

        Args:
            run_name: Name for the run.
            config: Run configuration dict.
            resume: Whether to resume an existing run.

        Returns:
            Run ID if successful, None otherwise.
        """
        if not WANDB_AVAILABLE:
            print("[WandB] wandb not installed, logging disabled")
            return None

        if self.config.mode == "disabled":
            print("[WandB] Logging disabled by config")
            return None

        try:
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=run_name,
                config=config,
                tags=self.config.tags,
                group=self.config.group,
                job_type=self.config.job_type,
                notes=self.config.notes,
                mode=self.config.mode,
                resume=resume,
            )
            self._initialized = True
            print(f"[WandB] Run initialized: {self.run.name} ({self.run.id})")
            return self.run.id
        except Exception as e:
            print(f"[WandB] Init failed: {e}")
            return None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB."""
        if not self._initialized or not self.run:
            return
        try:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            print(f"[WandB] Log failed: {e}")

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an artifact (file/directory) to WandB."""
        if not self._initialized or not self.run:
            return
        try:
            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            if Path(path).is_dir():
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            self.run.log_artifact(artifact)
            print(f"[WandB] Artifact logged: {name}")
        except Exception as e:
            print(f"[WandB] Artifact log failed: {e}")

    def log_table(self, name: str, columns: List[str], data: List[List[Any]]) -> None:
        """Log a table to WandB."""
        if not self._initialized or not self.run:
            return
        try:
            table = wandb.Table(columns=columns, data=data)
            self.run.log({name: table})
        except Exception as e:
            print(f"[WandB] Table log failed: {e}")

    def finish(self) -> None:
        """Finish the WandB run."""
        if self._initialized and self.run:
            try:
                wandb.finish()
                print(f"[WandB] Run finished: {self.run.name}")
            except Exception as e:
                print(f"[WandB] Finish failed: {e}")
        self._initialized = False
        self.run = None

    def get_run_url(self) -> Optional[str]:
        """Get the URL for the current run."""
        if self.run:
            return self.run.url
        return None


class LeaderboardManager:
    """
    Manages the benchmark leaderboard.

    Features:
    - Load/save leaderboard JSON
    - Add/update entries
    - Generate HTML report
    - Integration with WandB
    """

    def __init__(self, leaderboard_path: Optional[Path] = None):
        self.leaderboard_path = leaderboard_path or LEADERBOARD_PATH
        self.entries: List[LeaderboardEntry] = []
        self._load()

    def _load(self) -> None:
        """Load leaderboard from JSON file."""
        if not self.leaderboard_path.exists():
            print(f"[Leaderboard] No existing leaderboard at {self.leaderboard_path}")
            return
        try:
            with open(self.leaderboard_path, "r") as f:
                data = json.load(f)
            self.entries = [LeaderboardEntry.from_dict(d) for d in data]
            print(f"[Leaderboard] Loaded {len(self.entries)} entries")
        except Exception as e:
            print(f"[Leaderboard] Load failed: {e}")

    def save(self) -> None:
        """Save leaderboard to JSON file."""
        self.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.leaderboard_path, "w") as f:
                json.dump([e.to_dict() for e in self.entries], f, indent=2)
            print(f"[Leaderboard] Saved {len(self.entries)} entries to {self.leaderboard_path}")
        except Exception as e:
            print(f"[Leaderboard] Save failed: {e}")

    def add_entry(
        self,
        benchmark: str,
        model: str,
        avg_score: Optional[float],
        count: int,
        notes: str = "",
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        replace_existing: bool = True,
    ) -> LeaderboardEntry:
        """
        Add or update a leaderboard entry.

        Args:
            benchmark: Benchmark name.
            model: Model name.
            avg_score: Average score.
            count: Number of examples evaluated.
            notes: Additional notes.
            run_id: WandB run ID if applicable.
            config: Run configuration.
            replace_existing: If True, replace existing entry for same benchmark/model.

        Returns:
            The added/updated entry.
        """
        timestamp = datetime.now().isoformat()

        entry = LeaderboardEntry(
            benchmark=benchmark,
            model=model,
            avg_score=avg_score,
            count=count,
            notes=notes,
            timestamp=timestamp,
            run_id=run_id,
            config=config or {},
        )

        if replace_existing:
            # Remove existing entries for same benchmark/model
            self.entries = [
                e for e in self.entries
                if not (e.benchmark == benchmark and e.model == model)
            ]

        self.entries.append(entry)
        return entry

    def get_best_for_benchmark(self, benchmark: str) -> Optional[LeaderboardEntry]:
        """Get the best entry for a given benchmark."""
        benchmark_entries = [e for e in self.entries if e.benchmark == benchmark and e.avg_score is not None]
        if not benchmark_entries:
            return None
        return max(benchmark_entries, key=lambda e: e.avg_score or 0)

    def get_entries_for_model(self, model: str) -> List[LeaderboardEntry]:
        """Get all entries for a given model."""
        return [e for e in self.entries if e.model == model]

    def generate_html(self, output_path: Optional[Path] = None) -> str:
        """
        Generate an HTML leaderboard report.

        Returns:
            HTML string.
        """
        output_path = output_path or HTML_TEMPLATE_PATH

        # Group by benchmark
        benchmarks: Dict[str, List[LeaderboardEntry]] = {}
        for entry in self.entries:
            if entry.benchmark not in benchmarks:
                benchmarks[entry.benchmark] = []
            benchmarks[entry.benchmark].append(entry)

        # Sort each benchmark by score
        for bench in benchmarks:
            benchmarks[bench].sort(key=lambda e: e.avg_score or 0, reverse=True)

        # Generate HTML
        html = self._generate_html_content(benchmarks)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        print(f"[Leaderboard] HTML saved to {output_path}")

        return html

    def _generate_html_content(self, benchmarks: Dict[str, List[LeaderboardEntry]]) -> str:
        """Generate HTML content for the leaderboard."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SleepTrain Benchmark Leaderboard</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --border: #30363d;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }}
        h2 {{
            color: var(--text-primary);
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg-primary);
            font-weight: 600;
            color: var(--text-secondary);
        }}
        tr:hover {{
            background: rgba(88, 166, 255, 0.1);
        }}
        .rank-1 {{
            color: var(--success);
            font-weight: 600;
        }}
        .score {{
            font-family: 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace;
        }}
        .timestamp {{
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            background: var(--accent);
            color: var(--bg-primary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 SleepTrain Benchmark Leaderboard</h1>
        <p class="timestamp">Last updated: {timestamp}</p>
"""

        for benchmark, entries in sorted(benchmarks.items()):
            html += f"""
        <h2>{benchmark}</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Score</th>
                    <th>Examples</th>
                    <th>Notes</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
"""
            for rank, entry in enumerate(entries, 1):
                rank_class = "rank-1" if rank == 1 else ""
                score_str = f"{entry.avg_score:.4f}" if entry.avg_score is not None else "N/A"
                ts = entry.timestamp[:10] if entry.timestamp else "N/A"

                html += f"""
                <tr class="{rank_class}">
                    <td>{rank}</td>
                    <td>{entry.model}</td>
                    <td class="score">{score_str}</td>
                    <td>{entry.count}</td>
                    <td>{entry.notes}</td>
                    <td class="timestamp">{ts}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""

        html += f"""
        <div class="footer">
            <p>Generated by SleepTrain v2 | <a href="https://github.com/sleeptrain" style="color: var(--accent);">GitHub</a></p>
        </div>
    </div>
</body>
</html>
"""
        return html


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def log_benchmark_result(
    benchmark: str,
    model: str,
    avg_score: float,
    count: int,
    notes: str = "",
    config: Optional[Dict[str, Any]] = None,
    wandb_logger: Optional[WandBLogger] = None,
    update_leaderboard: bool = True,
) -> Dict[str, Any]:
    """
    Log a benchmark result to WandB and update the leaderboard.

    Args:
        benchmark: Benchmark name.
        model: Model name.
        avg_score: Average score achieved.
        count: Number of examples evaluated.
        notes: Additional notes.
        config: Run configuration.
        wandb_logger: Optional WandBLogger instance.
        update_leaderboard: Whether to update the local leaderboard.

    Returns:
        Dict with logged data.
    """
    result = {
        "benchmark": benchmark,
        "model": model,
        "avg_score": avg_score,
        "count": count,
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }

    # Log to WandB
    run_id = None
    if wandb_logger:
        wandb_logger.log({
            f"bench/{benchmark}/avg_score": avg_score,
            f"bench/{benchmark}/count": count,
        })
        run_id = wandb_logger.run.id if wandb_logger.run else None

    # Update leaderboard
    if update_leaderboard:
        manager = LeaderboardManager()
        manager.add_entry(
            benchmark=benchmark,
            model=model,
            avg_score=avg_score,
            count=count,
            notes=notes,
            run_id=run_id,
            config=config,
        )
        manager.save()
        manager.generate_html()

    return result


def create_wandb_logger(
    project: str = "sleeptrain",
    tags: Optional[List[str]] = None,
    mode: str = "online",
) -> WandBLogger:
    """Factory function for creating a WandB logger."""
    config = WandBConfig(
        project=project,
        tags=tags or [],
        mode=mode,
    )
    return WandBLogger(config=config)


def update_leaderboard_from_results(
    results_dir: str = "data/experiment_results/rescored",
) -> int:
    """
    Update leaderboard from experiment result files.

    Args:
        results_dir: Directory containing result JSON files.

    Returns:
        Number of entries updated.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[Leaderboard] Results directory not found: {results_dir}")
        return 0

    manager = LeaderboardManager()
    updated = 0

    for result_file in results_path.glob("*.json"):
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            # Extract benchmark info from result file
            if isinstance(data, dict):
                benchmark = data.get("benchmark", result_file.stem)
                model = data.get("model", "unknown")
                avg_score = data.get("avg_score") or data.get("accuracy")
                count = data.get("count", 0)
                notes = data.get("notes", f"From {result_file.name}")

                if avg_score is not None:
                    manager.add_entry(
                        benchmark=benchmark,
                        model=model,
                        avg_score=float(avg_score),
                        count=count,
                        notes=notes,
                    )
                    updated += 1
        except Exception as e:
            print(f"[Leaderboard] Error processing {result_file}: {e}")

    if updated > 0:
        manager.save()
        manager.generate_html()

    return updated


__all__ = [
    "LeaderboardEntry",
    "LeaderboardManager",
    "WandBConfig",
    "WandBLogger",
    "log_benchmark_result",
    "create_wandb_logger",
    "update_leaderboard_from_results",
    "WANDB_AVAILABLE",
]

