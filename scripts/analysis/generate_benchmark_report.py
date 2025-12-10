"""
Generate a simple HTML report for benchmark results.

Input: JSON file with a list of run dicts, each containing:
{
  "benchmark": "...",
  "split": "...",
  "avg_score": float,
  "count": int,
  "path": "...",
  "meta": {...}  # optional
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Benchmark Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f2f2f2; }
  </style>
</head>
<body>
  <h1>Benchmark Report</h1>
  <table>
    <tr>
      <th>Benchmark</th><th>Split</th><th>Avg Score</th><th>Count</th><th>Path</th>
    </tr>
    {rows}
  </table>
</body>
</html>
"""


def make_rows(runs: List[Dict[str, Any]]) -> str:
    row_html = []
    for r in runs:
        row_html.append(
            f"<tr><td>{r.get('benchmark','')}</td>"
            f"<td>{r.get('split','')}</td>"
            f"<td>{r.get('avg_score',0):.4f}</td>"
            f"<td>{r.get('count',0)}</td>"
            f"<td>{r.get('path','')}</td></tr>"
        )
    return "\n".join(row_html)


def generate_report(results_path: str, output_path: str) -> None:
    runs = json.loads(Path(results_path).read_text())
    rows = make_rows(runs if isinstance(runs, list) else [runs])
    html = TEMPLATE.format(rows=rows)
    Path(output_path).write_text(html, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark HTML report")
    parser.add_argument("--results", required=True, help="Path to JSON results file")
    parser.add_argument("--out", required=True, help="Output HTML file")
    args = parser.parse_args()

    generate_report(args.results, args.out)
