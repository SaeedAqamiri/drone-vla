from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .report import write_run
from .runner import LabRunner, RunnerConfig
from .sequences import load_queries
from .tiles import TileStore


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _summary_row(
    name: str,
    reference_name: str,
    matcher: str,
    mode: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "run": name,
        "reference": reference_name,
        "matcher": matcher,
        "mode": mode,
    }
    for key, value in summary.items():
        if key == "failure_counts":
            row[key] = json.dumps(value, sort_keys=True)
        else:
            row[key] = value
    return row


def run_bench(config_path: str | Path, out_dir: str | Path) -> Path:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_config(config_path)
    store = TileStore(payload["store"])
    queries, _ = load_queries(payload["queries"])
    runner_options = dict(payload.get("runner") or {})
    matchers = list(payload.get("matchers") or ["sift"])
    modes = list(payload.get("modes") or ["gray"])
    reference_sets = list(payload.get("reference_sets") or [{"name": "satellite"}])
    rows: list[dict[str, Any]] = []

    for reference in reference_sets:
        reference_name = str(reference.get("name") or "satellite")
        references = reference.get("references")
        reference_kind = reference.get("reference_kind")
        for matcher in matchers:
            for mode in modes:
                options = dict(runner_options)
                options.update(
                    {
                        "matcher": matcher,
                        "mode": mode,
                        "reference_kind": reference_kind,
                    }
                )
                config = RunnerConfig(**options)
                run_name = f"{reference_name}_{matcher}_{mode}"
                result = LabRunner(
                    store,
                    queries,
                    config,
                    references=references,
                ).run()
                reference_manifest = (
                    json.loads(Path(references).read_text(encoding="utf-8"))
                    if references
                    else None
                )
                write_run(
                    out_dir / "runs" / run_name,
                    result,
                    store=store,
                    reference_manifest=reference_manifest,
                )
                rows.append(
                    _summary_row(
                        run_name,
                        reference_name,
                        matcher,
                        mode,
                        result.summary,
                    )
                )

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        "run",
        "reference",
        "matcher",
        "mode",
        "attempted",
        "accepted",
        "match_rate",
        "fix_rate",
        "precision",
        "p95_error_m",
        "mean_latency_ms",
        "mean_candidates",
        "time_to_first_fix_s",
        "recovery_s",
    ]
    lines = [
        "# Visual localization benchmark",
        "",
        f"- store: `{payload['store']}`",
        f"- queries: `{payload['queries']}`",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "Per-run reports are under `runs/`.", ""])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir
