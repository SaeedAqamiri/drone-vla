from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .runner import RunResult
from .tiles import TileStore


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_run(
    out_dir: str | Path,
    result: RunResult,
    *,
    store: TileStore,
    reference_manifest: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "config": result.config,
        "summary": result.summary,
        "store": {
            "root": str(store.root),
            "source": store.manifest.get("source"),
            "gsd": store.gsd,
            "tile_px": store.manifest.get("tile_px"),
            "tile_ground_m": store.tile_ground_m,
            "tiles": store.size,
        },
        "references": reference_manifest,
        "extra": extra or {},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    frame_rows = [frame.as_dict() for frame in result.frames]
    fieldnames = [
        "index",
        "query_id",
        "timestamp",
        "truth_east",
        "truth_north",
        "prior_east",
        "prior_north",
        "candidates",
        "attempted",
        "matched",
        "accepted",
        "best_tile",
        "best_east",
        "best_north",
        "error_m",
        "prior_error_m",
        "latency_ms",
        "reason",
        "match_inliers",
        "match_inlier_ratio",
        "match_coverage",
        "match_scale",
        "match_rotation",
        "match_reproj",
    ]
    flattened = []
    for row in frame_rows:
        match = row.get("match") or {}
        flattened.append(
            {
                **{key: row.get(key) for key in fieldnames[:18]},
                "match_inliers": match.get("inliers"),
                "match_inlier_ratio": match.get("inlier_ratio"),
                "match_coverage": match.get("coverage"),
                "match_scale": match.get("scale"),
                "match_rotation": match.get("rotation"),
                "match_reproj": match.get("reproj"),
            }
        )
    _write_csv(out_dir / "frames.csv", flattened, fieldnames)

    result_row = {"matcher": result.config.get("matcher", "")}
    result_row.update(result.summary)
    _write_csv(
        out_dir / "results.csv",
        [result_row],
        list(result_row.keys()),
    )

    summary = result.summary
    lines = [
        "# Visual localization lab run",
        "",
        f"- matcher: `{result.config.get('matcher')}`",
        f"- mode: `{result.config.get('mode')}`",
        f"- store: `{store.root}` ({store.size} tiles, gsd={store.gsd:.3f} m)",
        f"- references: `{reference_manifest.get('name') if reference_manifest else 'satellite'}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key in (
        "frames",
        "attempted",
        "matched",
        "accepted",
        "match_rate",
        "fix_rate",
        "accepted_rate",
        "precision",
        "false_fix_rate",
        "mean_error_m",
        "median_error_m",
        "p95_error_m",
        "max_error_m",
        "mean_prior_error_m",
        "p95_prior_error_m",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_candidates",
        "time_to_first_fix_s",
        "mean_fix_interval_s",
        "recovery_s",
    ):
        value = summary.get(key)
        if isinstance(value, float):
            value = round(value, 4)
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Failure reasons", ""])
    failures = summary.get("failure_counts") or {}
    if failures:
        for reason, count in sorted(failures.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Config",
            "",
            "```json",
            json.dumps(result.config, indent=2),
            "```",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir
