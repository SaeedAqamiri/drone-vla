#!/usr/bin/env python3
"""Combine per-flight ``eval/summary.json`` files into one table.

``run_uav_visloc.py`` writes a summary as it goes; this rebuilds it from the
per-flight artefacts, which is what you want after partial or parallel runs::

    python scripts/summarize_uav_visloc.py --root data/uav_visloc/processed_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(root: Path) -> list[dict]:
    rows = []
    for site in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = site / "eval" / "summary.json"
        meta_path = site / "meta.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        layout = "?"
        if meta_path.is_file():
            layout = json.loads(meta_path.read_text(encoding="utf-8")).get("layout", "?")
        rows.append(
            {
                "site": site.name,
                "layout": layout,
                "matcher": summary.get("matcher", "features"),
                "frames": summary.get("frames"),
                "fix_rate": summary.get("fix_rate"),
                "median_m": (summary.get("error_m") or {}).get("median"),
                "p90_m": (summary.get("error_m") or {}).get("p90"),
                "bias_m": (summary.get("bias_m") or {}).get("magnitude"),
                "debiased_median_m": (summary.get("error_m_debiased") or {}).get("median"),
                "under_2m": summary.get("under_2m"),
                "score_median": (summary.get("score") or {}).get("median"),
                "latency_ms": summary.get("latency_ms_median"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    root = Path(args.root)
    rows = collect(root)
    if not rows:
        raise SystemExit(f"no eval/summary.json under {root}")

    header = ["site", "layout", "matcher", "frames", "fix_rate", "median_m", "p90_m",
              "debiased_median_m", "bias_m", "under_2m", "score_median", "latency_ms"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        print("| " + " | ".join("" if row[key] is None else str(row[key]) for key in header) + " |")

    fields = list(rows[0])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    medians = [row["median_m"] for row in rows if row["median_m"] is not None]
    fixes = [row["fix_rate"] for row in rows if row["fix_rate"] is not None]
    if medians:
        print(f"\nflights {len(rows)} | median error {sorted(medians)[len(medians) // 2]} m "
              f"| median fix rate {sorted(fixes)[len(fixes) // 2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
