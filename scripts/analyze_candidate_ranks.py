#!/usr/bin/env python3
"""Retrospective candidate coverage by spatial rank; not an online oracle."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navlab.lab.tiles import TileStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", nargs=2, action="append", metavar=("STUDY", "RESULT"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for study, result in args.run:
        store = TileStore(Path(study) / "tiles")
        with (Path(result) / "frames.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                if not all(row[k] for k in ("prior_east", "prior_north", "truth_east", "truth_north")):
                    continue
                e, n = float(row["truth_east"]), float(row["truth_north"])
                # Ordering is the normal selector ordering, but without the
                # radius cutoff; this estimates potential rank, not online recall.
                tiles = store.candidates(float(row["prior_east"]), float(row["prior_north"]), float("inf"))
                rank = next((i for i, tile in enumerate(tiles, 1)
                             if tile["bounds"]["min_east"] <= e <= tile["bounds"]["max_east"]
                             and tile["bounds"]["min_north"] <= n <= tile["bounds"]["max_north"]), None)
                rows.append({"query": row["query_id"], "region": Path(study).name, "rank": rank,
                             "actual_selected_tile": row["best_tile"], "accepted": row["accepted"] == "True"})
    summary = {str(k): sum(r["rank"] is not None and r["rank"] <= k for r in rows) for k in (1, 2, 3, 4, 8)}
    report = {"note": "Post-hoc label center containment by rank using priors from the original run; no radius cutoff. "
                      "Not a new closed-loop run, no check of full image footprint or match success.",
              "frames": len(rows), "covered_at_k": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"frames": len(rows), "covered_at_k": summary}, indent=2))


if __name__ == "__main__":
    main()
