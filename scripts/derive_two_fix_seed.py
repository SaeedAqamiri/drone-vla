#!/usr/bin/env python3
"""Create a two-fix baseline for an existing multi-fix initialization study.

Both studies then use the same map, evaluation frames and final known starting
position. The only difference is how much pre-outage history estimates velocity.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navlab.lab.sequences import load_queries, write_queries
from navlab.lab.tiles import TileStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    initial, _ = load_queries(args.study / "initialization_queries.json")
    queries, meta = load_queries(args.study / "queries.json")
    original = json.loads((args.study / "initial_prior.json").read_text())
    if len(initial) < 3 or not queries:
        raise ValueError("source needs at least three initial positions and evaluation frames")
    a, b = initial[-2:]
    if b.timestamp != original["timestamp"] or not 0 < b.timestamp-a.timestamp <= 30:
        raise ValueError("inconsistent or excessively spaced initial positions")
    if args.out.exists():
        raise FileExistsError(args.out)
    store = TileStore(args.study / "tiles")
    e0, n0 = store.to_local(a.lon, a.lat)
    e1, n1 = store.to_local(b.lon, b.lat)
    dt = b.timestamp-a.timestamp
    seed = {"lat": b.lat, "lon": b.lon, "timestamp": b.timestamp,
            "vx": (e1-e0)/dt, "vy": (n1-n0)/dt, "sigma_m": original["sigma_m"],
            "initialization_frames": [a.id, b.id], "velocity_method": "two_pre_outage_fixes"}
    args.out.mkdir(parents=True)
    (args.out / "tiles").symlink_to((args.study / "tiles").resolve(), target_is_directory=True)
    (args.out / "initial_prior.json").write_text(json.dumps(seed, indent=2))
    write_queries(args.out / "initialization_queries.json", [a, b], {"note": "Last two known pre-outage positions."})
    meta["initialization_frames"] = [a.id, b.id]
    meta["velocity_comparison_source"] = str(args.study)
    write_queries(args.out / "queries.json", queries, meta)
    print(json.dumps(seed, indent=2))


if __name__ == "__main__":
    main()
