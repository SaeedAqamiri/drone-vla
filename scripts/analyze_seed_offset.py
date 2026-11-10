#!/usr/bin/env python3
"""Offline offset diagnostic using ONLY the two known initialization positions.

The resulting correction is applied retrospectively to an existing run's
accepted measurements. That is not a new closed-loop flight evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navlab.lab.candidates import Prior
from navlab.lab.navigator import VisualNavigator
from navlab.lab.remote_gim import RemoteGIMMatcher
from navlab.lab.runner import RunnerConfig
from navlab.lab.sequences import load_queries, uav_visloc_queries
from navlab.lab.tiles import TileStore


def stats(values):
    if not values:
        return {"accepted": 0, "within_30m": 0, "over_50m": 0, "median_m": None, "p95_m": None}
    return {"accepted": len(values), "within_30m": sum(x <= 30 for x in values),
            "over_50m": sum(x > 50 for x in values), "median_m": float(np.median(values)),
            "p95_m": float(np.percentile(values, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/raw/uav_visloc/UAV_VisLoc_dataset"))
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    _, meta = load_queries(args.study / "queries.json")
    seed = json.loads((args.study / "initial_prior.json").read_text())
    ids = seed["initialization_frames"]
    if len(ids) != 2:
        raise ValueError("exactly two initialization frames required")
    candidates = uav_visloc_queries(args.root, str(meta["map"]), near_nadir_deg=float("inf"))
    by_id = {q.id: q for q in candidates}
    first_two = [by_id[qid] for qid in ids]
    if first_two[1].timestamp <= first_two[0].timestamp:
        raise ValueError("initialization timestamps must increase")
    store = TileStore(args.study / "tiles")
    config = RunnerConfig(matcher="gim_dkm", mode="gray", max_dim=768, max_candidates=2,
                          base_radius=60, max_radius=800, global_fallback=False,
                          max_jump_m=120, scale_tolerance=0, min_coverage=.1, confirmation=1)
    matcher = RemoteGIMMatcher(args.host, max_dim=768, min_coverage=.1)
    calibrations, offsets = [], []
    try:
        for q in first_two:
            true_e, true_n = store.to_local(q.lon, q.lat)
            prior = Prior(east=true_e, north=true_n, timestamp=q.timestamp, sigma=30, valid=True)
            nav = VisualNavigator(store, prior, config=config, matcher=matcher, rotations=(0, 90, 180, 270))
            fix = nav.process_frame(q.path, q.timestamp)
            record = {"query": q.id, "accepted": fix.accepted, "reason": fix.reason}
            if fix.execution_errors:
                raise RuntimeError("; ".join(fix.execution_errors))
            if fix.accepted:
                offset = (true_e-fix.east_m, true_n-fix.north_m)
                offsets.append(offset)
                record["offset_m"] = offset
            calibrations.append(record)
    finally:
        matcher.close()
    reason = "ok"
    if len(offsets) != 2:
        reason = "insufficient_initial_matches"
    elif any(math.hypot(*v) > 100 for v in offsets):
        reason = "initial_offset_too_large"
    elif math.dist(*offsets) > 20:
        reason = "inconsistent_initial_offsets"
    report = {"note": "Known initial labels used for two calibration matches only; later values evaluated retrospectively. "
                       "No corrected closed-loop run. Reference label/camera-center ambiguity remains.",
              "seed_matches": calibrations, "status": reason, "consistency_limit_m": 20,
              "max_offset_m": 100, "study": str(args.study), "result": str(args.result)}
    if reason == "ok":
        offset = np.mean(offsets, axis=0)
        before, after = [], []
        with (args.result / "frames.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["accepted"] != "True" or not row["error_m"]:
                    continue
                dx = float(row["best_east"]) - float(row["truth_east"])
                dy = float(row["best_north"]) - float(row["truth_north"])
                before.append(math.hypot(dx, dy))
                after.append(math.hypot(dx+offset[0], dy+offset[1]))
        report.update(offset_m=offset.tolist(), original=stats(before), retrospectively_shifted=stats(after))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
