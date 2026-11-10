#!/usr/bin/env python3
"""Prepare a repeatable local-search benchmark from a real UAV-VisLoc segment.

The initial fixes establish the initial position/velocity. Later labels are
used only for the evaluation and to define the fixed map extent, never as an
online per-frame prior. This is a known-start benchmark, not global retrieval.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from navlab.geo import local_crs
from navlab.lab.sequences import uav_visloc_queries, uav_visloc_satellite, write_queries
from navlab.lab.tiles import build_tiles_in_bbox


def initial_velocity(points, timestamps):
    """Least-squares velocity from positions already known at initialization."""
    if len(points) != len(timestamps) or len(points) < 2:
        raise ValueError("at least two paired initialization points required")
    times = [float(t)-float(timestamps[0]) for t in timestamps]
    mean_t = sum(times)/len(times)
    denom = sum((t-mean_t)**2 for t in times)
    if denom <= 0:
        raise ValueError("initialization times must span a nonzero duration")
    return tuple(sum((t-mean_t)*p[axis] for t,p in zip(times, points))/denom for axis in (0, 1))


def first_level_segment(queries, count: int, angle_deg: float, max_gap_s: float, segment_index: int = 0):
    """Select a consecutive measured-attitude run; never inspect match outcomes."""
    if count < 3 or angle_deg <= 0 or max_gap_s <= 0 or segment_index < 0:
        raise ValueError("count >= 3, positive angle and max gap required")
    segment = []
    found = 0
    counted_run = False
    for query in queries:
        meta = query.meta
        angles = (meta["pitch_deg"], meta["roll_deg"])
        level = (meta.get("attitude_present", False)
                 and all(math.isfinite(a) for a in angles)
                 and max(abs(a) for a in angles) < angle_deg)
        if not level:
            segment = []
            counted_run = False
            continue
        if segment and (query.meta["num"] != segment[-1].meta["num"] + 1
                        or not 0 < query.timestamp - segment[-1].timestamp <= max_gap_s):
            segment = []
            counted_run = False
        segment.append(query)
        if len(segment) == count and not counted_run:
            if found == segment_index:
                return segment
            found += 1
            counted_run = True
    raise ValueError(f"no qualifying run {segment_index} of {count} frames below {angle_deg} degrees")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/uav_visloc/UAV_VisLoc_dataset")
    parser.add_argument("--map", required=True)
    parser.add_argument("--satellite", help="optional georeferenced raster or VRT override")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=32, help="includes initialization frames")
    parser.add_argument("--init-count", type=int, default=2,
                        help="number of known pre-outage positions used to estimate initial velocity")
    parser.add_argument("--max-init-gap-s", type=float, default=30,
                        help="reject initialization frames farther apart than this")
    parser.add_argument("--margin-m", type=float, default=400)
    parser.add_argument("--tile-m", type=float, default=500)
    parser.add_argument("--stride-m", type=float, default=250)
    parser.add_argument("--tile-px", type=int, default=1024)
    parser.add_argument("--first-level-deg", type=float,
                        help="select first contiguous measured-attitude segment below this angle")
    parser.add_argument("--max-segment-gap-s", type=float, default=10)
    parser.add_argument("--level-segment-index", type=int, default=0,
                        help="zero-based qualifying contiguous run; ignored without --first-level-deg")
    args = parser.parse_args()
    if args.first_level_deg is None and args.level_segment_index:
        parser.error("--level-segment-index requires --first-level-deg")
    if args.first_level_deg is not None:
        if args.start:
            parser.error("--start cannot be combined with --first-level-deg")
        all_queries = uav_visloc_queries(args.root, args.map, near_nadir_deg=float("inf"))
        queries = first_level_segment(all_queries, args.count, args.first_level_deg, args.max_segment_gap_s,
                                      args.level_segment_index)
    else:
        queries = uav_visloc_queries(args.root, args.map, near_nadir_deg=15)[args.start:args.start + args.count]
    if args.init_count < 2 or len(queries) <= args.init_count:
        raise ValueError("at least two initialization frames and one evaluation frame required")
    if any(b.timestamp <= a.timestamp for a, b in zip(queries, queries[1:])):
        raise ValueError("nonmonotonic timestamp; use a contiguous segment")
    init_queries = queries[:args.init_count]
    init_gaps = [b.timestamp-a.timestamp for a, b in zip(init_queries, init_queries[1:])]
    if args.max_init_gap_s <= 0 or max(init_gaps) > args.max_init_gap_s:
        raise ValueError(f"initialization interval {max(init_gaps):.1f}s exceeds {args.max_init_gap_s:.1f}s; choose a contiguous start")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    satellite = Path(args.satellite) if args.satellite else uav_visloc_satellite(args.root, args.map)
    # The map ROI is fixed before evaluation; the bounding rectangle is an
    # experiment convenience using labels, not an online localization input.
    crs = local_crs(queries[0].lat, queries[0].lon)
    to_local = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    points = [to_local.transform(q.lon, q.lat) for q in queries]
    xs, ys = zip(*points)
    corners = [to_wgs.transform(x, y) for x in (min(xs)-args.margin_m, max(xs)+args.margin_m)
               for y in (min(ys)-args.margin_m, max(ys)+args.margin_m)]
    bbox = {"min_lon": min(p[0] for p in corners), "max_lon": max(p[0] for p in corners),
            "min_lat": min(p[1] for p in corners), "max_lat": max(p[1] for p in corners)}
    print("building fixed map ROI", bbox, flush=True)
    store = build_tiles_in_bbox(satellite, out / "tiles", bbox,
                                tile_ground_m=args.tile_m, tile_px=args.tile_px,
                                stride_m=args.stride_m)
    write_queries(out / "initialization_queries.json", init_queries,
                  {"note": "Known positions available before evaluation; not online ground truth."})
    write_queries(out / "queries.json", queries[args.init_count:],
                  {"dataset": "uav_visloc", "map": args.map, "start_index": args.start,
                   "selection": {"first_level_deg": args.first_level_deg,
                                 "level_segment_index": args.level_segment_index if args.first_level_deg is not None else None,
                                 "max_segment_gap_s": args.max_segment_gap_s if args.first_level_deg is not None else None},
                   "initialization_frames": [q.id for q in init_queries],
                   "map_roi": "fixed bbox from evaluation segment plus margin; offline only"})
    # Fit velocity to all available pre-outage positions. With two points this
    # is identical to the original finite-difference seed.
    vx, vy = initial_velocity(points[:args.init_count], [q.timestamp for q in init_queries])
    last_init = init_queries[-1]
    seed = {"lat": last_init.lat, "lon": last_init.lon, "timestamp": last_init.timestamp,
            "vx": vx, "vy": vy, "sigma_m": 30,
            "initialization_frames": [q.id for q in init_queries],
            "velocity_method": "least_squares_pre_outage"}
    (out / "initial_prior.json").write_text(json.dumps(seed, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(queries)-args.init_count, "tiles": len(store["tiles"]), "prior": seed}, indent=2))


if __name__ == "__main__":
    main()
