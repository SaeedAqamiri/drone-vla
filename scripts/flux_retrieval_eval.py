#!/usr/bin/env python3
"""Does a FLUX-restyled reference set identify the right place better than the raw one?

Given a run with ``N`` frames, each frame has one reference crop centred on its
true position. For each query frame we score it against **all** ``N`` reference
crops and ask which one wins. That is the question that matters for map
matching: not "how many metres off inside the right crop", but "did the matcher
pick the right part of the map at all".

Usage::

    python scripts/flux_retrieval_eval.py \
        --run data/uav_visloc/processed_v1/06 \
        --aug data/uav_visloc/processed_v1/06_flux
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sat2aerial.localize import _correlate  # noqa: E402


def load_refs(directory: Path) -> dict[str, np.ndarray]:
    refs = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            refs[path.stem] = image
    return refs


def score_matrix(
    queries: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    angles: tuple[float, ...],
    query_fraction: float,
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for qname, query in queries.items():
        row: dict[str, float] = {}
        for rname, ref in refs.items():
            height, width = query.shape[:2]
            keep_h = int(round(height * query_fraction))
            keep_w = int(round(width * query_fraction))
            top = (height - keep_h) // 2
            left = (width - keep_w) // 2
            crop = query[top : top + keep_h, left : left + keep_w]
            result = _correlate(ref, crop, coarse=320, angles=angles)
            row[rname] = 0.0 if result is None else result[0]
        table[qname] = row
    return table


def report(name: str, table: dict[str, dict[str, float]]) -> tuple[int, float]:
    print(f"\n== {name}")
    correct = 0
    margins = []
    for query, row in table.items():
        ranked = sorted(row.items(), key=lambda item: -item[1])
        best, best_score = ranked[0]
        own = row.get(query, float("nan"))
        rank = 1 + sum(1 for _, value in ranked if value > own)
        runner = max(value for key, value in row.items() if key != query)
        correct += int(best == query)
        margins.append(best_score - runner)
        print(f"  {query}  rank {rank}/{len(row)}  own {own:.3f}  best {best} {best_score:.3f}"
              f"  margin {best_score - runner:+.3f}")
    total = len(table)
    print(f"  top-1 {correct}/{total}  mean margin {np.mean(margins):+.3f} "
          f"min {np.min(margins):+.3f}")
    return correct, float(np.mean(margins))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--aug", required=True)
    parser.add_argument("--rotate-step", type=float, default=15.0)
    parser.add_argument("--query-fraction", type=float, default=0.6)
    parser.add_argument("--query-dir", default="",
                        help="directory of restyled queries (default: the run's own)")
    args = parser.parse_args(argv)

    run = Path(args.run)
    aug = Path(args.aug)
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    ref_gsd = float(meta["map"]["gsd_m"])
    frame_gsd = float(meta["map"]["frame_gsd_m"])
    poses = list(csv.DictReader((run / "poses.csv").open(encoding="utf-8")))

    raw_refs = load_refs(run / "pairs" / "satellite")
    aug_refs = load_refs(aug / "pairs" / "satellite")
    shared = sorted(set(raw_refs) & set(aug_refs))
    if not shared:
        raise SystemExit("no shared frames between the two runs")
    raw_refs = {key: raw_refs[key] for key in shared}
    aug_refs = {key: aug_refs[key] for key in shared}

    scale = frame_gsd / ref_gsd
    query_root = Path(args.query_dir) if args.query_dir else run / "pairs" / "aerial"
    queries = {}
    for row in poses:
        stem = row["frame"]
        if stem not in shared:
            continue
        candidates = [query_root / f"{stem}_00.jpg", query_root / f"{stem}.png",
                      query_root / f"{stem}.jpg"]
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            continue
        query = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if query is None:
            continue
        queries[stem] = cv2.resize(query, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    angles = tuple(np.arange(0.0, 360.0, args.rotate_step).tolist())
    print(f"frames {len(queries)}  refs {len(shared)}  angles {len(angles)}  "
          f"query scale {scale:.3f}  fraction {args.query_fraction}")
    raw_table = score_matrix(queries, raw_refs, angles, args.query_fraction)
    aug_table = score_matrix(queries, aug_refs, angles, args.query_fraction)
    raw_hits, raw_margin = report("raw satellite", raw_table)
    aug_hits, aug_margin = report("flux restyled", aug_table)
    print(f"\nraw {raw_hits}/{len(queries)} (margin {raw_margin:+.3f}) | "
          f"flux {aug_hits}/{len(queries)} (margin {aug_margin:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
