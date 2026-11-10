#!/usr/bin/env python3
"""Offline exploratory analysis: label-derived position residual vs attitude.

This is not an online calibration method. It requires labeled positions for
both fitting and scoring; never use its output as a validated correction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_segment(study: Path, run: Path, yaw_relative: bool = False):
    queries = {q["id"]: q for q in json.loads((study / "queries.json").read_text())["queries"]}
    with (run / "frames.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["accepted"] == "True"]
    features, residuals, ids = [], [], []
    for r in rows:
        q = queries.get(r["query_id"])
        if not q or not all(r.get(k) for k in ("truth_east", "truth_north", "best_east", "best_north")):
            continue
        meta = q.get("meta") or {}
        if "pitch_deg" not in meta or "roll_deg" not in meta:
            continue
        pitch, roll = float(meta["pitch_deg"]), float(meta["roll_deg"])
        if yaw_relative:
            yaw = np.deg2rad(float(q["yaw"]))
            features.append([1.0, pitch*np.sin(yaw) + roll*np.cos(yaw),
                             pitch*np.cos(yaw) - roll*np.sin(yaw)])
        else:
            features.append([1.0, pitch, roll])
        residuals.append([float(r["best_east"])-float(r["truth_east"]),
                          float(r["best_north"])-float(r["truth_north"])])
        ids.append(r["query_id"])
    return np.asarray(features, dtype=float).reshape(-1, 3), np.asarray(residuals, dtype=float).reshape(-1, 2), ids


def scores(errors: np.ndarray) -> dict:
    if not len(errors):
        return {"count": 0, "median_m": None, "p95_m": None}
    return {"count": len(errors), "median_m": float(np.median(errors)),
            "p95_m": float(np.percentile(errors, 95)),
            "within_20m": int(np.count_nonzero(errors <= 20)),
            "within_30m": int(np.count_nonzero(errors <= 30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-study", type=Path, required=True)
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True,
                        help="number of initial accepted labeled frames for offline fitting")
    parser.add_argument("--yaw-relative", action="store_true",
                        help="experimental: rotate pitch/roll features by metadata yaw")
    parser.add_argument("--eval-study", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=0,
                        help="skip initial accepted frames in the eval run, e.g. training frames")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    x, y, train_ids = load_segment(args.train_study, args.train_run, args.yaw_relative)
    if args.train_count < 3 or len(x) < args.train_count:
        parser.error("at least three and no more than the available labeled accepted training frames required")
    fit_x = x[:args.train_count]
    if np.linalg.matrix_rank(fit_x) < 3:
        parser.error("training attitude values do not provide full-rank calibration")
    beta = np.linalg.lstsq(fit_x, y[:args.train_count], rcond=None)[0]
    eval_x, eval_y, eval_ids = load_segment(args.eval_study, args.eval_run, args.yaw_relative)
    if args.skip < 0 or args.skip >= len(eval_x):
        parser.error("--skip must leave at least one labeled accepted evaluation frame")
    # Same-run tests cannot count training frames as independent holdout.
    if args.train_run.resolve() == args.eval_run.resolve() and args.skip < args.train_count:
        parser.error("same-run evaluation must skip all calibration frames")
    eval_x, eval_y, eval_ids = eval_x[args.skip:], eval_y[args.skip:], eval_ids[args.skip:]
    result = {"warning": "Offline label-fitted diagnostic; not a flight correction or model validation.",
              "train_run": str(args.train_run), "train_query_ids": train_ids[:args.train_count],
              "eval_run": str(args.eval_run), "eval_query_ids": eval_ids,
              "features": (["constant", "yaw_rotated_1_deg", "yaw_rotated_2_deg"]
                           if args.yaw_relative else ["constant", "pitch_deg", "roll_deg"]),
              "residual_coefficients_east_north": beta.tolist(),
              "raw": scores(np.linalg.norm(eval_y, axis=1)),
              "after_subtracting_fitted_residual": scores(np.linalg.norm(eval_y-eval_x@beta, axis=1))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
