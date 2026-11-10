#!/usr/bin/env python3
"""Run a prepared known-start UAV-VisLoc segment without later GT as prior."""
from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navlab.lab.candidates import Prior
from navlab.lab.report import write_run
from navlab.lab.runner import LabRunner, RunnerConfig, RunResult
from navlab.lab.metrics import compute_summary
from navlab.lab.navigator import VisualNavigator
from navlab.lab.types import FrameResult
from navlab.lab.sequences import load_queries
from navlab.lab.tiles import TileStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--matcher", default="sift")
    parser.add_argument("--mode", default="gray")
    parser.add_argument("--max-dim", type=int, default=1024)
    parser.add_argument("--limit-frames", type=int, help="evaluate only the first N prepared query frames")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--radius", type=float, default=60)
    parser.add_argument("--max-radius", type=float, default=800)
    parser.add_argument("--max-jump", type=float, default=120)
    parser.add_argument("--jump-sigma-scale", type=float, default=2,
                        help="heuristic multiplier on prior sigma in the jump gate")
    parser.add_argument("--max-speed-mps", type=float, default=80,
                        help="maximum speed used to update prior velocity, not a flight limit")
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--min-ratio", type=float, default=.15)
    parser.add_argument("--min-coverage", type=float, default=.03)
    parser.add_argument("--references")
    parser.add_argument("--confirmation", type=int, default=1)
    parser.add_argument("--confirmation-radius", type=float, default=30,
                        help="max metre residual when checking consecutive candidate fixes")
    parser.add_argument("--online", action="store_true", help="evaluate the incremental public interface")
    parser.add_argument("--rotations", default="0", help="fallback rotations, e.g. 0,180")
    parser.add_argument("--oracle-motion", action="store_true", help="diagnostic only: derive ideal incremental motion from ground truth")
    parser.add_argument("--motion-noise-m", type=float, default=0.0)
    parser.add_argument("--visual-propagation", action="store_true",
                        help="experimental inter-frame ground-plane tracking, no later labels used online")
    parser.add_argument("--camera-gsd", type=float, help="known/estimated ground metres per original camera pixel")
    parser.add_argument("--remote-gim-host", help="run GIM-DKM inference on a trusted SSH host")
    parser.add_argument("--adaptive-footprint-width", type=float,
                        help="estimated full camera footprint width in metres; enables one adaptive map crop")
    parser.add_argument("--crop-uncertainty-scale", type=float, default=2.0)
    parser.add_argument("--crop-max-side", type=float, default=1500.0)
    parser.add_argument("--crop-min-valid-fraction", type=float, default=.95,
                        help="minimum available map fraction; < .95 is an experimental partial-map mode")
    parser.add_argument("--crop-sigma-growth", type=float, default=3.0,
                        help="heuristic uncertainty growth in m/s since previous prior timestamp")
    parser.add_argument("--landmark-min-supported", type=int, default=0,
                        help="experimental map-only visual-anchor gate; requires remote GIM")
    args = parser.parse_args()
    if args.landmark_min_supported < 0 or (args.landmark_min_supported and not args.remote_gim_host):
        parser.error("landmark gate requires a nonnegative threshold and remote GIM")
    if args.adaptive_footprint_width is not None and (not args.online or args.references):
        parser.error("adaptive crop requires --online and cannot use generated references")
    if args.oracle_motion and not args.online:
        parser.error("--oracle-motion requires --online")
    if args.visual_propagation and (not args.online or args.oracle_motion or args.camera_gsd is not None):
        parser.error("--visual-propagation requires --online and cannot be combined with oracle motion or camera GSD")
    if args.motion_noise_m < 0:
        parser.error("--motion-noise-m must be nonnegative")
    if (args.min_inliers < 4 or not 0 <= args.min_ratio <= 1 or not 0 <= args.min_coverage <= 1
            or not math.isfinite(args.max_speed_mps) or args.max_speed_mps <= 0
            or not math.isfinite(args.jump_sigma_scale) or args.jump_sigma_scale < 0
            or not math.isfinite(args.max_jump) or args.max_jump < 0
            or args.confirmation < 1 or not math.isfinite(args.confirmation_radius)
            or args.confirmation_radius <= 0):
        parser.error("invalid matcher acceptance thresholds")
    if args.remote_gim_host and (not args.online or args.matcher != "gim_dkm"):
        parser.error("--remote-gim-host requires --online --matcher gim_dkm")
    if args.limit_frames is not None and args.limit_frames <= 0:
        parser.error("--limit-frames must be positive")
    cv2.setNumThreads(2)
    study = Path(args.study)
    store = TileStore(study / "tiles")
    queries, meta = load_queries(study / "queries.json")
    if args.limit_frames is not None:
        queries = queries[:args.limit_frames]
    seed = json.loads((study / "initial_prior.json").read_text())
    east, north = store.to_local(seed["lon"], seed["lat"])
    prior = Prior(east=east, north=north, vx=seed["vx"], vy=seed["vy"],
                  sigma=seed["sigma_m"], timestamp=seed["timestamp"], valid=True)
    config = RunnerConfig(matcher=args.matcher, mode=args.mode, max_dim=args.max_dim,
                          max_candidates=args.candidates, base_radius=args.radius,
                          max_radius=args.max_radius, global_fallback=False,
                          min_inliers=args.min_inliers, min_ratio=args.min_ratio, min_coverage=args.min_coverage,
                          max_speed_mps=args.max_speed_mps,
                          max_jump_m=args.max_jump, jump_sigma_scale=args.jump_sigma_scale, scale_tolerance=0,
                          confirmation=args.confirmation,
                          confirmation_radius_m=args.confirmation_radius)
    print(f"Running {args.matcher} on {len(queries)} frames; {store.size} map tiles", flush=True)
    start = time.monotonic()
    if args.online:
        rotations = tuple(int(x) for x in args.rotations.split(","))
        remote_matcher = None
        if args.remote_gim_host:
            from navlab.lab.remote_gim import RemoteGIMMatcher
            remote_matcher = RemoteGIMMatcher(args.remote_gim_host, max_dim=args.max_dim,
                                               min_inliers=args.min_inliers, min_ratio=args.min_ratio,
                                               min_coverage=args.min_coverage)
            atexit.register(remote_matcher.close)
            if args.landmark_min_supported:
                from navlab.lab.landmarks import LandmarkVerifiedMatcher
                remote_matcher = LandmarkVerifiedMatcher(remote_matcher, min_supported=args.landmark_min_supported,
                                                        max_dim=args.max_dim)
        cropper = None
        if args.adaptive_footprint_width is not None:
            from navlab.lab.adaptive_crop import AdaptiveCrop
            cropper = AdaptiveCrop(store, footprint_width_m=args.adaptive_footprint_width,
                                   uncertainty_scale=args.crop_uncertainty_scale,
                                   max_side_m=args.crop_max_side,
                                   sigma_growth_mps=args.crop_sigma_growth,
                                   min_valid_fraction=args.crop_min_valid_fraction)
            atexit.register(cropper.close)
        navigator = VisualNavigator(store, prior, config=config, references=args.references,
                                    rotations=rotations, matcher=remote_matcher, cropper=cropper)
        by_id = {t["id"]: t for t in navigator.store.tiles}
        results = []
        previous_truth = store.to_local(seed["lon"], seed["lat"])
        tracker = None
        tracking_diagnostics = []
        crop_diagnostics = []
        previous_time = None
        if args.visual_propagation:
            from navlab.lab.ground_tracker import VisualGroundTracker
            tracker = VisualGroundTracker(max_dim=768, max_speed_mps=args.max_speed_mps)
        rng = np.random.default_rng(0)
        for index, query in enumerate(queries):
            frame_start = time.monotonic()
            truth_east, truth_north = store.to_local(query.lon, query.lat)
            delta = None
            image = query.path
            if args.oracle_motion:
                noise = rng.normal(0, args.motion_noise_m, size=2)
                delta = (truth_east-previous_truth[0]+noise[0], truth_north-previous_truth[1]+noise[1])
            if tracker is not None:
                image = cv2.imread(str(query.path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot read {query.path}")
                delta = tracker.predict(image, query.timestamp)
                diagnostic = {"query_id": query.id, **tracker.last_diagnostic}
                if delta is not None:
                    # Ground truth is used only here for the retrospective report.
                    diagnostic["delta_error_m"] = math.hypot(delta[0] - (truth_east-previous_truth[0]),
                                                              delta[1] - (truth_north-previous_truth[1]))
                tracking_diagnostics.append(diagnostic)
            sigma = (args.motion_noise_m if args.oracle_motion else
                     3 * (query.timestamp-previous_time) if delta is not None and previous_time is not None else None)
            nav = navigator.process_frame(image, query.timestamp, motion_delta_m=delta,
                                         motion_sigma_m=sigma,
                                         camera_gsd_m=args.camera_gsd)
            if tracker is not None:
                tracker.finish(nav.image_to_local if nav.accepted else None)
            previous_time = query.timestamp
            if nav.execution_errors:
                nav.match = nav.match or {}
                nav.match["execution_errors"] = list(nav.execution_errors)
            if nav.match is not None:
                nav.match["query_rotation_deg"] = nav.query_rotation_deg
            previous_truth = (truth_east, truth_north)
            covered = any(b["min_east"] <= truth_east <= b["max_east"]
                          and b["min_north"] <= truth_north <= b["max_north"]
                          for b in nav.candidate_bounds)
            if cropper is not None:
                bounds = nav.candidate_bounds[0] if nav.candidate_bounds else None
                crop_diagnostics.append({"query_id": query.id, "crop_diagnostic": dict(cropper.last_diagnostic), "bounds": bounds,
                                         "side_m": bounds["max_east"] - bounds["min_east"] if bounds else None,
                                         "valid_fraction": nav.candidate_valid_fractions[0] if nav.candidate_valid_fractions else None,
                                         "truth_in_crop_retrospective": covered})
            error = math.hypot(nav.east_m-truth_east, nav.north_m-truth_north) if nav.accepted else None
            prior_error = math.hypot(nav.prior_east_m-truth_east, nav.prior_north_m-truth_north)
            results.append(FrameResult(index=index, query_id=query.id, timestamp=query.timestamp,
                truth_east=truth_east, truth_north=truth_north, prior_east=nav.prior_east_m,
                prior_north=nav.prior_north_m, candidates=len(nav.candidate_ids), attempted=True,
                matched=bool(nav.match and nav.match.get("success")), accepted=nav.accepted,
                best_tile=nav.best_tile, best_east=nav.east_m, best_north=nav.north_m,
                error_m=error, prior_error_m=prior_error,
                latency_ms=(time.monotonic()-frame_start)*1000, reason=nav.reason,
                match=nav.match, candidate_truth_covered=covered))
        result = RunResult(frames=results, config={**asdict(config), "engine": "VisualNavigator",
                           "oracle_motion": args.oracle_motion, "motion_noise_m": args.motion_noise_m,
                           "rotations": rotations,
                           "landmark_min_supported": args.landmark_min_supported,
                           "adaptive_footprint_width_m": args.adaptive_footprint_width,
                           "crop_uncertainty_scale": args.crop_uncertainty_scale,
                           "crop_max_side_m": args.crop_max_side,
                           "crop_sigma_growth_mps": args.crop_sigma_growth,
                           "crop_min_valid_fraction": args.crop_min_valid_fraction,
                           "camera_gsd_m": args.camera_gsd,
                           "limit_frames": args.limit_frames,
                           "visual_propagation": args.visual_propagation,
                           "remote_gim_host": args.remote_gim_host,
                           "initial_prior": seed}, summary=compute_summary(results, config))
    else:
        result = LabRunner(store, queries, config, references=args.references,
                           initial_prior=prior).run()
    refs = json.loads(Path(args.references).read_text()) if args.references else None
    write_run(args.out, result, store=store, reference_manifest=refs, extra={"sequence": meta,
              "note": f"{len(seed.get('initialization_frames', []))} pre-outage positions seed the prior; "
                      "map ROI uses segment labels offline."})
    if args.online and args.adaptive_footprint_width is not None:
        (Path(args.out) / "crop_details.json").write_text(json.dumps({
            "note": "Prior-centred crop; label coverage is retrospective; sigma and camera footprint are heuristic.",
            "frames": crop_diagnostics}, indent=2), encoding="utf-8")
    if args.online and args.visual_propagation:
        (Path(args.out) / "visual_motion.json").write_text(json.dumps({
            "note": "SIFT inter-frame planar motion; label errors are retrospective evaluation only; sigma is heuristic.",
            "frames": tracking_diagnostics}, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_s": round(time.monotonic()-start, 2), **result.summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
