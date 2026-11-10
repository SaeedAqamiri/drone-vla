#!/usr/bin/env python3
"""Evaluate calibrated SIFT or GIM-DKM + PnP on an OrthoLoC sample.

This uses the sample's camera intrinsics, already-selected orthophoto and DSM.
It does not perform map retrieval, sensor fusion or sequential navigation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navlab.lab.pnp import estimate_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-gim-host", help="use a trusted remote GIM-DKM worker instead of SIFT")
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=.2)
    parser.add_argument("--min-query-coverage", type=float, default=.03)
    parser.add_argument("--save-matches", type=Path, help="save 2D/3D matches for offline gate studies")
    parser.add_argument("--query-scale", type=float, default=1,
                        help="resize the camera image and intrinsics; not a simulation of altitude")
    args = parser.parse_args()
    started = time.monotonic()
    cv2.setNumThreads(2)
    with np.load(args.sample, allow_pickle=False) as data:
        query, dop = data["image_query"], data["image_dop"]
        dsm, camera = data["dsm"], data["intrinsics"]
        gt_pose = data["extrinsics"]
        sample_id = str(data["sample_id"])
        scale = data["scale"].tolist()
    if not math.isfinite(args.query_scale) or not 0 < args.query_scale <= 1:
        parser.error("--query-scale must be in (0, 1]")
    if args.query_scale != 1:
        old_h, old_w = query.shape[:2]
        new_w, new_h = max(1, round(old_w * args.query_scale)), max(1, round(old_h * args.query_scale))
        query = cv2.resize(query, (new_w, new_h), interpolation=cv2.INTER_AREA)
        camera = np.diag([new_w/old_w, new_h/old_h, 1]) @ camera
    if dsm.shape != (*dop.shape[:2], 3):
        raise ValueError("DSM must contain one metric 3-D point per orthophoto pixel")
    if query.ndim != 3 or dop.ndim != 3 or query.shape[2] != 3 or dop.shape[2] != 3:
        raise ValueError("expected RGB images")
    query_keypoint_count = map_keypoint_count = None
    if args.remote_gim_host:
        from navlab.lab.remote_gim import RemoteGIMMatcher
        remote = RemoteGIMMatcher(args.remote_gim_host, max_dim=768)
        try:
            outcome, map_pts, query_pts = remote.match_with_points(
                cv2.cvtColor(dop, cv2.COLOR_RGB2BGR), cv2.cvtColor(query, cv2.COLOR_RGB2BGR))
        finally:
            remote.close()
        if outcome.execution_error:
            raise RuntimeError(outcome.error)
        # The matcher returns coordinates after its max-dimension resize.
        map_pts /= min(1.0, 768 / max(dop.shape[:2]))
        query_pts /= min(1.0, 768 / max(query.shape[:2]))
        if len(map_pts) > 2000:
            indices = np.random.default_rng(0).choice(len(map_pts), 2000, replace=False)
            map_pts, query_pts = map_pts[indices], query_pts[indices]
        matcher_name = "gim_dkm"
    else:
        sift = cv2.SIFT_create(nfeatures=7000)
        query_keypoints, query_desc = sift.detectAndCompute(cv2.cvtColor(query, cv2.COLOR_RGB2GRAY), None)
        map_keypoints, map_desc = sift.detectAndCompute(cv2.cvtColor(dop, cv2.COLOR_RGB2GRAY), None)
        query_keypoint_count, map_keypoint_count = len(query_keypoints), len(map_keypoints)
        neighbours = (cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_desc, map_desc, k=2)
                      if query_desc is not None and map_desc is not None else [])
        matches = [a for pair in neighbours if len(pair) == 2 for a, b in [pair] if a.distance < .75*b.distance]
        map_pts = [map_keypoints[match.trainIdx].pt for match in matches]
        query_pts = [query_keypoints[match.queryIdx].pt for match in matches]
        matcher_name = "sift"
    points, world = [], []
    for (x, y), query_pt in zip(map_pts, query_pts):
        u, v = int(round(x)), int(round(y))
        if 0 <= v < dsm.shape[0] and 0 <= u < dsm.shape[1] and np.isfinite(dsm[v, u]).all():
            points.append(query_pt)
            world.append(dsm[v, u])
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    world = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    if args.min_inliers < 6 or not 0 <= args.min_inlier_ratio <= 1 or not 0 <= args.min_query_coverage <= 1:
        parser.error("invalid PnP acceptance thresholds")
    if args.save_matches:
        args.save_matches.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_matches, world=world, image=points, intrinsics=camera,
                            image_shape=query.shape[:2])
    pose = estimate_pose(world, points, camera, query.shape[:2], reproj_px=6*args.query_scale, min_inliers=args.min_inliers,
                         min_inlier_ratio=args.min_inlier_ratio, min_coverage=args.min_query_coverage)
    result = {"sample_id": sample_id, "sample": str(args.sample),
              "note": "Ground truth is used only for retrospective error; map crop, DSM and intrinsics are supplied.",
              "map_scale_m_per_px": scale, "query_shape": query.shape, "map_shape": dop.shape,
              "query_scale": args.query_scale,
              "matcher": matcher_name, "sift_query_keypoints": query_keypoint_count,
              "sift_map_keypoints": map_keypoint_count,
              "tentative_matches": len(map_pts), "finite_3d_matches": len(world),
              "success": pose.success, "reason": pose.reason, "inliers": pose.inliers,
              "thresholds": {"min_inliers": args.min_inliers, "min_inlier_ratio": args.min_inlier_ratio,
                             "min_query_coverage": args.min_query_coverage, "reprojection_px": 6*args.query_scale},
              "inlier_ratio": pose.inlier_ratio, "query_coverage": pose.query_coverage,
              "median_reprojection_px": pose.median_reprojection_px}
    if pose.success:
        gt_center = -gt_pose[:, :3].T @ gt_pose[:, 3]
        error = pose.camera_center_world - gt_center
        cos_angle = np.clip((np.trace(pose.rotation_world_to_camera @ gt_pose[:, :3].T)-1)/2, -1, 1)
        result.update(camera_center_world=pose.camera_center_world.tolist(), truth_center_world=gt_center.tolist(),
                      horizontal_error_m=float(np.linalg.norm(error[:2])), position_error_3d_m=float(np.linalg.norm(error)),
                      angle_error_deg=math.degrees(math.acos(float(cos_angle))))
    result["elapsed_s"] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
