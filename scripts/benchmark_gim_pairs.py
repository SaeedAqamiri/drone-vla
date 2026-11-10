#!/usr/bin/env python3
"""Evaluate GIM-LoFTR on truth-selected pairs; diagnostic, not online navigation."""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from navlab.lab.gim import GIMDKMMatcher, GIMLoFTRMatcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matcher", choices=("gim_loftr", "gim_dkm"), default="gim_loftr")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    payload = json.loads(args.pairs.read_text())
    model = (GIMDKMMatcher if args.matcher == "gim_dkm" else GIMLoFTRMatcher)(max_dim=768)
    results = []
    for item in payload["pairs"]:
        if "skipped" in item:
            results.append(item)
            continue
        started = time.monotonic()
        img = cv2.imread(str(args.pairs.parent / item["image"]))
        ref = cv2.imread(str(args.pairs.parent / item["ref"]))
        rec = {"region": item["region"], "query": item["query"],
               "source_tile": item["source_tile"], "success": False}
        if img is None or ref is None:
            raise ValueError(f"missing image for {item['query']}")
        for angle, rotate in [(0, None), (90, cv2.ROTATE_90_CLOCKWISE),
                              (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
            frame = img if rotate is None else cv2.rotate(img, rotate)
            outcome = model.match(ref, frame)
            if outcome.execution_error:
                rec["execution_error"] = outcome.error
                break
            if not outcome.success or outcome.homography is None:
                continue
            try:
                center = np.float32([[[frame.shape[1] / 2, frame.shape[0] / 2]]])
                pt = cv2.perspectiveTransform(center, np.linalg.inv(outcome.homography))[0, 0]
            except np.linalg.LinAlgError:
                continue
            bounds = item["bounds"]
            east = bounds["min_east"] + float(pt[0]) * (bounds["max_east"] - bounds["min_east"]) / ref.shape[1]
            north = bounds["max_north"] - float(pt[1]) * (bounds["max_north"] - bounds["min_north"]) / ref.shape[0]
            if not (bounds["min_east"] <= east <= bounds["max_east"]
                    and bounds["min_north"] <= north <= bounds["max_north"]):
                continue
            rec.update(success=True, error_m=math.hypot(east - item["truth_east"], north - item["truth_north"]),
                       angle=angle, inliers=outcome.inliers, inlier_ratio=outcome.inlier_ratio,
                       coverage=outcome.coverage)
            break
        rec["elapsed_s"] = round(time.monotonic() - started, 3)
        results.append(rec)
        print(len(results), rec, flush=True)
        args.out.write_text(json.dumps({"note": payload["note"], "pairs": results}, indent=2))
        if rec.get("execution_error"):
            raise RuntimeError(f"matcher execution failed on {item['query']}: {rec['execution_error']}")


if __name__ == "__main__":
    main()
