#!/usr/bin/env python3
"""Smoke-test matchers (classical + deep) on a synthetic satellite/UAV pair.

    ~/env_flight/bin/python scripts/deep_matcher_smoke.py
    ~/env_flight/bin/python scripts/deep_matcher_smoke.py --matchers loftr --max-dim 512

Prints inliers / ratio / coverage / latency for each matcher. Deep matchers need
torch + kornia and, on first run, download their weights to
``~/.cache/torch/hub/checkpoints/``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

# Allow `python scripts/deep_matcher_smoke.py` straight from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from navlab.lab.matchers import get_matcher
from navlab.lab.synthetic import make_queries, make_scene


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matchers", default="sift,loftr,lightglue")
    p.add_argument("--max-dim", type=int, default=640)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mode", default="gray", choices=["gray", "edges"])
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene = make_scene(root, size=args.size, seed=args.seed, gsd_m=0.5)
        queries = make_queries(
            root,
            scene,
            count=1,
            seed=args.seed,
            altitude_m=250.0,
            query_size=(512, 342),
        )
        query = queries[0]
        print(f"scene={scene['path']}\nquery={query.path}\n")

        failures = 0
        for name in [n.strip() for n in args.matchers.split(",") if n.strip()]:
            try:
                matcher = get_matcher(name, mode=args.mode, max_dim=args.max_dim)
            except Exception as error:  # noqa: BLE001
                print(f"{name:10s} UNAVAILABLE  {type(error).__name__}: {error}")
                failures += 1
                continue
            started = time.time()
            outcome = matcher.match(scene["path"], query.path)
            elapsed = (time.time() - started) * 1000.0
            status = "OK  " if outcome.success else "FAIL"
            error = f"  error={outcome.error}" if getattr(outcome, "error", None) else ""
            print(
                f"{name:10s} {status} inliers={outcome.inliers:4d} "
                f"ratio={outcome.inlier_ratio:5.3f} coverage={outcome.coverage:5.3f} "
                f"reproj={outcome.reproj} scale={outcome.scale} "
                f"latency={elapsed:7.1f}ms{error}"
            )
            if not outcome.success:
                failures += 1

        return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
