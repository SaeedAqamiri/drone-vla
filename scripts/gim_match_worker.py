#!/usr/bin/env python3
"""Research-only JSON-lines worker for a pre-installed GIM-DKM matcher.

Input is images encoded by OpenCV, not file paths supplied by remote clients.
The worker must run over a trusted local connection (e.g. SSH), not a public API.
"""
from __future__ import annotations

import base64
import json
import sys
import types
from collections import OrderedDict

import cv2
import numpy as np


def main():
    # The server's copied Kornia uses kornia_rs only for image IO, which this
    # worker never calls. Avoid installing into the user's SGLang environment.
    try:
        import kornia_rs  # noqa: F401
    except ImportError:
        sys.modules["kornia_rs"] = types.ModuleType("kornia_rs")
    from navlab.lab.gim import GIMDKMMatcher

    cv2.setNumThreads(2)
    model = GIMDKMMatcher(max_dim=768)
    cache: OrderedDict[str, np.ndarray] = OrderedDict()
    print(json.dumps({"ready": True, "matcher": "gim_dkm", "max_dim": 768}), flush=True)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            images = []
            for name in ("ref", "query"):
                key = message[f"{name}_key"]
                encoded = message.get(f"{name}_png")
                if encoded is not None:
                    binary = base64.b64decode(encoded, validate=True)
                    image = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"invalid {name} PNG")
                    cache[key] = image
                if key not in cache:
                    raise ValueError(f"missing cached {name}: {key}")
                cache.move_to_end(key)
                images.append(cache[key])
            while len(cache) > 256:
                cache.popitem(last=False)
            if message.get("want_points", False):
                result, src, dst = model.match_with_points(*images)
            else:
                result, src, dst = model.match(*images), None, None
            response = result.as_dict()
            # Preserve precision for any stricter client-side acceptance gate.
            response["coverage"], response["inlier_ratio"] = result.coverage, result.inlier_ratio
            response["homography"] = result.homography.tolist() if result.homography is not None else None
            if src is not None and dst is not None:
                response["ref_points"], response["query_points"] = src.tolist(), dst.tolist()
        except Exception as error:
            response = {"execution_error": True, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
