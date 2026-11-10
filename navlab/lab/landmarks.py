"""Experimental map-only visual-anchor evidence, not semantic landmarks.

These are spatially separated SIFT corners selected from the reference map.
Agreement of dense matches with several anchors is not independent evidence
and does not by itself prove correct geolocation or flight readiness.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .deep import _read_rgb


def map_anchors(reference: np.ndarray, *, max_anchors: int = 30, separation_px: float = 55) -> np.ndarray:
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    keypoints = cv2.SIFT_create(nfeatures=1500).detect(gray, None)
    selected = []
    for point in sorted(keypoints, key=lambda p: p.response, reverse=True):
        xy = np.array(point.pt)
        if all(np.linalg.norm(xy - prior) > separation_px for prior in selected):
            selected.append(xy)
        if len(selected) == max_anchors:
            break
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def anchor_support(anchors, ref_points, query_points, homography, *, radius_px=25,
                   reprojection_px=5, min_points=8):
    if homography is None or ref_points is None or query_points is None:
        return [], 0
    if len(ref_points) == 0:
        return [0] * len(anchors), 0
    mapped = cv2.perspectiveTransform(np.float32(ref_points).reshape(-1, 1, 2),
                                      homography).reshape(-1, 2)
    inliers = np.asarray(ref_points)[np.linalg.norm(mapped - query_points, axis=1) < reprojection_px]
    counts = [int((np.linalg.norm(inliers - point, axis=1) < radius_px).sum()) for point in anchors]
    return counts, sum(count >= min_points for count in counts)


class LandmarkVerifiedMatcher:
    """Optional experimental gate on an existing matcher with match_with_points."""
    def __init__(self, matcher, *, min_supported=4, max_dim=768):
        if not hasattr(matcher, 'match_with_points') or min_supported < 0:
            raise ValueError('point-producing matcher and nonnegative threshold required')
        self.matcher, self.min_supported, self.max_dim = matcher, min_supported, max_dim
        self._anchor_cache = {}

    def match(self, ref, query):
        outcome, ref_points, query_points = self.matcher.match_with_points(ref, query)
        if outcome.execution_error:
            return outcome
        try:
            key = None
            if isinstance(ref, (str, Path)):
                stat = Path(ref).stat()
                key = (str(ref), stat.st_size, stat.st_mtime_ns)
            anchors = self._anchor_cache.get(key) if key is not None else None
            if anchors is None:
                reference = _read_rgb(ref, self.max_dim)
                if reference is None:
                    raise ValueError('landmark reference read failed')
                anchors = map_anchors(reference[0])
                if key is not None:
                    if len(self._anchor_cache) >= 256:
                        self._anchor_cache.pop(next(iter(self._anchor_cache)))
                    self._anchor_cache[key] = anchors
            _, supported = anchor_support(anchors, ref_points, query_points, outcome.homography)
            return replace(outcome, success=outcome.success and supported >= self.min_supported,
                           landmark_supported=supported, landmark_total=len(anchors))
        except Exception as error:
            return replace(outcome, success=False, execution_error=True,
                           error=f'landmark evidence: {type(error).__name__}: {error}')
