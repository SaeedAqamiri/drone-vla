"""Experimental inter-frame planar motion, anchored by an image-to-map fit.

This is not inertial odometry: pitch, terrain, parallax, feature loss and
changes in altitude can move the projected image center without vehicle motion.
"""
from __future__ import annotations

import math
import time
import cv2
import numpy as np


class VisualGroundTracker:
    def __init__(self, *, max_dim=768, min_inliers=25, min_ratio=.4,
                 min_coverage=.05, max_reproj_px=2.5, max_speed_mps=200):
        self.max_dim = max_dim
        self.min_inliers, self.min_ratio = min_inliers, min_ratio
        self.min_coverage, self.max_reproj_px, self.max_speed_mps = min_coverage, max_reproj_px, max_speed_mps
        self.sift = cv2.SIFT_create(nfeatures=2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.previous = None
        self.pending = None
        self.last_diagnostic = {"reason": "unanchored"}

    def _features(self, image):
        h, w = image.shape[:2]
        factor = min(1.0, self.max_dim / max(h, w))
        resized = cv2.resize(image, (max(1, round(w*factor)), max(1, round(h*factor))),
                             interpolation=cv2.INTER_AREA) if factor < 1 else image
        points, desc = self.sift.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), None)
        return {"image_shape": (h, w), "shape": resized.shape[:2], "factor": factor,
                "points": points, "desc": desc}

    @staticmethod
    def _center(matrix, shape):
        h, w = shape
        value = matrix @ np.array([w / 2, h / 2, 1.0])
        if not np.isfinite(value).all() or abs(value[2]) < 1e-9:
            raise ValueError("center projection invalid")
        return value[:2] / value[2]

    def predict(self, image, timestamp):
        """Estimate ENU increment from immediately preceding call, or None."""
        started = time.monotonic()
        current = self._features(image)
        current["timestamp"] = float(timestamp)
        self.pending = current
        if self.previous is None:
            self.last_diagnostic = {"reason": "unanchored"}
            return None
        previous = self.previous
        try:
            dt = float(timestamp) - previous["timestamp"]
            if dt <= 0 or current["desc"] is None or previous["desc"] is None:
                raise ValueError("invalid time or missing descriptors")
            pairs = self.matcher.knnMatch(current["desc"], previous["desc"], k=2)
            good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .75*b.distance]
            if len(good) < self.min_inliers:
                raise ValueError("few correspondences")
            src = np.float32([current["points"][m.queryIdx].pt for m in good])
            dst = np.float32([previous["points"][m.trainIdx].pt for m in good])
            h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3, maxIters=5000, confidence=.999)
            if h is None or mask is None:
                raise ValueError("homography failed")
            inliers = mask.ravel().astype(bool)
            count = int(inliers.sum()); ratio = count / len(good)
            coverage = cv2.contourArea(cv2.convexHull(src[inliers])) / np.prod(current["shape"]) if count >= 3 else 0
            predicted = cv2.perspectiveTransform(src[inliers, None], h)[:, 0]
            reproj = float(np.median(np.linalg.norm(predicted-dst[inliers], axis=1))) if count else math.inf
            if count < self.min_inliers or ratio < self.min_ratio or coverage < self.min_coverage or reproj > self.max_reproj_px:
                raise ValueError("inter-frame geometry gate")
            # Feature coordinates are resized; return to original pixel coordinates.
            s_cur = np.diag([current["factor"], current["factor"], 1.0])
            s_prev_inv = np.diag([1/previous["factor"], 1/previous["factor"], 1.0])
            curr_to_prev = s_prev_inv @ h @ s_cur
            current["image_to_local"] = previous["image_to_local"] @ curr_to_prev
            prior_center = self._center(previous["image_to_local"], previous["image_shape"])
            current_center = self._center(current["image_to_local"], current["image_shape"])
            delta = current_center - prior_center
            if not np.isfinite(delta).all() or np.linalg.norm(delta) / dt > self.max_speed_mps:
                raise ValueError("motion speed gate")
            self.last_diagnostic = {"reason": "ok", "inliers": count, "ratio": ratio,
                                    "coverage": float(coverage), "reproj_px": reproj,
                                    "dx_m": float(delta[0]), "dy_m": float(delta[1]),
                                    "elapsed_ms": (time.monotonic()-started)*1000}
            return float(delta[0]), float(delta[1])
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            self.last_diagnostic = {"reason": str(error), "elapsed_ms": (time.monotonic()-started)*1000}
            return None

    def finish(self, image_to_local):
        """Anchor on a new map fix, or carry a valid previous projection."""
        if self.pending is None:
            raise RuntimeError("predict must precede finish")
        if image_to_local is not None:
            self.pending["image_to_local"] = image_to_local
        self.previous = self.pending if "image_to_local" in self.pending else None
        self.pending = None
