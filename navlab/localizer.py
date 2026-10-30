from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np


class MapLocalizer:
    def __init__(self, map_dir: str | Path):
        self.map_dir = Path(map_dir)
        manifest_path = self.map_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.frames = manifest.get("frames", [])
        self.z0 = float(manifest.get("z0") or 0.0)
        self.small: list[np.ndarray] = []
        self.keypoints: list[tuple] = []
        self.descriptors: list[np.ndarray | None] = []
        self.sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=0.005)
        for item in self.frames:
            image = cv2.imread(item["frame"], cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
            small = (small - small.mean()) / (small.std() + 1e-6)
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            self.small.append(small.ravel())
            self.keypoints.append(keypoints)
            self.descriptors.append(descriptors)
        self.small_matrix = (
            np.stack(self.small) if self.small else np.zeros((0, 1024), np.float32)
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.east = np.array([float(item["x"]) for item in self.frames], np.float32)
        self.north = np.array([float(item["y"]) for item in self.frames], np.float32)

    @property
    def size(self) -> int:
        return len(self.small)

    def locate(
        self,
        frame_path: str | Path,
        prior_east: float | None = None,
        prior_north: float | None = None,
        radius: float = 35.0,
        last_keyframe: int | None = None,
    ) -> dict | None:
        started = time.time()
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None or not self.frames:
            return None
        allowed = np.arange(len(self.frames))
        if last_keyframe is not None and last_keyframe >= 0:
            low = max(0, int(last_keyframe) - 10)
            high = min(len(self.frames), int(last_keyframe) + 50)
            allowed = np.arange(low, high)
        if prior_east is not None and prior_north is not None:
            near = (
                (self.east[allowed] - prior_east) ** 2
                + (self.north[allowed] - prior_north) ** 2
            ) <= radius**2
            if near.any():
                allowed = allowed[near]
        if len(allowed) == 0:
            allowed = np.arange(len(self.frames))
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = (small - small.mean()) / (small.std() + 1e-6)
        distances = ((self.small_matrix[allowed] - small.ravel()) ** 2).sum(axis=1)
        order = np.argsort(distances)[: min(8, len(allowed))]
        candidates = allowed[order]
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 4:
            return None

        best = None
        for index in candidates:
            map_des = self.descriptors[int(index)]
            if map_des is None or len(map_des) < 4:
                continue
            knn = self.matcher.knnMatch(descriptors, map_des, k=2)
            good = [
                pair[0]
                for pair in knn
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
            ]
            if len(good) < 4:
                continue
            src = np.float32([keypoints[m.queryIdx].pt for m in good])
            dst = np.float32([self.keypoints[int(index)][m.trainIdx].pt for m in good])
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            if homography is None or mask is None:
                continue
            inliers = int(mask.sum())
            inlier_dst = dst[mask.ravel().astype(bool)]
            coverage = 0.0
            if len(inlier_dst) >= 3:
                hull = cv2.convexHull(inlier_dst)
                coverage = float(cv2.contourArea(hull) / float(image.shape[0] * image.shape[1]))
            if inliers < 12 or coverage < 0.02:
                continue
            if best is None or inliers > best["inliers"]:
                item = self.frames[int(index)]
                best = {
                    "east": float(item["x"]),
                    "north": float(item["y"]),
                    "z": float(item["z"]) - self.z0,
                    "inliers": inliers,
                    "coverage": round(coverage, 3),
                    "keyframe": int(index),
                    "latency_ms": round((time.time() - started) * 1000.0, 1),
                }
        return best
