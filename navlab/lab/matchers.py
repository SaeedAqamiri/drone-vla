from __future__ import annotations

import time
from pathlib import Path

import cv2

from sat2aerial.matching import match_images

from .types import MatchOutcome

CLASSICAL = ("sift", "orb", "akaze")
DEEP = ("loftr", "lightglue", "superpoint_lightglue")


class ClassicalMatcher:
    def __init__(
        self,
        detector: str = "sift",
        mode: str = "edges",
        max_dim: int = 1024,
        ratio: float = 0.75,
        ransac_px: float = 5.0,
        min_inliers: int = 15,
        min_ratio: float = 0.15,
        min_coverage: float = 0.03,
        rng_seed: int = 0,
    ):
        if detector not in CLASSICAL:
            raise ValueError(f"Unknown detector {detector!r}")
        self.detector = detector
        self.mode = mode
        self.max_dim = max_dim
        self.ratio = ratio
        self.ransac_px = ransac_px
        self.min_inliers = min_inliers
        self.min_ratio = min_ratio
        self.min_coverage = min_coverage
        self.rng_seed = int(rng_seed)

    def match(self, ref: str | Path, query: str | Path) -> MatchOutcome:
        started = time.time()
        cv2.setRNGSeed(self.rng_seed)
        try:
            result, homography = match_images(
                ref,
                query,
                mode=self.mode,
                max_dim=self.max_dim,
                ratio=self.ratio,
                ransac_px=self.ransac_px,
                min_inliers=self.min_inliers,
                min_ratio=self.min_ratio,
                min_coverage=self.min_coverage,
                detector=self.detector,
            )
        except Exception as error:
            return MatchOutcome(
                ref=str(ref),
                query=str(query),
                detector=self.detector,
                inliers=0,
                inlier_ratio=0.0,
                coverage=0.0,
                reproj=None,
                scale=None,
                rotation=None,
                latency_ms=(time.time() - started) * 1000.0,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )
        return MatchOutcome(
            ref=str(ref),
            query=str(query),
            detector=self.detector,
            inliers=int(result.inliers),
            inlier_ratio=float(result.inlier_ratio),
            coverage=float(result.coverage),
            reproj=result.median_reproj,
            scale=result.scale,
            rotation=result.rotation_deg,
            latency_ms=(time.time() - started) * 1000.0,
            success=bool(result.success),
            homography=homography,
        )


def get_matcher(
    name: str = "sift",
    *,
    mode: str = "edges",
    max_dim: int = 1024,
    min_inliers: int = 15,
    min_ratio: float = 0.15,
    min_coverage: float = 0.03,
    ratio: float = 0.75,
    ransac_px: float = 5.0,
    rng_seed: int = 0,
) -> ClassicalMatcher:
    key = name.lower().strip()
    if key in DEEP:
        from .deep import DeepMatcher

        return DeepMatcher(
            kind=key,
            max_dim=max_dim,
            min_inliers=min_inliers,
            min_ratio=min_ratio,
            min_coverage=min_coverage,
            ransac_px=ransac_px,
        )
    if key not in CLASSICAL:
        raise ValueError(
            f"Unknown matcher {name!r}; available: "
            f"{', '.join(CLASSICAL + DEEP)}"
        )
    return ClassicalMatcher(
        detector=key,
        mode=mode,
        max_dim=max_dim,
        ratio=ratio,
        ransac_px=ransac_px,
        min_inliers=min_inliers,
        min_ratio=min_ratio,
        min_coverage=min_coverage,
        rng_seed=rng_seed,
    )
