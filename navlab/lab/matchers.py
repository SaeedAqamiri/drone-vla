from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from sat2aerial.matching import match_images, match_arrays

from .types import MatchOutcome

CLASSICAL = ("sift", "orb", "akaze")
DEEP = ("loftr", "lightglue", "disk_lightglue", "loretta", "redi", "gim_loftr", "gim_dkm")


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
            in_memory = isinstance(ref, np.ndarray) or isinstance(query, np.ndarray)
            first = ref if not in_memory or isinstance(ref, np.ndarray) else cv2.imread(str(ref), cv2.IMREAD_COLOR)
            second = query if not in_memory or isinstance(query, np.ndarray) else cv2.imread(str(query), cv2.IMREAD_COLOR)
            fn = match_arrays if in_memory else match_images
            result, homography = fn(
                first if in_memory else ref,
                second if in_memory else query,
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
                ref="<array>" if isinstance(ref, np.ndarray) else str(ref),
                query="<array>" if isinstance(query, np.ndarray) else str(query),
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
                execution_error=True,
            )
        return MatchOutcome(
            ref="<array>" if isinstance(ref, np.ndarray) else str(ref),
            query="<array>" if isinstance(query, np.ndarray) else str(query),
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
        if key == "gim_dkm":
            from .gim import GIMDKMMatcher
            return GIMDKMMatcher(max_dim=max_dim, min_inliers=min_inliers,
                                 min_ratio=min_ratio, min_coverage=min_coverage,
                                 ransac_px=ransac_px)
        if key == "gim_loftr":
            from .gim import GIMLoFTRMatcher
            return GIMLoFTRMatcher(max_dim=max_dim, min_inliers=min_inliers,
                                   min_ratio=min_ratio, min_coverage=min_coverage,
                                   ransac_px=ransac_px)
        if key == "loretta":
            from .loretta import LoRettaMatcher
            return LoRettaMatcher(max_dim=max_dim, min_inliers=min_inliers,
                                  min_ratio=min_ratio, min_coverage=min_coverage,
                                  ransac_px=ransac_px)
        if key == "redi":
            from .redi import REDIMatcher
            return REDIMatcher(max_dim=max_dim, min_inliers=min_inliers,
                               min_ratio=min_ratio, min_coverage=min_coverage,
                               ransac_px=ransac_px)
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
