from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Pair:
    site_id: str
    satellite: Path
    aerials: list[Path] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchResult:
    ref: str
    target: str
    mode: str
    keypoints_ref: int
    keypoints_target: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    median_reproj: float | None
    scale: float | None
    rotation_deg: float | None
    coverage: float
    success: bool
    detector: str = "sift"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "target": self.target,
            "mode": self.mode,
            "detector": self.detector,
            "keypoints_ref": self.keypoints_ref,
            "keypoints_target": self.keypoints_target,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "median_reproj": None
            if self.median_reproj is None
            else round(self.median_reproj, 4),
            "scale": None if self.scale is None else round(self.scale, 4),
            "rotation_deg": None
            if self.rotation_deg is None
            else round(self.rotation_deg, 3),
            "coverage": round(self.coverage, 4),
            "success": self.success,
        }
