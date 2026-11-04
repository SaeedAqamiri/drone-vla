from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class QueryFrame:
    id: str
    path: Path
    timestamp: float
    lat: float
    lon: float
    altitude: float
    yaw: float
    meta: dict[str, Any] = field(default_factory=dict)
    has_truth: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "timestamp": self.timestamp,
            "lat": self.lat,
            "lon": self.lon,
            "altitude": self.altitude,
            "yaw": self.yaw,
            "meta": self.meta,
            "has_truth": self.has_truth,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueryFrame":
        return cls(
            id=str(payload["id"]),
            path=Path(payload["path"]),
            timestamp=float(payload["timestamp"]),
            lat=float(payload["lat"]),
            lon=float(payload["lon"]),
            altitude=float(payload.get("altitude", 0.0)),
            yaw=float(payload.get("yaw", 0.0)),
            meta=dict(payload.get("meta") or {}),
            has_truth=bool(payload.get("has_truth", True)),
        )


@dataclass
class MatchOutcome:
    ref: str
    query: str
    detector: str
    inliers: int
    inlier_ratio: float
    coverage: float
    reproj: float | None
    scale: float | None
    rotation: float | None
    latency_ms: float
    success: bool
    error: str = ""
    homography: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "query": self.query,
            "detector": self.detector,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "coverage": round(self.coverage, 4),
            "reproj": None if self.reproj is None else round(self.reproj, 4),
            "scale": None if self.scale is None else round(self.scale, 4),
            "rotation": None if self.rotation is None else round(self.rotation, 3),
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class FrameResult:
    index: int
    query_id: str
    timestamp: float
    truth_east: float | None
    truth_north: float | None
    prior_east: float | None
    prior_north: float | None
    candidates: int
    attempted: bool
    matched: bool
    accepted: bool
    best_tile: str | None
    best_east: float | None
    best_north: float | None
    error_m: float | None
    prior_error_m: float | None
    latency_ms: float
    reason: str
    match: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "query_id": self.query_id,
            "timestamp": self.timestamp,
            "truth_east": self.truth_east,
            "truth_north": self.truth_north,
            "prior_east": self.prior_east,
            "prior_north": self.prior_north,
            "candidates": self.candidates,
            "attempted": self.attempted,
            "matched": self.matched,
            "accepted": self.accepted,
            "best_tile": self.best_tile,
            "best_east": self.best_east,
            "best_north": self.best_north,
            "error_m": None if self.error_m is None else round(self.error_m, 3),
            "prior_error_m": None
            if self.prior_error_m is None
            else round(self.prior_error_m, 3),
            "latency_ms": round(self.latency_ms, 2),
            "reason": self.reason,
            "match": self.match,
        }
