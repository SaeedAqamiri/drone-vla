from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .tiles import TileStore


@dataclass
class Prior:
    east: float | None = None
    north: float | None = None
    vx: float = 0.0
    vy: float = 0.0
    timestamp: float | None = None
    sigma: float = 30.0
    valid: bool = False

    def predict(self, timestamp: float) -> tuple[float, float, float] | None:
        if not self.valid or self.east is None or self.north is None:
            return None
        if self.timestamp is None:
            return float(self.east), float(self.north), float(self.sigma)
        dt = max(0.0, float(timestamp) - float(self.timestamp))
        return (
            float(self.east) + self.vx * dt,
            float(self.north) + self.vy * dt,
            float(self.sigma),
        )

    def update(
        self,
        east: float,
        north: float,
        timestamp: float,
        *,
        sigma: float = 10.0,
        max_speed: float = 80.0,
    ) -> None:
        if self.valid and self.timestamp is not None:
            dt = float(timestamp) - float(self.timestamp)
            if dt > 1e-6 and self.east is not None and self.north is not None:
                vx = (float(east) - float(self.east)) / dt
                vy = (float(north) - float(self.north)) / dt
                speed = (vx * vx + vy * vy) ** 0.5
                if speed <= max_speed:
                    self.vx = 0.5 * self.vx + 0.5 * vx
                    self.vy = 0.5 * self.vy + 0.5 * vy
        self.east = float(east)
        self.north = float(north)
        self.timestamp = float(timestamp)
        self.sigma = float(sigma)
        self.valid = True

    def decay(self, timestamp: float, drift_mps: float = 3.0) -> None:
        if not self.valid or self.timestamp is None:
            return
        dt = max(0.0, float(timestamp) - float(self.timestamp))
        self.sigma = max(self.sigma, 10.0) + drift_mps * dt
        self.timestamp = float(timestamp)


def _descriptor(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small - small.mean()) / (small.std() + 1e-6)
    return small.ravel()


class CandidateSelector:
    def __init__(
        self,
        store: TileStore,
        *,
        base_radius: float = 40.0,
        max_radius: float = 400.0,
        max_candidates: int = 8,
        sigma_scale: float = 3.0,
        global_fallback: bool = True,
    ):
        self.store = store
        self.base_radius = float(base_radius)
        self.max_radius = float(max_radius)
        self.max_candidates = int(max_candidates)
        self.sigma_scale = float(sigma_scale)
        self.global_fallback = bool(global_fallback)
        self._tiles: list[dict[str, Any]] = []
        self._descriptors: np.ndarray | None = None
        self._descriptor_keys: list[str] = []

    def _ensure_descriptors(self, kind: str | None) -> None:
        keys = [str(tile["id"]) for tile in self.store.all_tiles(kind)]
        if self._descriptors is not None and keys == self._descriptor_keys:
            return
        tiles = self.store.all_tiles(kind)
        rows: list[np.ndarray] = []
        kept: list[dict[str, Any]] = []
        for tile in tiles:
            descriptor = _descriptor(self.store.path_for(tile))
            if descriptor is None:
                continue
            rows.append(descriptor)
            kept.append(tile)
        self._tiles = kept
        self._descriptor_keys = [str(tile["id"]) for tile in kept]
        self._descriptors = (
            np.stack(rows).astype(np.float32)
            if rows
            else np.zeros((0, 1024), np.float32)
        )

    def global_candidates(
        self,
        query_path: str | Path,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_descriptors(kind)
        if self._descriptors is None or len(self._tiles) == 0:
            return []
        query = _descriptor(Path(query_path))
        if query is None:
            return []
        distances = ((self._descriptors - query) ** 2).sum(axis=1)
        order = np.argsort(distances)[: limit or self.max_candidates]
        return [self._tiles[int(index)] for index in order]

    def select(
        self,
        prior: Prior,
        timestamp: float,
        *,
        query_path: str | Path | None = None,
        kind: str | None = None,
    ) -> tuple[list[dict[str, Any]], tuple[float, float, float, float] | None]:
        predicted = prior.predict(timestamp)
        if predicted is None:
            if query_path is not None:
                return self.global_candidates(query_path, kind=kind), None
            return self.store.all_tiles(kind)[: self.max_candidates], None
        east, north, sigma = predicted
        radius = min(self.max_radius, self.base_radius + self.sigma_scale * sigma)
        candidates = self.store.candidates(
            east, north, radius, limit=self.max_candidates, kind=kind
        )
        if not candidates and self.global_fallback:
            radius = self.max_radius
            candidates = self.store.candidates(
                east, north, radius, limit=self.max_candidates, kind=kind
            )
        if not candidates and self.global_fallback and query_path is not None:
            candidates = self.global_candidates(query_path, kind=kind)
        return candidates, (east, north, sigma, radius)
