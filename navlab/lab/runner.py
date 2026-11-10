from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .candidates import CandidateSelector, Prior
from .matchers import get_matcher
from .metrics import compute_summary
from .tiles import TileStore
from .types import FrameResult, MatchOutcome, QueryFrame


@dataclass
class RunnerConfig:
    matcher: str = "sift"
    mode: str = "edges"
    max_dim: int = 1024
    max_candidates: int = 8
    bootstrap_candidates: int = 20
    base_radius: float = 120.0
    max_radius: float = 1200.0
    sigma_scale: float = 3.0
    global_fallback: bool = True
    attempt_every: int = 1
    min_inliers: int = 15
    min_ratio: float = 0.15
    min_coverage: float = 0.03
    ratio: float = 0.75
    ransac_px: float = 5.0
    max_jump_m: float = 250.0
    scale_tolerance: float = 1.0
    camera_fov_deg: float = 73.0
    fix_sigma: float = 6.0
    max_error_m: float = 30.0
    confirmation: int = 1
    confirmation_radius_m: float = 200.0
    rng_seed: int = 0
    dropout_start: int | None = None
    dropout_length: int = 0
    reference_kind: str | None = None


@dataclass
class RunResult:
    frames: list[FrameResult]
    config: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "summary": self.summary,
            "frames": [frame.as_dict() for frame in self.frames],
        }


class LabRunner:
    def __init__(
        self,
        store: TileStore,
        queries: list[QueryFrame],
        config: RunnerConfig | None = None,
        references: str | Path | None = None,
    ):
        self.base_store = store
        self.store = store.with_references(references) if references else store
        self.queries = list(queries)
        self.config = config or RunnerConfig()

    def _query_shape(self, query: QueryFrame) -> tuple[int, int] | None:
        image = cv2.imread(str(query.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image.shape[:2]

    def _expected_scale(
        self,
        query: QueryFrame,
        tile: dict[str, Any],
        query_shape: tuple[int, int],
    ) -> float | None:
        if query.altitude <= 0:
            return None
        height, width = query_shape
        footprint_w = 2.0 * query.altitude * math.tan(
            math.radians(self.config.camera_fov_deg) / 2.0
        )
        query_gsd = footprint_w / float(max(width, 1))
        query_factor = min(
            1.0, self.config.max_dim / float(max(height, width, 1))
        )
        tile_px = float(tile.get("tile_px") or self.config.max_dim)
        ref_factor = min(1.0, self.config.max_dim / tile_px)
        ref_gsd = float(tile.get("gsd") or self.store.gsd) / ref_factor
        query_pixel_gsd = query_gsd / query_factor
        if query_pixel_gsd <= 0:
            return None
        return ref_gsd / query_pixel_gsd

    def _refine_position(
        self,
        tile: dict[str, Any],
        outcome: MatchOutcome,
        query_shape: tuple[int, int],
    ) -> tuple[float, float]:
        center_east = float(tile["east"])
        center_north = float(tile["north"])
        homography = outcome.homography
        if homography is None:
            return center_east, center_north
        height, width = query_shape
        query_factor = min(
            1.0, self.config.max_dim / float(max(height, width, 1))
        )
        query_width = width * query_factor
        query_height = height * query_factor
        center = np.array(
            [[[query_width / 2.0, query_height / 2.0]]], dtype=np.float32
        )
        try:
            inverse = np.linalg.inv(homography)
            ref_point = cv2.perspectiveTransform(center, inverse)[0, 0]
        except np.linalg.LinAlgError:
            return center_east, center_north
        tile_px = float(tile.get("tile_px") or self.config.max_dim)
        ref_factor = min(1.0, self.config.max_dim / tile_px)
        ref_x = float(ref_point[0]) / ref_factor
        ref_y = float(ref_point[1]) / ref_factor
        bounds = tile["bounds"]
        gsd = float(tile.get("gsd") or self.store.gsd)
        east = float(bounds["min_east"]) + ref_x * gsd
        north = float(bounds["max_north"]) - ref_y * gsd
        return east, north

    @staticmethod
    def _score(outcome: MatchOutcome) -> tuple:
        reproj = outcome.reproj if outcome.reproj is not None else 1e9
        return (
            bool(outcome.success),
            int(outcome.inliers),
            float(outcome.coverage),
            -reproj,
        )

    def run(self) -> RunResult:
        matcher = get_matcher(
            self.config.matcher,
            mode=self.config.mode,
            max_dim=self.config.max_dim,
            min_inliers=self.config.min_inliers,
            min_ratio=self.config.min_ratio,
            min_coverage=self.config.min_coverage,
            ratio=self.config.ratio,
            ransac_px=self.config.ransac_px,
            rng_seed=self.config.rng_seed,
        )
        selector = CandidateSelector(
            self.store,
            base_radius=self.config.base_radius,
            max_radius=self.config.max_radius,
            max_candidates=self.config.max_candidates,
            sigma_scale=self.config.sigma_scale,
            global_fallback=self.config.global_fallback,
        )
        prior = Prior()
        pending: tuple[float, float] | None = None
        frames: list[FrameResult] = []
        dropout_start = self.config.dropout_start
        dropout_length = max(0, self.config.dropout_length)

        for index, query in enumerate(self.queries):
            timestamp = float(query.timestamp)
            if query.has_truth:
                truth_east, truth_north = self.store.to_local(query.lon, query.lat)
            else:
                truth_east, truth_north = None, None
            predicted = prior.predict(timestamp)
            prior_east = predicted[0] if predicted else None
            prior_north = predicted[1] if predicted else None
            prior_error = (
                math.hypot(prior_east - truth_east, prior_north - truth_north)
                if predicted is not None
                and truth_east is not None
                and truth_north is not None
                else None
            )

            in_dropout = bool(
                dropout_start is not None
                and dropout_length > 0
                and dropout_start <= index < dropout_start + dropout_length
            )
            cadence_skip = index > 0 and index % max(1, self.config.attempt_every) != 0
            if in_dropout or cadence_skip:
                if in_dropout:
                    pending = None
                if prior.valid:
                    prior.decay(timestamp)
                frames.append(
                    FrameResult(
                        index=index,
                        query_id=query.id,
                        timestamp=timestamp,
                        truth_east=truth_east,
                        truth_north=truth_north,
                        prior_east=prior_east,
                        prior_north=prior_north,
                        candidates=0,
                        attempted=False,
                        matched=False,
                        accepted=False,
                        best_tile=None,
                        best_east=None,
                        best_north=None,
                        error_m=None,
                        prior_error_m=prior_error,
                        latency_ms=0.0,
                        reason="dropout" if in_dropout else "cadence",
                    )
                )
                continue

            query_shape = self._query_shape(query)
            if query_shape is None:
                pending = None
                frames.append(
                    FrameResult(
                        index=index,
                        query_id=query.id,
                        timestamp=timestamp,
                        truth_east=truth_east,
                        truth_north=truth_north,
                        prior_east=prior_east,
                        prior_north=prior_north,
                        candidates=0,
                        attempted=False,
                        matched=False,
                        accepted=False,
                        best_tile=None,
                        best_east=None,
                        best_north=None,
                        error_m=None,
                        prior_error_m=prior_error,
                        latency_ms=0.0,
                        reason="missing_query",
                    )
                )
                continue

            if prior.valid:
                candidates, _ = selector.select(
                    prior,
                    timestamp,
                    query_path=query.path,
                    kind=self.config.reference_kind,
                )
            else:
                candidates = selector.global_candidates(
                    query.path,
                    kind=self.config.reference_kind,
                    limit=self.config.bootstrap_candidates,
                )

            best_tile: dict[str, Any] | None = None
            best_outcome: MatchOutcome | None = None
            latency = 0.0
            for tile in candidates:
                outcome = matcher.match(self.store.path_for(tile), query.path)
                latency += outcome.latency_ms
                if best_outcome is None or self._score(outcome) > self._score(
                    best_outcome
                ):
                    best_outcome = outcome
                    best_tile = tile

            if best_outcome is None or best_tile is None:
                pending = None
                if prior.valid:
                    prior.decay(timestamp)
                frames.append(
                    FrameResult(
                        index=index,
                        query_id=query.id,
                        timestamp=timestamp,
                        truth_east=truth_east,
                        truth_north=truth_north,
                        prior_east=prior_east,
                        prior_north=prior_north,
                        candidates=len(candidates),
                        attempted=True,
                        matched=False,
                        accepted=False,
                        best_tile=None,
                        best_east=None,
                        best_north=None,
                        error_m=None,
                        prior_error_m=prior_error,
                        latency_ms=latency,
                        reason="no_candidates",
                    )
                )
                continue

            matched = bool(best_outcome.success)
            accepted = False
            reason = "no_match" if not matched else "ok"
            position_east, position_north = self._refine_position(
                best_tile, best_outcome, query_shape
            )
            error_m = (
                math.hypot(position_east - truth_east, position_north - truth_north)
                if truth_east is not None and truth_north is not None
                else None
            )

            if matched:
                expected_scale = self._expected_scale(query, best_tile, query_shape)
                if (
                    self.config.scale_tolerance > 0
                    and expected_scale
                    and best_outcome.scale
                    and expected_scale > 0
                ):
                    ratio = float(best_outcome.scale) / expected_scale
                    limit = 1.0 + self.config.scale_tolerance
                    if ratio < 1.0 / limit or ratio > limit:
                        reason = "scale"
                    else:
                        accepted = True
                else:
                    accepted = True

            if accepted and predicted is not None:
                jump = math.hypot(position_east - predicted[0], position_north - predicted[1])
                if jump > self.config.max_jump_m + 2.0 * predicted[2]:
                    accepted = False
                    reason = "jump"
                else:
                    reason = "ok"

            if not matched or not accepted:
                pending = None
            elif self.config.confirmation > 1:
                if pending is None:
                    pending = (position_east, position_north)
                    accepted = False
                    reason = "pending"
                elif (
                    math.hypot(
                        position_east - pending[0],
                        position_north - pending[1],
                    )
                    <= self.config.confirmation_radius_m
                ):
                    pending = None
                else:
                    pending = (position_east, position_north)
                    accepted = False
                    reason = "pending_reset"

            if accepted:
                prior.update(
                    position_east,
                    position_north,
                    timestamp,
                    sigma=self.config.fix_sigma,
                )
            elif prior.valid:
                prior.decay(timestamp)

            frames.append(
                FrameResult(
                    index=index,
                    query_id=query.id,
                    timestamp=timestamp,
                    truth_east=truth_east,
                    truth_north=truth_north,
                    prior_east=prior_east,
                    prior_north=prior_north,
                    candidates=len(candidates),
                    attempted=True,
                    matched=matched,
                    accepted=accepted,
                    best_tile=str(best_tile["id"]),
                    best_east=position_east if accepted else None,
                    best_north=position_north if accepted else None,
                    error_m=error_m if accepted else None,
                    prior_error_m=prior_error,
                    latency_ms=latency,
                    reason=reason,
                    match=best_outcome.as_dict(),
                )
            )

        summary = compute_summary(frames, self.config)
        return RunResult(
            frames=frames,
            config=asdict(self.config),
            summary=summary,
        )
