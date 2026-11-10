"""Incremental, autopilot-independent image-to-map localization.

This is a research prototype, not a flight-qualified navigation system. It
accepts BGR camera frames and monotonic timestamps, and does not use GPS truth
after initialization. The caller owns sensor fusion and all flight decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .candidates import CandidateSelector, Prior
from .matchers import get_matcher
from .runner import LabRunner, RunnerConfig
from .tiles import TileStore
from .types import MatchOutcome, QueryFrame


@dataclass
class NavigationResult:
    timestamp: float
    accepted: bool
    reason: str
    latitude: float | None = None
    longitude: float | None = None
    east_m: float | None = None
    north_m: float | None = None
    prior_east_m: float | None = None
    prior_north_m: float | None = None
    prior_sigma_m: float | None = None
    candidate_ids: tuple[str, ...] = ()
    candidate_bounds: tuple[dict[str, float], ...] = ()
    candidate_valid_fractions: tuple[float, ...] = ()
    best_tile: str | None = None
    query_rotation_deg: int | None = None
    match: dict[str, Any] | None = None
    execution_errors: tuple[str, ...] = ()
    # Experimental ground-plane projection, not a calibrated 6-DoF pose.
    image_to_local: np.ndarray | None = None


def _image_to_local(tile: dict, homography: np.ndarray, image_shape: tuple[int, int],
                    angle: int, max_dim: int, default_gsd: float) -> np.ndarray | None:
    """Map original image pixels to local east/north using the accepted plane fit."""
    h, w = image_shape
    rotations = {
        0: np.eye(3),
        90: np.array([[0, -1, h-1], [1, 0, 0], [0, 0, 1]], dtype=float),
        180: np.array([[-1, 0, w-1], [0, -1, h-1], [0, 0, 1]], dtype=float),
        270: np.array([[0, 1, 0], [-1, 0, w-1], [0, 0, 1]], dtype=float),
    }
    qfactor = min(1.0, max_dim / float(max(h, w)))
    scale_query = np.diag([qfactor, qfactor, 1.0])
    tile_px = float(tile.get("tile_px") or max_dim)
    ref_factor = min(1.0, max_dim / tile_px)
    metres_per_pixel = float(tile.get("gsd") or default_gsd) / ref_factor
    bounds = tile["bounds"]
    ref_to_local = np.array([[metres_per_pixel, 0, bounds["min_east"]],
                             [0, -metres_per_pixel, bounds["max_north"]], [0, 0, 1]], dtype=float)
    try:
        matrix = ref_to_local @ np.linalg.inv(homography) @ scale_query @ rotations[angle]
    except np.linalg.LinAlgError:
        return None
    return matrix if np.isfinite(matrix).all() else None


class VisualNavigator:
    """Track a known initial location with a local map; one call per frame.

    Map preparation and coordinates use TileStore's projected metre frame.
    ``initial_prior`` is mandatory; there is no implicit truth-based bootstrap.
    Generated references must pass the explicit geometry check before use.
    ``prior_sigma_m`` is a heuristic search radius parameter, not a calibrated
    navigation covariance or a guarantee that the correct tile is covered.
    """

    def __init__(self, store: TileStore, initial_prior: Prior, *,
                 config: RunnerConfig | None = None, references: str | Path | None = None,
                 matcher: Any | None = None, rotations: tuple[int, ...] = (0,),
                 cropper: Any | None = None):
        if not initial_prior.valid or initial_prior.east is None or initial_prior.north is None:
            raise ValueError("an explicit valid initial prior is required")
        self.config = config or RunnerConfig(mode="gray", confirmation=2,
                                             confirmation_radius_m=30, max_jump_m=60)
        if not rotations or any(angle not in (0, 90, 180, 270) for angle in rotations):
            raise ValueError("rotations must contain right-angle values in {0, 90, 180, 270}")
        self.rotations = tuple(rotations)
        if cropper is not None and references is not None:
            raise ValueError("adaptive crops do not support generated references")
        self.cropper = cropper
        self.store = store.with_references(references) if references else store
        for tile in self.store.tiles:
            if tile.get("kind") == "generated" and not tile.get("geometry_check", {}).get("accepted"):
                raise ValueError("generated references must pass geometry validation")
        self.prior = replace(initial_prior)
        self.matcher = matcher or get_matcher(
            self.config.matcher, mode=self.config.mode, max_dim=self.config.max_dim,
            min_inliers=self.config.min_inliers, min_ratio=self.config.min_ratio,
            min_coverage=self.config.min_coverage, ratio=self.config.ratio,
            ransac_px=self.config.ransac_px, rng_seed=self.config.rng_seed)
        self.selector = CandidateSelector(
            self.store, base_radius=self.config.base_radius, max_radius=self.config.max_radius,
            max_candidates=self.config.max_candidates, sigma_scale=self.config.sigma_scale,
            global_fallback=False)
        # Reuse the tested resize/georeferencing calculations. The helper is
        # never run and has no access to a trajectory or future ground truth.
        self._geometry = LabRunner(store, [], self.config, references=references)
        self._last_timestamp: float | None = None
        self._pending: tuple[float, float, float, int] | None = None

    def process_frame(self, frame: np.ndarray | str | Path, timestamp: float, *,
                      altitude_agl_m: float | None = None,
                      camera_gsd_m: float | None = None,
                      motion_delta_m: tuple[float, float] | None = None,
                      motion_sigma_m: float | None = None) -> NavigationResult:
        """Process a BGR frame; optional relative motion is east/north in metres.

        Motion must cover the interval since the previous call (or initial
        prior). It can come from VIO/INS; it is not an absolute GPS position.
        The uncertainty is an approximate incremental standard deviation.
        """
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self._last_timestamp is not None and timestamp <= self._last_timestamp):
            raise ValueError("frame timestamps must increase strictly")
        if self.prior.timestamp is not None and timestamp < self.prior.timestamp:
            raise ValueError("frame precedes the initial prior")
        image = frame if isinstance(frame, np.ndarray) else cv2.imread(str(frame), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("frame must be a nonempty BGR image")
        if camera_gsd_m is not None and (not math.isfinite(camera_gsd_m) or camera_gsd_m <= 0):
            raise ValueError("camera GSD must be positive and finite")
        if motion_delta_m is not None:
            dx, dy = motion_delta_m
            if not math.isfinite(dx) or not math.isfinite(dy):
                raise ValueError("motion delta must be finite")
            if motion_sigma_m is not None and (not math.isfinite(motion_sigma_m) or motion_sigma_m < 0):
                raise ValueError("motion uncertainty must be finite and nonnegative")
            dt = max(0.0, timestamp - float(self.prior.timestamp if self.prior.timestamp is not None else timestamp))
            self.prior.east += float(dx)
            self.prior.north += float(dy)
            self.prior.sigma = (math.hypot(self.prior.sigma, motion_sigma_m)
                                if motion_sigma_m is not None else self.prior.sigma + 3 * dt)
            self.prior.timestamp = timestamp
        self._last_timestamp = timestamp
        predicted = self.prior.predict(timestamp)
        assert predicted is not None
        if self.cropper is None:
            candidates, _ = self.selector.select(self.prior, timestamp, kind=self.config.reference_kind)
        else:
            elapsed = max(0.0, timestamp - self.prior.timestamp) if self.prior.timestamp is not None else 0.0
            crop_sigma = predicted[2] + self.cropper.sigma_growth_mps * elapsed
            try:
                crop = self.cropper.crop(predicted[0], predicted[1], crop_sigma, image.shape[:2],
                                         altitude_agl_m=altitude_agl_m)
            except Exception as error:
                self._pending = None
                self.prior.decay(timestamp)
                return NavigationResult(timestamp=timestamp, accepted=False, reason="crop_error",
                                        prior_east_m=predicted[0], prior_north_m=predicted[1],
                                        prior_sigma_m=predicted[2],
                                        execution_errors=(f"{type(error).__name__}: {error}",))
            candidates = [crop] if crop is not None else []
        ids = tuple(str(tile["id"]) for tile in candidates)
        valid: list[tuple[tuple, dict, MatchOutcome, float, float, int]] = []
        best_debug: tuple[tuple, dict, MatchOutcome, int] | None = None
        execution_errors: list[str] = []
        fake_query = QueryFrame(id="frame", path=Path("<memory>"), timestamp=timestamp,
                                lat=0, lon=0, altitude=float(altitude_agl_m or 0), yaw=0,
                                has_truth=False)
        for angle in self.rotations:
            rotated = image if angle == 0 else cv2.rotate(image, {
                90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
            shape = rotated.shape[:2]
            for tile in candidates:
                working = rotated
                resize_factor = 1.0
                if camera_gsd_m is not None:
                    tile_px = float(tile.get("tile_px") or self.config.max_dim)
                    ref_factor = min(1.0, self.config.max_dim / tile_px)
                    ref_gsd = float(tile.get("gsd") or self.store.gsd) / ref_factor
                    resize_factor = min(camera_gsd_m / ref_gsd,
                                        self.config.max_dim / max(rotated.shape[:2]))
                    new_w = max(1, int(round(rotated.shape[1] * resize_factor)))
                    new_h = max(1, int(round(rotated.shape[0] * resize_factor)))
                    if min(new_w, new_h) < 16:
                        continue
                    working = cv2.resize(rotated, (new_w, new_h),
                                         interpolation=cv2.INTER_AREA if resize_factor < 1 else cv2.INTER_CUBIC)
                shape = working.shape[:2]
                ref = tile["_image"] if "_image" in tile else self.store.path_for(tile)
                outcome = self.matcher.match(ref, working)
                if outcome.execution_error:
                    execution_errors.append(outcome.error or "matcher execution failed")
                    break
                score = self._geometry._score(outcome)
                if best_debug is None or score > best_debug[0]:
                    best_debug = (score, tile, outcome, angle)
                if not outcome.success or outcome.homography is None:
                    continue
                east, north = self._geometry._refine_position(tile, outcome, shape)
                bounds = tile["bounds"]
                if (not math.isfinite(east) or not math.isfinite(north)
                        or not bounds["min_east"] <= east <= bounds["max_east"]
                        or not bounds["min_north"] <= north <= bounds["max_north"]):
                    continue
                # For partially available map crops, never accept a projected
                # centre on a missing pixel. This does not validate all matches.
                if "_valid_mask" in tile:
                    col = int((east-bounds["min_east"]) / tile["gsd"])
                    row = int((bounds["max_north"]-north) / tile["gsd"])
                    mask = tile["_valid_mask"]
                    if not (0 <= row < mask.shape[0] and 0 <= col < mask.shape[1] and mask[row, col] == 255):
                        continue
                # FOV is that of the original camera image; quarter turns
                # swap dimensions but do not change its ground pixel size.
                expected = self._geometry._expected_scale(fake_query, tile, image.shape[:2])
                if camera_gsd_m is not None:
                    expected = ref_gsd * resize_factor / camera_gsd_m
                if self.config.scale_tolerance > 0 and expected and outcome.scale:
                    ratio = outcome.scale / expected
                    limit = 1 + self.config.scale_tolerance
                    if not 1 / limit <= ratio <= limit:
                        continue
                if math.hypot(east - predicted[0], north - predicted[1]) > self.config.max_jump_m + self.config.jump_sigma_scale * predicted[2]:
                    continue
                valid.append((score, tile, outcome, east, north, angle))
            # Rotations are a fallback, not independent measurements. This
            # preserves latency but can miss a better-scoring later rotation.
            if valid or execution_errors:
                break

        base = dict(timestamp=timestamp, prior_east_m=predicted[0], prior_north_m=predicted[1],
                    prior_sigma_m=predicted[2], candidate_ids=ids,
                    candidate_bounds=tuple(tile["bounds"] for tile in candidates),
                    candidate_valid_fractions=tuple(tile.get("valid_fraction", 1.0) for tile in candidates),
                    execution_errors=tuple(execution_errors))
        if execution_errors:
            # Candidate evaluation was incomplete; even an earlier match cannot
            # be ranked reliably against the candidates that were not checked.
            self._pending = None
            self.prior.decay(timestamp)
            return NavigationResult(**base, accepted=False, reason="matcher_error",
                                    best_tile=str(best_debug[1]["id"]) if best_debug else None,
                                    query_rotation_deg=best_debug[3] if best_debug else None,
                                    match=best_debug[2].as_dict() if best_debug else None)
        if not valid:
            self._pending = None
            self.prior.decay(timestamp)
            return NavigationResult(**base, accepted=False, reason="no_valid_match" if candidates else "no_candidates",
                                    best_tile=str(best_debug[1]["id"]) if best_debug else None,
                                    query_rotation_deg=best_debug[3] if best_debug else None,
                                    match=best_debug[2].as_dict() if best_debug else None)
        _, tile, outcome, east, north, angle = max(valid, key=lambda item: item[0])
        if self.config.confirmation > 1:
            count = 1
            if self._pending is not None:
                prev_east, prev_north, prev_time, prev_count = self._pending
                dt = timestamp - prev_time
                dx, dy = motion_delta_m if motion_delta_m is not None else (self.prior.vx * dt, self.prior.vy * dt)
                delta = math.hypot(east - (prev_east + dx), north - (prev_north + dy))
                if delta <= self.config.confirmation_radius_m:
                    count = prev_count + 1
            self._pending = (east, north, timestamp, count)
            if count < self.config.confirmation:
                self.prior.decay(timestamp)
                return NavigationResult(**base, accepted=False, reason="pending_confirmation",
                                        best_tile=str(tile["id"]), query_rotation_deg=angle, match=outcome.as_dict())
        self._pending = None
        self.prior.update(east, north, timestamp, sigma=self.config.fix_sigma,
                          max_speed=self.config.max_speed_mps)
        lon, lat = self.store.to_wgs(east, north)
        image_to_local = (_image_to_local(tile, outcome.homography, image.shape[:2], angle,
                                          self.config.max_dim, self.store.gsd)
                          if camera_gsd_m is None else None)
        return NavigationResult(**base, accepted=True, reason="ok", latitude=lat, longitude=lon,
                                east_m=east, north_m=north, best_tile=str(tile["id"]),
                                query_rotation_deg=angle,
                                match=outcome.as_dict(), image_to_local=image_to_local)
