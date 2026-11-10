"""2.5D orthophoto flythrough — synthetic drone frames with exact ground truth.

The point is to answer "where do the test images come from?" without building a
3D city. A georeferenced orthophoto *is* the map the localizer will match
against, so rendering a pinhole camera over that same map along a known
trajectory gives us:

* frames whose geometry is guaranteed consistent with the reference map,
* exact per-frame ground truth (lat/lon, altitude, yaw, pitch, roll, GSD),
* a controllable difficulty ladder (altitude, tilt, blur, JPEG, noise).

What it does **not** give is parallax and occlusion: the ground is a plane.
For those, use ``scripts/make_geo_world.py`` (Gazebo, L2). For near-nadir
missions this is enough, and it is the same trick UAV-VisLoc uses for the nine
synthetic trajectories it builds from OpenAerialMap.

Geometry
--------
Ground point ``(X, Y, 0)`` in local ENU, camera at ``C`` with orthonormal axes
``(r, d, f)`` (right, down-in-image, view direction)::

    v  = (X, Y, 0) - C
    u  = W/2 + focal * (v . r) / (v . f)
    v' = H/2 + focal * (v . d) / (v . f)

Because the ground is the plane ``z = 0`` this is a homography in ``(X, Y)``,
so one ``cv2.warpPerspective`` per frame renders it exactly. The CRS mapping
(ENU -> raster CRS) is absorbed by a locally fitted affine, which keeps
geographic (degree) rasters and rotated rasters working.

Usage
-----
    # real map
    python -m sat2aerial.orthofly --ortho site.tif --out outputs/orthofly/site \\
        --alt 60 --frames 60 --path line --length 300

    # no data at all (procedural city, 1 km across, 0.5 m/px)
    python -m sat2aerial.orthofly --procedural --out outputs/orthofly/demo
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from navlab.geo import local_crs

try:  # rasterio is optional for the procedural path
    import rasterio
    from rasterio.transform import Affine
except Exception:  # pragma: no cover - only hit without rasterio
    rasterio = None
    Affine = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# camera and pose
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 640
    hfov_deg: float = 60.0

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def ground_footprint_m(self, altitude: float) -> float:
        return 2.0 * altitude * math.tan(math.radians(self.hfov_deg) / 2.0)

    def gsd_m(self, altitude: float) -> float:
        return self.ground_footprint_m(altitude) / self.width


@dataclass
class Pose:
    """Camera pose in a local ENU frame (metres), yaw clockwise from north."""

    x: float = 0.0
    y: float = 0.0
    alt: float = 60.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0


def camera_axes(yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal camera axes ``(right, down, forward)`` in ENU.

    ``pitch`` tilts the optical axis forward from nadir; ``roll`` rotates about
    the optical axis. At ``yaw=pitch=roll=0`` the camera looks straight down
    with image +u to the east and image +v to the south.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    horizontal_forward = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    forward = math.cos(pitch) * np.array([0.0, 0.0, -1.0]) + math.sin(pitch) * horizontal_forward
    down = np.cross(forward, right)

    if roll:
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        right, down = cos_r * right + sin_r * down, -sin_r * right + cos_r * down
    return right, down, forward


def ground_homography(pose: Pose, camera: Camera) -> np.ndarray:
    """3x3 homography mapping local ENU ground ``(X, Y)`` to image pixels."""
    right, down, forward = camera_axes(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    focal = camera.focal_px
    cx, cy, cz = float(pose.x), float(pose.y), float(pose.alt)
    centre = np.array([cx, cy, cz])

    # v = P - C with P = (X, Y, 0); split into the X, Y and constant parts.
    def offset(axis: np.ndarray) -> float:
        return float(-axis @ centre)

    a = offset(right)
    b = offset(down)
    c = offset(forward)

    half_u = camera.width / 2.0
    half_v = camera.height / 2.0
    return np.array(
        [
            [
                focal * right[0] + half_u * forward[0],
                focal * right[1] + half_u * forward[1],
                focal * a + half_u * c,
            ],
            [
                focal * down[0] + half_v * forward[0],
                focal * down[1] + half_v * forward[1],
                focal * b + half_v * c,
            ],
            [forward[0], forward[1], c],
        ],
        dtype=np.float64,
    )


def project(pose: Pose, camera: Camera, x: float, y: float, z: float = 0.0) -> tuple[float, float] | None:
    """Project one ground point (or any ENU point) into image pixels."""
    right, down, forward = camera_axes(pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    v = np.array([x - pose.x, y - pose.y, z - pose.alt])
    depth = float(v @ forward)
    if depth <= 1e-6:
        return None
    focal = camera.focal_px
    return (
        camera.width / 2.0 + focal * float(v @ right) / depth,
        camera.height / 2.0 + focal * float(v @ down) / depth,
    )


# --------------------------------------------------------------------------- #
# ground map
# --------------------------------------------------------------------------- #


def _as_matrix(affine: Any) -> np.ndarray:
    return np.array(
        [[affine.a, affine.b, affine.c], [affine.d, affine.e, affine.f], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _local_affine(transform: Callable[[float, float], tuple[float, float]], x0: float, y0: float, scale: float = 50.0) -> np.ndarray:
    """Locally fitted affine of a (possibly non-linear) CRS transform.

    The linear part is measured around ``(x0, y0)`` but the matrix is applied to
    *absolute* ENU coordinates, so the constant term must be re-based to
    ``ENU (0, 0)``. Without that, the camera position cancels out and every
    frame is rendered at ``2 * pose`` — a bug that is invisible for
    ``from_array`` maps and for poses at the origin.
    """
    ax, ay = transform(x0, y0)
    bx, by = transform(x0 + scale, y0)
    cx, cy = transform(x0, y0 + scale)
    a = (bx - ax) / scale
    b = (cx - ax) / scale
    c = (by - ay) / scale
    d = (cy - ay) / scale
    return np.array(
        [
            [a, b, ax - a * x0 - b * y0],
            [c, d, ay - c * x0 - d * y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class GroundMap:
    """An in-memory orthophoto plus the transforms needed to sample it."""

    def __init__(
        self,
        image: np.ndarray,
        affine: Any,
        origin_lat: float,
        origin_lon: float,
        to_crs: Any = None,
        raster_crs: Any = None,
    ) -> None:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.image = np.ascontiguousarray(image)
        self.affine = affine
        self.origin_lat = float(origin_lat)
        self.origin_lon = float(origin_lon)
        self._to_crs = to_crs  # ENU -> raster CRS, or None when identical
        self._enu_crs = local_crs(origin_lat, origin_lon)
        from pyproj import Transformer

        self._to_latlon = Transformer.from_crs(self._enu_crs, "EPSG:4326", always_xy=True)
        self._from_crs = (
            None
            if to_crs is None
            else Transformer.from_crs(to_crs.target_crs, to_crs.source_crs, always_xy=True)
        )
        self._pixel_to_crs = _as_matrix(affine)
        self._crs_to_pixel = np.linalg.inv(self._pixel_to_crs)
        self._fill = tuple(int(v) for v in np.median(self.image.reshape(-1, 3)[::13], axis=0))

        # Pixel size in metres. For a geographic CRS ``affine`` is in degrees, so
        # reporting it raw would silently turn every metric error into ~0.
        a, b, _, d, e, _ = affine[:6]
        size_x = math.hypot(a, d)
        size_y = math.hypot(b, e)
        if raster_crs is not None and raster_crs.is_geographic:
            size_x *= 111_320.0 * math.cos(math.radians(origin_lat))
            size_y *= 111_320.0
        self._gsd_m = float((size_x + size_y) / 2.0)
        self.gsd_xy_m = (float(size_x), float(size_y))

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_geotiff(
        cls,
        path: str | Path,
        origin: tuple[float, float] | None = None,
        max_pixels: int = 40_000_000,
    ) -> "GroundMap":
        if rasterio is None:
            raise RuntimeError("rasterio is required to read a GeoTIFF")
        path = Path(path)
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"{path} has no CRS; georeference it first")
            factor = 1
            if dataset.width * dataset.height > max_pixels:
                factor = int(math.ceil(math.sqrt(dataset.width * dataset.height / max_pixels)))
            height = max(1, dataset.height // factor)
            width = max(1, dataset.width // factor)
            bands = min(3, dataset.count)
            data = dataset.read(indexes=list(range(1, bands + 1)), out_shape=(bands, height, width))
            image = np.transpose(data, (1, 2, 0))
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 2:
                image = np.repeat(image[:, :, :1], 3, axis=2)
            else:
                image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
            image = _stretch(image)
            affine = dataset.transform @ Affine.scale(factor)
            crs = dataset.crs

            if origin is None:
                mx, my = affine @ (width / 2.0, height / 2.0)
                if crs.to_epsg() == 4326:
                    lon, lat = mx, my
                else:
                    from pyproj import Transformer

                    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(mx, my)
            else:
                lat, lon = origin

        from pyproj import Transformer

        enu = local_crs(lat, lon)
        to_crs = Transformer.from_crs(enu, crs, always_xy=True)
        return cls(image, affine, lat, lon, to_crs=to_crs, raster_crs=crs)

    @classmethod
    def from_array(
        cls,
        image: np.ndarray,
        gsd_m: float = 0.5,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        x0: float = -100.0,
        y0: float = 100.0,
    ) -> "GroundMap":
        """Wrap a metric array: column 0 is ``x0``, row 0 is ``y0`` (north)."""
        affine = Affine(gsd_m, 0.0, x0, 0.0, -gsd_m, y0)
        return cls(image, affine, origin_lat, origin_lon, to_crs=None)

    # -- transforms -------------------------------------------------------- #

    def _enu_to_crs_xy(self, x: float, y: float) -> tuple[float, float]:
        if self._to_crs is None:
            return float(x), float(y)
        return self._to_crs.transform(x, y)

    def _crs_to_enu_matrix(self, x0: float, y0: float) -> np.ndarray:
        if self._to_crs is None:
            return np.eye(3)
        return np.linalg.inv(_local_affine(self._to_crs.transform, x0, y0))

    def local_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """ENU metres -> ``(col, row)`` floats in the map array."""
        cx, cy = self._enu_to_crs_xy(x, y)
        col, row, _ = self._crs_to_pixel @ np.array([cx, cy, 1.0])
        return float(col), float(row)

    def pixel_to_local(self, col: float, row: float) -> tuple[float, float]:
        cx, cy, _ = self._pixel_to_crs @ np.array([col, row, 1.0])
        if self._to_crs is None:
            return float(cx), float(cy)
        return self._from_crs.transform(cx, cy)

    def latlon(self, x: float, y: float) -> tuple[float, float]:
        lon, lat = self._to_latlon.transform(x, y)
        return float(lat), float(lon)

    @property
    def gsd_m(self) -> float:
        """Mean pixel size in metres (never in CRS units)."""
        return self._gsd_m

    # -- sampling ---------------------------------------------------------- #

    def image_matrix(self, pose: Pose, camera: Camera) -> np.ndarray:
        """3x3 homography mapping map pixels ``(col, row)`` to image pixels."""
        homography = ground_homography(pose, camera)
        crs_to_enu = self._crs_to_enu_matrix(pose.x, pose.y)
        return homography @ crs_to_enu @ self._pixel_to_crs

    def render(self, pose: Pose, camera: Camera, fill: tuple[int, int, int] | None = None) -> np.ndarray:
        matrix = self.image_matrix(pose, camera)
        if fill is None:
            fill = self._fill
        return cv2.warpPerspective(
            self.image,
            matrix,
            (camera.width, camera.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )

    def crop(self, x: float, y: float, size: int, fill: tuple[int, int, int] | None = None) -> np.ndarray:
        """Square reference crop of ``size`` pixels centred on ENU ``(x, y)``."""
        size = int(size)
        col, row = self.local_to_pixel(x, y)
        half = size // 2
        left = int(round(col)) - half
        top = int(round(row)) - half
        return _window(self.image, top, left, size, size, fill or self._fill)


def _stretch(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    values = image.astype(np.float32)
    low, high = np.percentile(values, 2), np.percentile(values, 98)
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _window(image: np.ndarray, top: int, left: int, height: int, width: int, fill: tuple[int, int, int] | None) -> np.ndarray:
    """Slice with replication outside the array, matching Gazebo's behaviour."""
    if fill is None:
        fill = tuple(int(v) for v in np.median(image.reshape(-1, 3), axis=0))
    out = np.empty((height, width, 3), np.uint8)
    out[:] = fill
    src_top = max(0, top)
    src_left = max(0, left)
    src_bottom = min(image.shape[0], top + height)
    src_right = min(image.shape[1], left + width)
    if src_bottom <= src_top or src_right <= src_left:
        return out
    out[src_top - top : src_bottom - top, src_left - left : src_right - left] = image[src_top:src_bottom, src_left:src_right]
    return out


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #


def line_path(length_m: float, heading_deg: float, start: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    heading = math.radians(heading_deg)
    step = np.array([math.sin(heading), math.cos(heading)]) * length_m
    return [start, (start[0] + step[0], start[1] + step[1])]


def snake_path(
    length_m: float,
    heading_deg: float,
    legs: int = 3,
    leg_spacing_m: float = 60.0,
    start: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    heading = math.radians(heading_deg)
    along = np.array([math.sin(heading), math.cos(heading)])
    across = np.array([math.cos(heading), -math.sin(heading)])
    points = [start]
    current = np.array(start, dtype=float)
    for leg in range(max(1, legs)):
        direction = along if leg % 2 == 0 else -along
        current = current + direction * length_m
        points.append((current[0], current[1]))
        if leg != legs - 1:
            current = current + across * leg_spacing_m
            points.append((current[0], current[1]))
    return points


def orbit_path(radius_m: float, start_angle_deg: float = 0.0, turns: float = 1.0) -> list[tuple[float, float]]:
    count = max(2, int(round(64 * turns)))
    points = []
    for index in range(count + 1):
        angle = math.radians(start_angle_deg) + 2 * math.pi * turns * index / count
        points.append((radius_m * math.cos(angle), radius_m * math.sin(angle)))
    return points


def _resample(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    if len(points) < 2 or count < 2:
        return [points[0]] * max(count, 1)
    lengths = [0.0]
    for index in range(1, len(points)):
        lengths.append(lengths[-1] + math.dist(points[index - 1], points[index]))
    total = lengths[-1]
    if total <= 1e-9:
        return [points[0]] * count
    result = []
    for index in range(count):
        target = total * index / (count - 1)
        for segment in range(1, len(lengths)):
            if lengths[segment] >= target:
                fraction = (target - lengths[segment - 1]) / max(lengths[segment] - lengths[segment - 1], 1e-12)
                a, b = points[segment - 1], points[segment]
                result.append((a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1])))
                break
        else:
            result.append(points[-1])
    return result


def _tangent_yaw(points: list[tuple[float, float]], index: int) -> float:
    if len(points) < 2:
        return 0.0
    a = points[max(0, index - 1)]
    b = points[min(len(points) - 1, index + 1)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, dy)) % 360.0


# --------------------------------------------------------------------------- #
# sensor effects
# --------------------------------------------------------------------------- #


@dataclass
class Effects:
    """Everything that makes a rendered frame *not* a perfect map lookup."""

    blur_sigma: float = 0.6
    noise_sigma: float = 2.0
    gamma_jitter: float = 0.08
    exposure_jitter: float = 0.06
    jpeg_quality: int = 0  # 0 disables the round-trip
    vignette: float = 0.15
    exposure_ms: float = 2.0

    def apply(self, frame: np.ndarray, rng: np.random.Generator, speed_mps: float, altitude: float, focal_px: float) -> np.ndarray:
        out = frame.astype(np.float32)

        # Camera yaw tracks the flight direction, so image +x is the motion axis.
        if self.exposure_ms > 0 and speed_mps > 0 and altitude > 0:
            travel_px = speed_mps * (self.exposure_ms / 1000.0) * focal_px / altitude
            if travel_px >= 1.5:
                kernel = np.zeros((1, int(round(travel_px))), np.float32)
                kernel[:] = 1.0 / kernel.size
                out = cv2.filter2D(out, -1, kernel)

        if self.vignette > 0:
            height, width = out.shape[:2]
            yy, xx = np.mgrid[0:height, 0:width]
            radius = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
            mask = 1.0 - self.vignette * np.clip(radius / 1.4142, 0.0, 1.0) ** 2
            out *= mask[..., None]

        if self.exposure_jitter > 0:
            out *= float(1.0 + rng.uniform(-self.exposure_jitter, self.exposure_jitter))
        if self.gamma_jitter > 0:
            gamma = float(1.0 + rng.uniform(-self.gamma_jitter, self.gamma_jitter))
            out = np.clip(((np.clip(out, 0, 255) / 255.0) ** gamma) * 255.0, 0, 255)
        if self.blur_sigma > 0:
            out = cv2.GaussianBlur(out, (0, 0), self.blur_sigma)
        if self.noise_sigma > 0:
            out += rng.normal(0.0, self.noise_sigma, out.shape).astype(np.float32)

        out = np.clip(out, 0, 255).astype(np.uint8)
        if self.jpeg_quality and self.jpeg_quality < 100:
            ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
            if ok:
                out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return out


# --------------------------------------------------------------------------- #
# flythrough
# --------------------------------------------------------------------------- #


@dataclass
class FlythroughConfig:
    out: Path
    path_kind: str = "line"
    frames: int = 60
    camera: Camera = field(default_factory=Camera)
    altitude_m: float = 60.0
    altitude_ramp_m: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    yaw_mode: str = "track"  # track | fixed
    yaw_jitter_deg: float = 0.0
    heading_deg: float = 90.0
    length_m: float = 300.0
    legs: int = 3
    leg_spacing_m: float = 60.0
    radius_m: float = 80.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    ref_size_px: int = 512
    ref_jitter_m: float = 0.0
    speed_mps: float = 5.0
    seed: int = 0
    effects: Effects = field(default_factory=Effects)
    pairs_dir: Path | None = None


def build_poses(cfg: FlythroughConfig) -> list[Pose]:
    """Sample the requested path into ``cfg.frames`` poses with yaw per frame."""
    if cfg.path_kind == "line":
        points = line_path(cfg.length_m, cfg.heading_deg, cfg.start_xy)
    elif cfg.path_kind == "snake":
        points = snake_path(cfg.length_m, cfg.heading_deg, cfg.legs, cfg.leg_spacing_m, cfg.start_xy)
    elif cfg.path_kind == "orbit":
        points = orbit_path(cfg.radius_m)
        offset = np.array(cfg.start_xy, dtype=float)
        points = [(x + offset[0], y + offset[1]) for x, y in points]
    else:
        raise ValueError(f"unknown path kind {cfg.path_kind!r}")

    sampled = _resample(points, cfg.frames)
    rng = np.random.default_rng(cfg.seed)
    poses: list[Pose] = []
    for index, (x, y) in enumerate(sampled):
        fraction = index / max(cfg.frames - 1, 1)
        if cfg.yaw_mode == "track":
            yaw = _tangent_yaw(sampled, index)
        else:
            yaw = cfg.yaw_deg
        if cfg.yaw_jitter_deg:
            yaw += float(rng.normal(0.0, cfg.yaw_jitter_deg))
        poses.append(
            Pose(
                x=float(x),
                y=float(y),
                alt=cfg.altitude_m + cfg.altitude_ramp_m * fraction,
                yaw_deg=yaw % 360.0,
                pitch_deg=cfg.pitch_deg,
                roll_deg=cfg.roll_deg,
            )
        )
    return poses


def _contact_sheet(frames: list[np.ndarray], columns: int = 4, tile: int = 176) -> np.ndarray:
    count = len(frames)
    columns = min(columns, max(1, count))
    rows = int(math.ceil(count / columns))
    sheet = np.full((rows * tile, columns * tile, 3), 24, np.uint8)
    for index, frame in enumerate(frames):
        row, col = divmod(index, columns)
        sheet[row * tile : (row + 1) * tile, col * tile : (col + 1) * tile] = cv2.resize(frame, (tile, tile), interpolation=cv2.INTER_AREA)
    return sheet


def run_flythrough(map_: GroundMap, cfg: FlythroughConfig) -> dict[str, Any]:
    out = Path(cfg.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if cfg.pairs_dir is not None:
        (Path(cfg.pairs_dir) / "satellite").mkdir(parents=True, exist_ok=True)
        (Path(cfg.pairs_dir) / "aerial").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed + 1)
    poses = build_poses(cfg)
    rows: list[dict[str, Any]] = []
    preview: list[np.ndarray] = []
    path_length = sum(math.dist((poses[i].x, poses[i].y), (poses[i + 1].x, poses[i + 1].y)) for i in range(len(poses) - 1))

    for index, pose in enumerate(poses):
        frame = map_.render(pose, cfg.camera)
        frame = cfg.effects.apply(frame, rng, cfg.speed_mps, pose.alt, cfg.camera.focal_px)
        stem = f"frame_{index:04d}"
        cv2.imwrite(str(frames_dir / f"{stem}.png"), frame)
        if len(preview) < 16:
            preview.append(frame)

        jitter_x = jitter_y = 0.0
        if cfg.ref_jitter_m > 0:
            jitter_x, jitter_y = (float(v) for v in rng.uniform(-cfg.ref_jitter_m, cfg.ref_jitter_m, 2))

        lat, lon = map_.latlon(pose.x, pose.y)
        col, row = map_.local_to_pixel(pose.x, pose.y)
        rows.append(
            {
                "frame": stem,
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "x_m": round(pose.x, 4),
                "y_m": round(pose.y, 4),
                "alt_m": round(pose.alt, 4),
                "yaw_deg": round(pose.yaw_deg, 4),
                "pitch_deg": round(pose.pitch_deg, 4),
                "roll_deg": round(pose.roll_deg, 4),
                "gsd_m": round(cfg.camera.gsd_m(pose.alt), 5),
                "map_col": round(col, 3),
                "map_row": round(row, 3),
                "ref_center_x_m": round(pose.x + jitter_x, 4),
                "ref_center_y_m": round(pose.y + jitter_y, 4),
            }
        )

        if cfg.pairs_dir is not None:
            reference = map_.crop(pose.x + jitter_x, pose.y + jitter_y, cfg.ref_size_px)
            cv2.imwrite(str(Path(cfg.pairs_dir) / "satellite" / f"{stem}.png"), reference)
            cv2.imwrite(str(Path(cfg.pairs_dir) / "aerial" / f"{stem}_00.png"), frame)

    with (out / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cv2.imwrite(str(out / "preview.png"), _contact_sheet(preview))

    meta = {
        "config": {
            "path_kind": cfg.path_kind,
            "frames": cfg.frames,
            "camera": asdict(cfg.camera),
            "altitude_m": cfg.altitude_m,
            "altitude_ramp_m": cfg.altitude_ramp_m,
            "pitch_deg": cfg.pitch_deg,
            "roll_deg": cfg.roll_deg,
            "yaw_mode": cfg.yaw_mode,
            "heading_deg": cfg.heading_deg,
            "length_m": cfg.length_m,
            "speed_mps": cfg.speed_mps,
            "seed": cfg.seed,
            "ref_size_px": cfg.ref_size_px,
            "ref_jitter_m": cfg.ref_jitter_m,
            "effects": asdict(cfg.effects),
        },
        "map": {
            "origin_lat": map_.origin_lat,
            "origin_lon": map_.origin_lon,
            "gsd_m": map_.gsd_m,
            "gsd_x_m": map_.gsd_xy_m[0],
            "gsd_y_m": map_.gsd_xy_m[1],
            "width_px": int(map_.image.shape[1]),
            "height_px": int(map_.image.shape[0]),
        },
        "mean_gsd_m": round(cfg.camera.gsd_m(cfg.altitude_m + cfg.altitude_ramp_m / 2.0), 5),
        "distance_m": round(path_length, 3),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# procedural map (offline smoke test)
# --------------------------------------------------------------------------- #


def procedural_map(size: int = 2048, gsd_m: float = 0.5, seed: int = 0) -> GroundMap:
    from .synthetic import make_satellite

    rng = np.random.default_rng(seed)
    image = make_satellite(rng, size=size)
    extent = size * gsd_m
    return GroundMap.from_array(image, gsd_m=gsd_m, x0=-extent / 2.0, y0=extent / 2.0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_arguments(parser: Any) -> None:
    parser.add_argument("--ortho", default=None, help="georeferenced GeoTIFF to fly over")
    parser.add_argument("--procedural", action="store_true", help="generate a procedural city instead (no data)")
    parser.add_argument("--procedural-size", type=int, default=2048)
    parser.add_argument("--procedural-gsd", type=float, default=0.5)
    parser.add_argument("--center-lat", type=float, default=None)
    parser.add_argument("--center-lon", type=float, default=None)
    parser.add_argument("--max-pixels", type=int, default=40_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--size", type=int, default=640, help="frame width/height in pixels")
    parser.add_argument("--hfov", type=float, default=60.0)
    parser.add_argument("--alt", type=float, default=60.0)
    parser.add_argument("--alt-ramp", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--path", default="line", choices=["line", "snake", "orbit"])
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--length", type=float, default=300.0)
    parser.add_argument("--legs", type=int, default=3)
    parser.add_argument("--leg-spacing", type=float, default=60.0)
    parser.add_argument("--radius", type=float, default=80.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--yaw-mode", default="track", choices=["track", "fixed"])
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--yaw-jitter", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--ref-size", type=int, default=512)
    parser.add_argument("--ref-jitter", type=float, default=0.0, help="random offset of each reference crop (metres)")
    parser.add_argument("--pairs-dir", default=None, help="also write sat2aerial satellite/ + aerial/ pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blur", type=float, default=0.6)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--gamma-jitter", type=float, default=0.08)
    parser.add_argument("--exposure-jitter", type=float, default=0.06)
    parser.add_argument("--exposure-ms", type=float, default=2.0)
    parser.add_argument("--jpeg", type=int, default=0)
    parser.add_argument("--vignette", type=float, default=0.15)
    parser.add_argument("--clean", action="store_true")


def run_from_args(args: Any) -> int:
    """Body of the CLI, shared by ``python -m sat2aerial.orthofly`` and ``sat2aerial flythrough``."""
    if args.ortho:
        origin = None
        if args.center_lat is not None and args.center_lon is not None:
            origin = (args.center_lat, args.center_lon)
        ground = GroundMap.from_geotiff(args.ortho, origin=origin, max_pixels=args.max_pixels)
    elif args.procedural:
        ground = procedural_map(args.procedural_size, args.procedural_gsd, args.seed)
    else:
        raise SystemExit("orthofly: pass --ortho PATH or --procedural")

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)

    cfg = FlythroughConfig(
        out=out,
        path_kind=args.path,
        frames=args.frames,
        camera=Camera(width=args.size, height=args.size, hfov_deg=args.hfov),
        altitude_m=args.alt,
        altitude_ramp_m=args.alt_ramp,
        pitch_deg=args.pitch,
        roll_deg=args.roll,
        yaw_deg=args.yaw,
        yaw_mode=args.yaw_mode,
        yaw_jitter_deg=args.yaw_jitter,
        heading_deg=args.heading,
        length_m=args.length,
        legs=args.legs,
        leg_spacing_m=args.leg_spacing,
        radius_m=args.radius,
        start_xy=(args.start_x, args.start_y),
        ref_size_px=args.ref_size,
        ref_jitter_m=args.ref_jitter,
        speed_mps=args.speed,
        seed=args.seed,
        effects=Effects(
            blur_sigma=args.blur,
            noise_sigma=args.noise,
            gamma_jitter=args.gamma_jitter,
            exposure_jitter=args.exposure_jitter,
            jpeg_quality=args.jpeg,
            vignette=args.vignette,
            exposure_ms=args.exposure_ms,
        ),
        pairs_dir=Path(args.pairs_dir) if args.pairs_dir else None,
    )
    meta = run_flythrough(ground, cfg)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="sat2aerial.orthofly", description=__doc__.splitlines()[0])
    _add_arguments(parser)
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
