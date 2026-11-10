from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2

from .types import QueryFrame


def _timestamp(value: str | None, index: int) -> float:
    if value:
        text = value.strip()
        try:
            # UAV-VisLoc 07 uses compact YYYYMMDDHMMSS with a one-digit
            # hour. fromisoformat misreads the hour as a separator and treats
            # minutes/seconds as hours/minutes (60x time error).
            if text.isdigit() and len(text) in (13, 14):
                return datetime.strptime(text, "%Y%m%d%H%M%S").timestamp()
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            pass
    return float(index)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def uav_visloc_queries(
    root: str | Path,
    map_id: str,
    *,
    near_nadir_deg: float = 15.0,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[QueryFrame]:
    root = Path(root)
    map_dir = root / str(map_id)
    csv_path = map_dir / f"{map_id}.csv"
    drone_dir = map_dir / "drone"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    rows = _read_csv(csv_path)
    kept: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        try:
            pitch = abs(float(row.get("Omega") or 0.0))
            roll = abs(float(row.get("Kappa") or 0.0))
            number = int(float(row.get("num") or 0))
        except ValueError:
            continue
        if pitch > near_nadir_deg or roll > near_nadir_deg:
            continue
        kept.append((number, row))
    kept.sort(key=lambda item: item[0])
    if stride > 1:
        kept = kept[::stride]
    if max_frames is not None:
        kept = kept[: max(1, int(max_frames))]

    queries: list[QueryFrame] = []
    for index, (number, row) in enumerate(kept):
        filename = (row.get("filename") or "").strip()
        image_path = drone_dir / filename
        if not image_path.is_file():
            continue
        queries.append(
            QueryFrame(
                id=Path(filename).stem,
                path=image_path,
                timestamp=_timestamp(row.get("date"), index),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                altitude=float(row.get("height") or 0.0),
                yaw=float(row.get("Phi1") or 0.0),
                meta={
                    "num": number,
                    "date": row.get("date", ""),
                    # Older CSVs may not contain these fields. Zero is used
                    # above for backwards compatibility, but must not be
                    # mistaken for a measured level attitude.
                    "attitude_present": all(
                        str(row.get(field) or "").strip() for field in ("Omega", "Kappa")
                    ),
                    "pitch_deg": float(row.get("Omega") or 0.0),
                    "roll_deg": float(row.get("Kappa") or 0.0),
                    "yaw_alt_deg": float(row.get("Phi2") or 0.0),
                },
            )
        )
    return queries


def uav_visloc_satellite(root: str | Path, map_id: str) -> Path:
    return Path(root) / str(map_id) / f"satellite{map_id}.tif"


def route_lonlat(queries: Iterable[QueryFrame]) -> list[tuple[float, float]]:
    return [(query.lon, query.lat) for query in queries]


def write_queries(
    path: str | Path,
    queries: Iterable[QueryFrame],
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [query.as_dict() for query in queries]
    payload = {"meta": meta or {}, "queries": frames}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_queries(path: str | Path) -> tuple[list[QueryFrame], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = [QueryFrame.from_dict(item) for item in payload.get("queries", [])]
    return frames, dict(payload.get("meta") or {})


def frame_folder_queries(
    frames_dir: str | Path,
    csv_path: str | Path,
    *,
    image_column: str = "filename",
    lat_column: str = "lat",
    lon_column: str = "lon",
    altitude_column: str = "height",
    yaw_column: str = "yaw",
    time_column: str = "timestamp",
) -> list[QueryFrame]:
    frames_dir = Path(frames_dir)
    rows = _read_csv(Path(csv_path))
    queries: list[QueryFrame] = []
    for index, row in enumerate(rows):
        filename = (row.get(image_column) or "").strip()
        image_path = frames_dir / filename
        if not image_path.is_file():
            continue
        queries.append(
            QueryFrame(
                id=Path(filename).stem,
                path=image_path,
                timestamp=float(row.get(time_column) or index),
                lat=float(row[lat_column]),
                lon=float(row[lon_column]),
                altitude=float(row.get(altitude_column) or 0.0),
                yaw=float(row.get(yaw_column) or 0.0),
                meta=dict(row),
            )
        )
    return queries


def _telemetry_rows(
    path: str | Path,
    *,
    time_column: str,
    lat_column: str,
    lon_column: str,
    altitude_column: str,
    yaw_column: str,
) -> list[dict[str, float]]:
    rows = _read_csv(Path(path))
    parsed: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        try:
            parsed.append(
                {
                    "t": float(row.get(time_column) or index),
                    "lat": float(row[lat_column]),
                    "lon": float(row[lon_column]),
                    "alt": float(row.get(altitude_column) or 0.0),
                    "yaw": float(row.get(yaw_column) or 0.0),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    parsed.sort(key=lambda item: item["t"])
    return parsed


def _nearest_telemetry(
    rows: list[dict[str, float]], timestamp: float
) -> dict[str, float] | None:
    if not rows:
        return None
    best = min(rows, key=lambda row: abs(row["t"] - timestamp))
    return best


def video_queries(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    telemetry_csv: str | Path | None = None,
    sample_fps: float = 2.0,
    image_ext: str = "jpg",
    time_column: str = "timestamp",
    lat_column: str = "lat",
    lon_column: str = "lon",
    altitude_column: str = "alt",
    yaw_column: str = "yaw",
    time_offset: float = 0.0,
) -> list[QueryFrame]:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    telemetry = (
        _telemetry_rows(
            telemetry_csv,
            time_column=time_column,
            lat_column=lat_column,
            lon_column=lon_column,
            altitude_column=altitude_column,
            yaw_column=yaw_column,
        )
        if telemetry_csv
        else []
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(video_path)
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    if video_fps <= 0:
        video_fps = 30.0
    step = max(1, int(round(video_fps / max(sample_fps, 1e-6))))
    queries: list[QueryFrame] = []
    index = 0
    saved = 0
    started = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            timestamp = index / video_fps
            filename = f"{saved:06d}.{image_ext}"
            image_path = frames_dir / filename
            cv2.imwrite(str(image_path), frame)
            telemetry_row = _nearest_telemetry(telemetry, timestamp + time_offset)
            has_truth = telemetry_row is not None
            queries.append(
                QueryFrame(
                    id=f"video_{saved:06d}",
                    path=image_path,
                    timestamp=started + timestamp,
                    lat=float(telemetry_row["lat"]) if telemetry_row else 0.0,
                    lon=float(telemetry_row["lon"]) if telemetry_row else 0.0,
                    altitude=float(telemetry_row["alt"]) if telemetry_row else 0.0,
                    yaw=float(telemetry_row["yaw"]) if telemetry_row else 0.0,
                    meta={
                        "video": str(video_path),
                        "video_time_s": timestamp,
                        "has_telemetry": has_truth,
                    },
                    has_truth=has_truth,
                )
            )
            saved += 1
        index += 1
    capture.release()
    payload = {
        "meta": {
            "kind": "video",
            "video": str(video_path),
            "telemetry": str(telemetry_csv) if telemetry_csv else None,
            "sample_fps": sample_fps,
            "count": len(queries),
        },
        "queries": [query.as_dict() for query in queries],
    }
    (out_dir / "queries.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return queries
