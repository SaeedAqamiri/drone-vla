from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from .config import MISSIONS_DIR, RUNS_DIR
from .db import engine
from .generation import generate_reference
from .geo import mission_route_points, tile_bounds_multi
from .imagery import crop_bounds, whole_image_png
from .matching import best_match, save_montage, save_overlay
from .models import GeneratedTile, Job, MapSource, Mission, Run, now

JOB_QUEUE: queue.Queue[int] = queue.Queue()
_WORKER_LOCK = threading.Lock()
_WORKER_STARTED = False


def start_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        thread = threading.Thread(target=_worker, daemon=True, name="navlab-worker")
        thread.start()
        _WORKER_STARTED = True


def enqueue(job_id: int) -> None:
    start_worker()
    JOB_QUEUE.put(job_id)


def update_job(job_id: int, **fields: Any) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = now()
        session.add(job)
        session.commit()


def _worker() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        try:
            _process(job_id)
        except Exception as error:
            update_job(job_id, status="error", message=f"{type(error).__name__}: {error}")
        finally:
            JOB_QUEUE.task_done()


def _process(job_id: int) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        kind = job.kind
        mission_id = job.mission_id
    update_job(job_id, status="running", message="starting")
    if kind == "generate":
        _run_generation(job_id)
    elif kind == "match":
        _run_matching(job_id)
    elif kind == "quick":
        _run_quick(job_id)
    else:
        raise ValueError(f"Unknown job kind {kind!r}")


def _mission_dir(mission_id: int) -> Path:
    path = MISSIONS_DIR / str(mission_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_map(session: Session, mission_id: int, kind: str) -> MapSource | None:
    statement = (
        select(MapSource)
        .where(MapSource.mission_id == mission_id, MapSource.kind == kind)
        .order_by(MapSource.id.desc())
    )
    return session.exec(statement).first()


def _run_generation(job_id: int) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        mission = session.get(Mission, job.mission_id) if job.mission_id else None
        if mission is None:
            raise ValueError("Mission not found")
        map_source = _latest_map(session, mission.id, "satellite")
        if map_source is None:
            raise ValueError("Upload a satellite map first")
        config = dict(job.config or {})
        mission_id = mission.id
        mission_dir = _mission_dir(mission_id)
        map_path = Path(map_source.path)
        bounds = map_source.bounds
        gsd = map_source.gsd_m or 0.5
        tile_px = int(config.get("tile_px", 1024))
        limit = max(1, min(int(config.get("limit", 4)), 64))
        backend = config.get("backend", "dummy")
        view = config.get("view", "nadir")
        size = config.get("size", "768x768")
        seed = int(config.get("seed", 0))
        altitude = mission.altitude_m
        points = mission_route_points(mission)
        concurrency = int(config.get("concurrency") or 0)
        if concurrency <= 0:
            concurrency = 2 if backend == "sglang" else 1
        concurrency = max(1, min(concurrency, 4))

    tiles: list[dict[str, Any]]
    if bounds:
        tile_ground_m = gsd * tile_px
        tiles = tile_bounds_multi(points, tile_ground_m=tile_ground_m, limit=limit)
    else:
        tiles = [{"index": 0, "center": None, "bounds": None}]

    # Crop source tiles up front; rasterio reads stay single-threaded.
    prepared: list[dict[str, Any]] = []
    for tile in tiles:
        source_png = mission_dir / "tiles" / f"source_{tile['index']:03d}.png"
        if tile.get("bounds"):
            cropped = crop_bounds(map_path, tile["bounds"], source_png, tile_px=tile_px)
        else:
            cropped = whole_image_png(map_path, source_png, size=tile_px)
        if cropped is None:
            continue
        prepared.append(
            {
                "tile": tile,
                "source": source_png,
                "generated": mission_dir / "generated" / f"gen_{tile['index']:03d}.png",
            }
        )

    update_job(
        job_id,
        total=len(prepared),
        progress=0,
        message=f"tiles prepared ({len(prepared)}, concurrency={concurrency})",
    )

    def _generate(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        tile = item["tile"]
        meta = generate_reference(
            satellite=item["source"],
            out_path=item["generated"],
            backend=backend,
            view=view,
            seed=seed + int(tile["index"]),
            size=size,
            altitude_m=altitude,
        )
        return item, meta

    def _record(item: dict[str, Any], meta: dict[str, Any]) -> int:
        with Session(engine) as session:
            record = GeneratedTile(
                job_id=job_id,
                mission_id=mission_id,
                source_path=str(item["source"]),
                generated_path=str(item["generated"]),
                bounds=item["tile"].get("bounds"),
                prompt=str(meta.get("prompt", "")),
                seed=seed + int(item["tile"]["index"]),
                variant=view,
                meta={k: v for k, v in meta.items() if k not in {"prompt"}},
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return int(record.id)

    created: list[int] = []
    errors: list[dict[str, Any]] = []
    done = 0

    def _handle(item: dict[str, Any], meta: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        nonlocal done
        done += 1
        index = item["tile"]["index"]
        if error is not None:
            errors.append({"index": index, "error": str(error)[:1000]})
            update_job(
                job_id,
                progress=done,
                message=f"tile {index} failed ({done}/{len(prepared)})",
            )
            return
        created.append(_record(item, meta or {}))
        update_job(
            job_id,
            progress=done,
            message=f"generated tile {index} ({done}/{len(prepared)})",
        )

    if concurrency <= 1 or len(prepared) <= 1:
        for item in prepared:
            try:
                item, meta = _generate(item)
            except Exception as error:
                _handle(item, error=error)
            else:
                _handle(item, meta=meta)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_generate, item): item for item in prepared}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    item, meta = future.result()
                except Exception as error:
                    _handle(item, error=error)
                else:
                    _handle(item, meta=meta)

    if created:
        update_job(
            job_id,
            status="done",
            progress=len(prepared),
            message="generation finished",
            result={
                "tile_ids": created,
                "backend": backend,
                "view": view,
                "concurrency": concurrency,
                "errors": errors[:20],
            },
        )
    else:
        update_job(
            job_id,
            status="error",
            progress=done,
            message="generation failed for every tile",
            result={
                "backend": backend,
                "view": view,
                "concurrency": concurrency,
                "errors": errors[:20],
            },
        )


def _aerial_sources(session: Session, mission_id: int, aerial_id: int | None) -> list[MapSource]:
    if aerial_id:
        source = session.get(MapSource, aerial_id)
        return [source] if source else []
    statement = (
        select(MapSource)
        .where(MapSource.mission_id == mission_id, MapSource.kind == "aerial")
        .order_by(MapSource.id.asc())
    )
    return list(session.exec(statement))


def _run_matching(job_id: int) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        mission = session.get(Mission, job.mission_id) if job.mission_id else None
        if mission is None:
            raise ValueError("Mission not found")
        config = dict(job.config or {})
        generated: list[GeneratedTile] = []
        ids = config.get("tile_ids") or []
        if ids:
            for tile_id in ids:
                tile = session.get(GeneratedTile, int(tile_id))
                if tile:
                    generated.append(tile)
        else:
            statement = (
                select(GeneratedTile)
                .where(GeneratedTile.mission_id == mission.id)
                .order_by(GeneratedTile.id.asc())
            )
            generated = list(session.exec(statement))
        if not generated:
            raise ValueError("Generate at least one aerial reference first")
        aerials = _aerial_sources(session, mission.id, config.get("aerial_id"))
        mission_id = mission.id

    output_dir = RUNS_DIR / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    if not aerials:
        for tile in generated:
            result = best_match(Path(tile.source_path), [Path(tile.generated_path)])
            ref, payload = result
            frames.append(
                {
                    "aerial": tile.source_path,
                    "best_reference": str(ref),
                    "kind": "consistency",
                    **payload,
                }
            )
    else:
        refs = [Path(tile.generated_path) for tile in generated]
        baseline_refs = [Path(tile.source_path) for tile in generated]
        for aerial in aerials:
            best_ref, payload = best_match(Path(aerial.path), refs)
            baseline_ref, baseline_payload = best_match(Path(aerial.path), baseline_refs)
            if payload and best_ref:
                overlay = output_dir / f"{Path(aerial.filename).stem}_match.png"
                save_overlay(best_ref, Path(aerial.path), overlay)
                montage = output_dir / f"{Path(aerial.filename).stem}_montage.png"
                save_montage(
                    [Path(best_ref), Path(aerial.path)],
                    ["generated reference", "real aerial"],
                    montage,
                )
                payload["overlay"] = str(overlay)
                payload["montage"] = str(montage)
            frames.append(
                {
                    "aerial": aerial.path,
                    "aerial_name": aerial.filename,
                    "best_reference": str(best_ref) if best_ref else None,
                    "generated": payload,
                    "baseline": baseline_payload,
                    "baseline_reference": str(baseline_ref) if baseline_ref else None,
                }
            )

    success_count = sum(
        1
        for frame in frames
        if (frame.get("generated") or frame).get("success")
    )
    aggregate = {
        "frames": len(frames),
        "success": success_count,
        "success_rate": success_count / max(len(frames), 1),
        "mean_inliers": _mean(
            [(frame.get("generated") or frame).get("match", {}).get("inliers") for frame in frames]
        ),
        "mean_ratio": _mean(
            [
                (frame.get("generated") or frame).get("match", {}).get("inlier_ratio")
                for frame in frames
            ]
        ),
        "mean_coverage": _mean(
            [
                (frame.get("generated") or frame).get("match", {}).get("coverage")
                for frame in frames
            ]
        ),
    }
    with Session(engine) as session:
        run = Run(
            mission_id=mission_id,
            job_id=job_id,
            name=f"match job {job_id}",
            algorithm=str(config.get("algorithm", "sift_ransac")),
            status="done",
            metrics=aggregate,
            result={"frames": frames, "output_dir": str(output_dir)},
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id
    update_job(
        job_id,
        status="done",
        total=len(frames),
        progress=len(frames),
        message="matching finished",
        result={"run_id": run_id, "metrics": aggregate},
    )


def _run_quick(job_id: int) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        config = dict(job.config or {})
    satellite = Path(config["satellite_path"])
    aerial = Path(config["aerial_path"])
    backend = config.get("backend", "dummy")
    view = config.get("view", "nadir")
    seed = int(config.get("seed", 0))
    size = config.get("size", "768x768")
    output_dir = RUNS_DIR / f"job_{job_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = output_dir / "generated.png"
    update_job(job_id, total=3, progress=1, message="generating")
    meta = generate_reference(
        satellite=satellite,
        out_path=generated,
        backend=backend,
        view=view,
        seed=seed,
        size=size,
        altitude_m=float(config.get("altitude_m", 200)),
    )
    update_job(job_id, progress=2, message="matching")
    generated_result = best_match(aerial, [generated])
    baseline_result = best_match(aerial, [satellite])
    best_ref, generated_payload = generated_result
    baseline_ref, baseline_payload = baseline_result
    if best_ref:
        save_overlay(best_ref, aerial, output_dir / "match_overlay.png")
        save_montage(
            [satellite, best_ref, aerial],
            ["satellite", "generated", "real aerial"],
            output_dir / "montage.png",
        )
    aggregate = {
        "generated": generated_payload,
        "baseline": baseline_payload,
        "generated_success": bool(generated_payload and generated_payload["success"]),
        "baseline_success": bool(baseline_payload and baseline_payload["success"]),
    }
    with Session(engine) as session:
        run = Run(
            mission_id=None,
            job_id=job_id,
            name=f"quick job {job_id}",
            algorithm="sift_ransac",
            status="done",
            metrics=aggregate,
            result={
                "generated": str(generated),
                "overlay": str(output_dir / "match_overlay.png"),
                "montage": str(output_dir / "montage.png"),
                "meta": {k: v for k, v in meta.items() if k != "prompt"},
            },
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id
    update_job(
        job_id,
        status="done",
        progress=3,
        message="quick test finished",
        result={"run_id": run_id, "metrics": aggregate},
    )


def _mean(values: list[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None
