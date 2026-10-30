from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from .camera import LATEST as CAMERA_LATEST
from .config import MISSIONS_DIR, OUTPUTS, QUICK_DIR, ROOT
from .db import engine, get_session, init_db
from .generation import BACKENDS, VIEWS, backend_status
from .geo import mission_route_points, route_summary_multi
from .imagery import inspect_image, make_preview, save_bytes
from .jobs import enqueue, start_worker
from .models import GeneratedTile, Job, MapSource, Mission, Run
from .sim import SIM

app = FastAPI(title="NavLab", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "navlab" / "static")), name="static")
app.mount("/media", StaticFiles(directory=str(OUTPUTS)), name="media")

templates = Jinja2Templates(directory=str(ROOT / "navlab" / "templates"))


def media_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to(OUTPUTS.resolve())
        return "/media/" + relative.as_posix()
    except Exception:
        return None


templates.env.globals["media_url"] = media_url
templates.env.globals["backends"] = BACKENDS
templates.env.globals["views"] = VIEWS

SessionDep = Annotated[Session, Depends(get_session)]


def _startup() -> None:
    init_db()
    start_worker()


app.on_event("startup")(_startup)


def _mission_or_404(session: Session, mission_id: int) -> Mission:
    mission = session.get(Mission, mission_id)
    if mission is None:
        raise HTTPException(status_code=404, detail="Mission not found")
    return mission


def _ensure_route(session: Session, mission: Mission) -> dict[str, Any]:
    source = session.exec(
        select(MapSource)
        .where(MapSource.mission_id == mission.id, MapSource.kind == "satellite")
        .order_by(MapSource.id.desc())
    ).first()
    gsd = source.gsd_m if source else None
    summary = route_summary_multi(
        mission_route_points(mission),
        mission.corridor_km,
        gsd,
    )
    mission.meta = {**(mission.meta or {}), "route": summary}
    session.add(mission)
    session.commit()
    session.refresh(mission)
    return summary


def _mission_context(session: Session, mission: Mission) -> dict[str, Any]:
    maps = list(
        session.exec(
            select(MapSource)
            .where(MapSource.mission_id == mission.id)
            .order_by(MapSource.id.desc())
        )
    )
    satellite = next((item for item in maps if item.kind == "satellite"), None)
    aerials = [item for item in maps if item.kind == "aerial"]
    generated = list(
        session.exec(
            select(GeneratedTile)
            .where(GeneratedTile.mission_id == mission.id)
            .order_by(GeneratedTile.id.desc())
        )
    )
    jobs = list(
        session.exec(
            select(Job).where(Job.mission_id == mission.id).order_by(Job.id.desc()).limit(20)
        )
    )
    runs = list(
        session.exec(
            select(Run).where(Run.mission_id == mission.id).order_by(Run.id.desc()).limit(20)
        )
    )
    summary = mission.meta.get("route") if mission.meta else None
    if not summary or "points" not in summary:
        summary = _ensure_route(session, mission)
    overlay = None
    if satellite and satellite.bounds and satellite.preview_path:
        overlay = {
            "url": media_url(satellite.preview_path),
            "bounds": [
                [satellite.bounds["min_lat"], satellite.bounds["min_lon"]],
                [satellite.bounds["max_lat"], satellite.bounds["max_lon"]],
            ],
        }
    return {
        "mission": mission,
        "summary": summary,
        "maps": maps,
        "satellite": satellite,
        "aerials": aerials,
        "generated": generated,
        "jobs": jobs,
        "runs": runs,
        "overlay": overlay,
    }


def index(request: Request, session: SessionDep):
    missions = list(session.exec(select(Mission).order_by(Mission.id.desc())))
    counts = {
        "missions": len(missions),
        "jobs": len(list(session.exec(select(Job)))),
        "runs": len(list(session.exec(select(Run)))),
        "tiles": len(list(session.exec(select(GeneratedTile)))),
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "missions": missions,
            "counts": counts,
            "backend": backend_status(),
            "sim": SIM.status(),
        },
    )


def create_mission(
    session: SessionDep,
    name: str = Form(...),
    start_lat: float = Form(...),
    start_lon: float = Form(...),
    end_lat: float = Form(...),
    end_lon: float = Form(...),
    altitude_m: float = Form(200.0),
    speed_kmh: float = Form(180.0),
    corridor_km: float = Form(2.0),
    camera_fov: float = Form(73.0),
    waypoints: str = Form(""),
):
    parsed: list[list[float]] = []
    for line in waypoints.replace(";", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part for part in line.replace(",", " ").split() if part]
        if len(parts) < 2:
            continue
        lat = float(parts[0])
        lon = float(parts[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise HTTPException(status_code=400, detail="invalid waypoint")
        parsed.append([lat, lon])
    if len(parsed) > 30:
        raise HTTPException(status_code=400, detail="too many waypoints")
    mission = Mission(
        name=name,
        start_lat=start_lat,
        start_lon=start_lon,
        end_lat=end_lat,
        end_lon=end_lon,
        altitude_m=altitude_m,
        speed_kmh=speed_kmh,
        corridor_km=corridor_km,
        camera_fov=camera_fov,
        meta={"waypoints": parsed},
    )
    session.add(mission)
    session.commit()
    session.refresh(mission)
    _ensure_route(session, mission)
    return RedirectResponse(url=f"/missions/{mission.id}", status_code=303)


def mission_detail(request: Request, mission_id: int, session: SessionDep):
    mission = _mission_or_404(session, mission_id)
    return templates.TemplateResponse(
        request=request,
        name="mission.html",
        context=_mission_context(session, mission),
    )


def _save_upload(upload: UploadFile, directory: Path) -> tuple[Path, str]:
    filename = Path(upload.filename or "upload.bin").name
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{uuid.uuid4().hex[:8]}_{filename}"
    save_bytes(upload.file.read(), dest)
    return dest, filename


def upload_map(
    mission_id: int,
    session: SessionDep,
    kind: str = Form("satellite"),
    file: UploadFile = File(...),
):
    mission = _mission_or_404(session, mission_id)
    if kind not in {"satellite", "aerial"}:
        raise HTTPException(status_code=400, detail="kind must be satellite or aerial")
    mission_dir = MISSIONS_DIR / str(mission.id)
    dest, filename = _save_upload(file, mission_dir / "uploads")
    info = inspect_image(dest)
    preview = make_preview(dest, mission_dir / "previews" / f"{dest.stem}.png")
    source = MapSource(
        mission_id=mission.id,
        kind=kind,
        filename=filename,
        path=str(dest),
        preview_path=str(preview) if preview else None,
        width=info.get("width"),
        height=info.get("height"),
        bounds=info.get("bounds"),
        crs=info.get("crs"),
        gsd_m=info.get("gsd_m"),
        meta={"content_type": file.content_type or ""},
    )
    session.add(source)
    session.commit()
    _ensure_route(session, mission)
    return RedirectResponse(url=f"/missions/{mission.id}", status_code=303)


def start_generation(
    mission_id: int,
    session: SessionDep,
    backend: str = Form("dummy"),
    view: str = Form("nadir"),
    seed: int = Form(0),
    limit: int = Form(4),
    tile_px: int = Form(1024),
    size: str = Form("768x768"),
):
    mission = _mission_or_404(session, mission_id)
    if backend not in BACKENDS:
        raise HTTPException(status_code=400, detail="unknown backend")
    if view not in VIEWS:
        raise HTTPException(status_code=400, detail="unknown view")
    job = Job(
        mission_id=mission.id,
        kind="generate",
        backend=backend,
        config={
            "backend": backend,
            "view": view,
            "seed": seed,
            "limit": limit,
            "tile_px": tile_px,
            "size": size,
        },
        message="queued",
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    enqueue(job.id)
    return RedirectResponse(url=f"/jobs/{job.id}", status_code=303)


def start_matching(
    mission_id: int,
    session: SessionDep,
    aerial_id: int | None = Form(None),
    mode: str = Form("edges"),
):
    mission = _mission_or_404(session, mission_id)
    job = Job(
        mission_id=mission.id,
        kind="match",
        backend="sift",
        config={"aerial_id": aerial_id, "algorithm": "sift_ransac", "mode": mode},
        message="queued",
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    enqueue(job.id)
    return RedirectResponse(url=f"/jobs/{job.id}", status_code=303)


def job_detail(request: Request, job_id: int, session: SessionDep):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(
        request=request,
        name="job.html",
        context={"job": job, "result": job.result or {}},
    )


def job_json(job_id: int, session: SessionDep):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(
        {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "progress": job.progress,
            "total": job.total,
            "message": job.message,
            "result": job.result or {},
            "mission_id": job.mission_id,
        }
    )


def run_detail(request: Request, run_id: int, session: SessionDep):
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    frames = (run.result or {}).get("frames", [])
    return templates.TemplateResponse(
        request=request,
        name="run.html",
        context={"run": run, "frames": frames, "metrics": run.metrics or {}},
    )


def quick_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="quick.html",
        context={"backend": backend_status()},
    )


def quick_run(
    satellite: UploadFile = File(...),
    aerial: UploadFile = File(...),
    backend: str = Form("dummy"),
    view: str = Form("nadir"),
    seed: int = Form(0),
    size: str = Form("768x768"),
):
    if backend not in BACKENDS:
        raise HTTPException(status_code=400, detail="unknown backend")
    directory = QUICK_DIR / uuid.uuid4().hex[:12]
    sat_path, _ = _save_upload(satellite, directory)
    aerial_path, _ = _save_upload(aerial, directory)
    with Session(engine) as session:
        job = Job(
            kind="quick",
            backend=backend,
            config={
                "satellite_path": str(sat_path),
                "aerial_path": str(aerial_path),
                "backend": backend,
                "view": view,
                "seed": seed,
                "size": size,
            },
            message="queued",
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id
    enqueue(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


def sim_page(request: Request, session: SessionDep):
    missions = list(session.exec(select(Mission).order_by(Mission.id.desc())))
    return templates.TemplateResponse(
        request=request,
        name="sim.html",
        context={"missions": missions, "sim": SIM.status()},
    )


async def sim_start(
    mission_id: int = Form(...),
    gps_denied: int = Form(0),
    source: str = Form("truth"),
    map_dir: str | None = Form(None),
):
    with Session(engine) as session:
        mission = session.get(Mission, mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="Mission not found")
    result = await run_in_threadpool(SIM.start, mission)
    if result.get("ok"):
        if gps_denied:
            result["ev"] = SIM.enable_gps_denied(mission, source, map_dir)
        else:
            result["gps_restored"] = SIM.restore_gps_async()
    return JSONResponse(result)


def sim_stop():
    return JSONResponse(SIM.stop())


def sim_ev_start(
    mission_id: int = Form(...),
    source: str = Form("truth"),
    map_dir: str | None = Form(None),
):
    with Session(engine) as session:
        mission = session.get(Mission, mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="Mission not found")
    return JSONResponse(SIM.enable_gps_denied(mission, source, map_dir))


def sim_ev_stop():
    return JSONResponse(SIM.disable_gps_denied())


def sim_status():
    return JSONResponse(SIM.status())


def sim_fly(mission_id: int = Form(...)):
    with Session(engine) as session:
        mission = session.get(Mission, mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="Mission not found")
    return JSONResponse(SIM.fly_to_end(mission))


def sim_camera():
    if not CAMERA_LATEST.exists():
        raise HTTPException(status_code=404, detail="no camera frame")
    return FileResponse(
        str(CAMERA_LATEST),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def sim_camera_meta():
    return JSONResponse(SIM.camera.status())


def hardware_page(request: Request):
    return templates.TemplateResponse(request=request, name="hardware.html", context={})


def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"backend": backend_status(), "outputs": str(OUTPUTS)},
    )


def health():
    status = SIM.status()
    return JSONResponse(
        {
            "ok": True,
            "backend": backend_status(),
            "sim": {"running": status["running"], "connected": status["connected"]},
        }
    )


app.add_api_route("/", index, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/missions", create_mission, methods=["POST"])
app.add_api_route("/missions/{mission_id}", mission_detail, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/missions/{mission_id}/upload", upload_map, methods=["POST"])
app.add_api_route("/missions/{mission_id}/generate", start_generation, methods=["POST"])
app.add_api_route("/missions/{mission_id}/match", start_matching, methods=["POST"])
app.add_api_route("/jobs/{job_id}", job_detail, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/api/jobs/{job_id}", job_json, methods=["GET"])
app.add_api_route("/runs/{run_id}", run_detail, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/quick", quick_page, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/quick", quick_run, methods=["POST"])
app.add_api_route("/sim", sim_page, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/api/sim/start", sim_start, methods=["POST"])
app.add_api_route("/api/sim/stop", sim_stop, methods=["POST"])
app.add_api_route("/api/sim/ev/start", sim_ev_start, methods=["POST"])
app.add_api_route("/api/sim/ev/stop", sim_ev_stop, methods=["POST"])
app.add_api_route("/api/sim/status", sim_status, methods=["GET"])
app.add_api_route("/api/sim/fly", sim_fly, methods=["POST"])
app.add_api_route("/api/sim/camera.jpg", sim_camera, methods=["GET"])
app.add_api_route("/api/sim/camera", sim_camera_meta, methods=["GET"])
app.add_api_route("/hardware", hardware_page, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/settings", settings_page, methods=["GET"], response_class=HTMLResponse)
app.add_api_route("/api/health", health, methods=["GET"])
