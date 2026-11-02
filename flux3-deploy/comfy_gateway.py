"""Queued, parallel HTTP API in front of one or more ComfyUI workers.

Each ComfyUI process executes prompts serially, so the gateway runs N ComfyUI
instances (one per port) and dispatches jobs to whichever worker is free.  That
gives the service a durable-ish job queue *and* real concurrent execution.

Endpoints (FastAPI, docs at /docs):
  POST /v1/jobs                 enqueue a job
  GET  /v1/jobs/{id}            job status / result
  GET  /v1/jobs/{id}/image      result PNG
  GET  /v1/jobs                 recent jobs
  DELETE /v1/jobs/{id}          cancel a queued job
  GET  /v1/queue                queue depths
  GET  /v1/health               gateway + ComfyUI worker health
  POST /v1/generate?wait=true   enqueue and wait for the result

The graph is the one validated for FLUX.3 Klein int8 on ComfyUI:
UNETLoader -> CLIPLoader(type=flux3) -> CLIPTextEncode -> EmptyFlux2LatentImage
-> ModelSamplingFlux -> KSampler(euler/simple, cfg=1) -> VAEDecode -> SaveImage,
with optional VAEEncode -> ReferenceLatent chains for reference images.

Environment:
  COMFY_URLS      comma separated, default http://127.0.0.1:8188
  COMFY_INPUT_DIR shared ComfyUI --input-directory
  OUTPUT_DIR      gateway output directory
  DB_PATH         sqlite file
  HOST/PORT       gateway bind, default 0.0.0.0:8090
  API_TOKEN       optional; require X-API-Key
  DEFAULT_STEPS   default 16
  DEFAULT_ALPHA   default 16.0
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("FLUX3_ROOT", "/home/ava/flux3-klein"))
COMFY_URLS = [u.strip().rstrip("/") for u in os.environ.get(
    "COMFY_URLS", "http://127.0.0.1:8188").split(",") if u.strip()]
COMFY_INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", str(ROOT / "comfy_input")))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(ROOT / "outputs")))
DB_PATH = Path(os.environ.get("DB_PATH", str(OUTPUT_DIR / "jobs.db")))
API_TOKEN = os.environ.get("API_TOKEN", "")
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "16"))
DEFAULT_ALPHA = float(os.environ.get("DEFAULT_ALPHA", "16.0"))
GENERATION_TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "1800"))

for d in (COMFY_INPUT_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DB_LOCK = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
with DB_LOCK:
    _conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            prompt TEXT NOT NULL,
            negative TEXT NOT NULL DEFAULT '',
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            steps INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            alpha REAL NOT NULL,
            references_json TEXT NOT NULL DEFAULT '[]',
            filename_prefix TEXT NOT NULL DEFAULT 'api',
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            comfy_url TEXT,
            prompt_id TEXT,
            image_path TEXT,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at);
        """
    )
    _conn.commit()


def db_exec(sql: str, params: tuple = ()) -> None:
    with DB_LOCK:
        _conn.execute(sql, params)
        _conn.commit()


def db_one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    with DB_LOCK:
        row = _conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def db_all(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with DB_LOCK:
        rows = _conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


class JobRequest(BaseModel):
    prompt: str = Field(min_length=1)
    negative: str = ""
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    alpha: float | None = Field(default=None, gt=1.0, le=1000.0)
    references: list[str] = Field(default_factory=list)
    filename_prefix: str = "api"


class JobResponse(BaseModel):
    id: str
    status: str
    queue_position: int | None = None
    image_url: str | None = None


def require_token(x_api_key: str | None = Header(default=None)) -> None:
    if API_TOKEN and x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def worker_count() -> int:
    return len(COMFY_URLS)


async def stage_reference(data: str, index: int, width: int, height: int) -> str:
    """Decode a base64/data-URI image into the shared ComfyUI input dir."""
    try:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        raw = base64.b64decode(data, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"reference {index} is not a valid image: {error}")
    if index == 0:
        image = image.resize((width, height), Image.LANCZOS)
    else:
        image.thumbnail((max(width, height), max(width, height)), Image.LANCZOS)
    name = f"flux3_ref_{uuid.uuid4().hex[:16]}_{index}.png"
    image.save(COMFY_INPUT_DIR / name, format="PNG")
    return name


def build_graph(prompt: str, negative: str, width: int, height: int, steps: int,
                seed: int, alpha: float, names: list[str], prefix: str) -> dict[str, Any]:
    shift = math.log(alpha)
    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "flux3-image-klein-int8-convrot.safetensors",
            "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_4b_int8_convrot.safetensors",
            "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "10": {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": width, "height": height, "batch_size": 1}},
        "11": {"class_type": "ModelSamplingFlux", "inputs": {
            "model": ["1", 0], "max_shift": shift, "base_shift": shift,
            "width": width, "height": height}},
        "12": {"class_type": "KSampler", "inputs": {
            "model": ["11", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple",
            "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["10", 0], "denoise": 1.0}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {
            "images": ["13", 0], "filename_prefix": prefix}},
    }
    positive: list[Any] = ["4", 0]
    neg: list[Any] = ["5", 0]
    node_id = 20
    for name in names:
        load_id, enc_id = str(node_id), str(node_id + 1)
        pos_ref, neg_ref = str(node_id + 2), str(node_id + 3)
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph[enc_id] = {"class_type": "VAEEncode", "inputs": {
            "pixels": [load_id, 0], "vae": ["3", 0]}}
        graph[pos_ref] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": positive, "latent": [enc_id, 0]}}
        graph[neg_ref] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": neg, "latent": [enc_id, 0]}}
        positive, neg = [pos_ref, 0], [neg_ref, 0]
        node_id += 4
    graph["12"]["inputs"]["positive"] = positive
    graph["12"]["inputs"]["negative"] = neg
    return graph


async def json_post(session: aiohttp.ClientSession, url: str, payload: dict,
                    timeout: float = 120.0) -> dict:
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        text = await r.text()
        if r.status >= 400:
            raise RuntimeError(f"POST {url} -> {r.status}: {text[:2000]}")
        return json.loads(text)


async def system_stats(session: aiohttp.ClientSession, url: str, timeout: float = 3.0) -> bool:
    try:
        async with session.get(f"{url}/system_stats", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            return r.status == 200
    except Exception:
        return False


async def wait_history(session: aiohttp.ClientSession, url: str, prompt_id: str,
                       timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        try:
            async with session.get(f"{url}/history/{prompt_id}",
                                   timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status != 200:
                    continue
                history = await r.json()
        except Exception:
            continue
        item = history.get(prompt_id)
        if not item:
            continue
        status = item.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status)[:2000]}")
        if status.get("completed"):
            return item.get("outputs", {})
    raise TimeoutError(f"generation timed out after {timeout:.0f}s")


async def fetch_image(session: aiohttp.ClientSession, url: str, outputs: dict,
                      dest: Path) -> None:
    files: list[dict] = []
    for node_output in outputs.values():
        files.extend(node_output.get("images", []))
    if not files:
        raise RuntimeError(f"ComfyUI returned no image: {json.dumps(outputs)[:1000]}")
    first = files[0]
    params = {"filename": first["filename"], "subfolder": first.get("subfolder", ""),
              "type": first.get("type", "output")}
    async with session.get(f"{url}/view", params=params,
                           timeout=aiohttp.ClientTimeout(total=120)) as r:
        if r.status != 200:
            raise RuntimeError(f"GET /view -> {r.status}")
        blob = await r.read()
    Image.open(io.BytesIO(blob)).convert("RGB").save(dest, format="PNG")


async def run_job(session: aiohttp.ClientSession, url: str, job: dict) -> None:
    job_id = job["id"]
    started = time.time()
    db_exec("UPDATE jobs SET status='running', started_at=?, comfy_url=? WHERE id=?",
            (started, url, job_id))
    try:
        width, height = job["width"], job["height"]
        names = []
        refs = json.loads(job["references_json"])
        for i, ref in enumerate(refs):
            names.append(await stage_reference(ref, i, width, height))
        prefix = f"flux3_{job['filename_prefix']}_{job_id[:8]}"
        graph = build_graph(job["prompt"], job["negative"], width, height,
                            job["steps"], job["seed"], job["alpha"], names, prefix)
        response = await json_post(session, f"{url}/prompt",
                                   {"prompt": graph,
                                    "client_id": f"flux3-gw-{uuid.uuid4().hex[:8]}"})
        if response.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected graph: {json.dumps(response)[:3000]}")
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {response}")
        db_exec("UPDATE jobs SET prompt_id=? WHERE id=?", (prompt_id, job_id))
        outputs = await wait_history(session, url, prompt_id, GENERATION_TIMEOUT)
        dest = OUTPUT_DIR / f"{job_id}.png"
        await fetch_image(session, url, outputs, dest)
        db_exec("UPDATE jobs SET status='succeeded', finished_at=?, image_path=?, error=NULL "
                "WHERE id=?", (time.time(), str(dest), job_id))
    except Exception as error:
        attempts = int(job.get("attempts") or 0) + 1
        if attempts < 2:
            db_exec("UPDATE jobs SET status='queued', attempts=?, started_at=NULL, comfy_url=NULL, "
                    "error=? WHERE id=?", (attempts, str(error)[:1500], job_id))
            await asyncio.sleep(2.0)
            await QUEUE.put(job_id)
        else:
            db_exec("UPDATE jobs SET status='failed', finished_at=?, attempts=?, error=? WHERE id=?",
                    (time.time(), attempts, str(error)[:2000], job_id))


QUEUE: asyncio.Queue[str]
app = FastAPI(title="FLUX.3 Klein ComfyUI queue", version="1.0")


async def worker_loop(url: str, index: int) -> None:
    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            if not await system_stats(session, url):
                await asyncio.sleep(2.0)
                continue
            job_id = await QUEUE.get()
            try:
                job = db_one("SELECT * FROM jobs WHERE id=?", (job_id,))
                if not job or job["status"] != "queued":
                    continue
                await run_job(session, url, job)
            except Exception as error:
                db_exec("UPDATE jobs SET status='failed', finished_at=?, error=? WHERE id=?",
                        (time.time(), f"worker {index}: {error}"[:2000], job_id))
            finally:
                QUEUE.task_done()


@app.on_event("startup")
async def startup() -> None:
    global QUEUE
    QUEUE = asyncio.Queue()
    for job in db_all("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at"):
        await QUEUE.put(job["id"])
    db_exec("UPDATE jobs SET status='failed', finished_at=?, error='gateway restarted while running' "
            "WHERE status='running'", (time.time(),))
    for i, url in enumerate(COMFY_URLS):
        asyncio.create_task(worker_loop(url, i))


def queue_position(job_id: str) -> int:
    job = db_one("SELECT created_at FROM jobs WHERE id=?", (job_id,))
    if not job:
        return 0
    row = db_one("SELECT COUNT(*) AS n FROM jobs WHERE status='queued' AND created_at < ?",
                 (job["created_at"],))
    return int(row["n"]) + 1


def job_public(job: dict) -> dict:
    out = {k: job[k] for k in (
        "id", "status", "prompt", "negative", "width", "height", "steps", "seed",
        "alpha", "filename_prefix", "attempts", "created_at", "started_at",
        "finished_at", "comfy_url", "prompt_id", "error")}
    out["references"] = len(json.loads(job["references_json"]))
    out["queue_position"] = queue_position(job["id"]) if job["status"] == "queued" else None
    out["image_url"] = f"/v1/jobs/{job['id']}/image" if job["status"] == "succeeded" else None
    return out


@app.post("/v1/jobs", response_model=JobResponse, dependencies=[Depends(require_token)])
async def create_job(req: JobRequest) -> JobResponse:
    if len(req.references) > 10:
        raise HTTPException(status_code=422, detail="at most 10 reference images")
    total = sum(len(r) for r in req.references)
    if total > 80_000_000:
        raise HTTPException(status_code=413, detail="references too large (max ~60 MB decoded)")
    width, height = req.width, req.height
    if width is None or height is None:
        if req.references:
            raw = req.references[0]
            if raw.startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                image = Image.open(io.BytesIO(base64.b64decode(raw, validate=True)))
                width = width or image.width
                height = height or image.height
            except Exception:
                width, height = width or 1024, height or 1024
        else:
            width, height = width or 1024, height or 1024
    width = max(64, min(4096, int(width) // 16 * 16))
    height = max(64, min(4096, int(height) // 16 * 16))
    job_id = uuid.uuid4().hex
    seed = req.seed if req.seed is not None else int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    steps = req.steps or DEFAULT_STEPS
    alpha = req.alpha or DEFAULT_ALPHA
    db_exec("INSERT INTO jobs (id,status,prompt,negative,width,height,steps,seed,alpha,"
            "references_json,filename_prefix,created_at) VALUES "
            "(?,'queued',?,?,?,?,?,?,?,?,?,?)",
            (job_id, req.prompt, req.negative, width, height, steps, seed, alpha,
             json.dumps(req.references), req.filename_prefix, time.time()))
    await QUEUE.put(job_id)
    return JobResponse(id=job_id, status="queued", queue_position=queue_position(job_id))


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
async def get_job(job_id: str) -> dict:
    job = db_one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job_public(job)


@app.get("/v1/jobs/{job_id}/image", dependencies=[Depends(require_token)])
async def get_image(job_id: str) -> FileResponse:
    job = db_one("SELECT status,image_path FROM jobs WHERE id=?", (job_id,))
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "succeeded" or not job["image_path"]:
        raise HTTPException(status_code=409, detail=f"job is {job['status']}")
    path = Path(job["image_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="image file missing")
    return FileResponse(path, media_type="image/png", filename=f"{job_id}.png")


@app.get("/v1/jobs", dependencies=[Depends(require_token)])
async def list_jobs(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    rows = db_all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    return {"jobs": [job_public(r) for r in rows]}


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(require_token)])
async def cancel_job(job_id: str) -> dict:
    job = db_one("SELECT status FROM jobs WHERE id=?", (job_id,))
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "queued":
        raise HTTPException(status_code=409, detail=f"cannot cancel job in status {job['status']}")
    db_exec("UPDATE jobs SET status='cancelled', finished_at=?, error='cancelled by client' WHERE id=?",
            (time.time(), job_id))
    return {"id": job_id, "status": "cancelled"}


@app.get("/v1/queue")
async def get_queue() -> dict:
    rows = db_all("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
    counts = {r["status"]: r["n"] for r in rows}
    return {"workers": worker_count(), "counts": counts,
            "queued": counts.get("queued", 0), "running": counts.get("running", 0)}


@app.get("/v1/health")
async def health() -> dict:
    async with aiohttp.ClientSession() as session:
        checks = await asyncio.gather(*(system_stats(session, u) for u in COMFY_URLS))
    return {"gateway": "ok", "workers": worker_count(),
            "comfyui": [{"url": u, "ok": ok} for u, ok in zip(COMFY_URLS, checks)]}


@app.post("/v1/generate", dependencies=[Depends(require_token)])
async def generate_now(req: JobRequest, timeout: float = Query(default=1800, ge=10, le=7200)) -> dict:
    created = await create_job(req)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        job = db_one("SELECT * FROM jobs WHERE id=?", (created.id,))
        if job and job["status"] in ("succeeded", "failed", "cancelled"):
            return job_public(job)
    raise HTTPException(status_code=504, detail=f"job {created.id} still {job['status']} after {timeout}s")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8090")), log_level="info")
