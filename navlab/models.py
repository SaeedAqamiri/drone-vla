from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def now() -> datetime:
    return datetime.now(timezone.utc)


class Mission(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    altitude_m: float = 200.0
    speed_kmh: float = 180.0
    corridor_km: float = 2.0
    camera_fov: float = 73.0
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class MapSource(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    mission_id: int | None = Field(default=None, foreign_key="mission.id")
    kind: str = "satellite"
    filename: str
    path: str
    preview_path: str | None = None
    width: int | None = None
    height: int | None = None
    bounds: dict[str, float] | None = Field(default=None, sa_column=Column(JSON))
    crs: str | None = None
    gsd_m: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class Job(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    mission_id: int | None = Field(default=None, foreign_key="mission.id")
    kind: str
    status: str = "queued"
    progress: int = 0
    total: int = 0
    message: str = ""
    backend: str = ""
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class GeneratedTile(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    job_id: int | None = Field(default=None, foreign_key="job.id")
    mission_id: int | None = Field(default=None, foreign_key="mission.id")
    source_path: str
    generated_path: str
    bounds: dict[str, float] | None = Field(default=None, sa_column=Column(JSON))
    prompt: str = ""
    seed: int = 0
    variant: str = ""
    meta: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)


class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    mission_id: int | None = Field(default=None, foreign_key="mission.id")
    job_id: int | None = Field(default=None, foreign_key="job.id")
    name: str
    algorithm: str = "sift_ransac"
    status: str = "done"
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
