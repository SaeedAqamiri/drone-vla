from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import cv2


class Backend(ABC):
    name = "base"
    model = "none"

    def __init__(self, seed: int = 0):
        self.seed = seed

    @abstractmethod
    def generate(
        self,
        satellite: Path,
        prompt: str,
        out_path: Path,
        size: tuple[int, int],
        seed: int,
        style_refs: Sequence[Path] = (),
    ) -> dict:
        raise NotImplementedError

    def _finish(self, out_path: Path, started: float, meta: dict) -> dict:
        payload = dict(meta)
        payload.setdefault("backend", self.name)
        payload.setdefault("model", self.model)
        payload.setdefault("out", str(out_path))
        payload["seconds"] = round(time.time() - started, 3)
        return payload


def save_png(image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image):
        raise RuntimeError(f"Failed to write {out_path}")
