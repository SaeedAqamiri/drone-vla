from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Sequence

from sat2aerial.backends import make_backend
from sat2aerial.prompts import build_prompt

from .config import BFL_API_KEY, COMFY_INPUT, COMFY_URL

BACKENDS = ["dummy", "comfy", "bfl"]
VIEWS = ["nadir", "oblique15", "oblique30", "oblique45"]


def parse_size(raw: str) -> tuple[int, int]:
    token = raw.lower().strip()
    if "x" in token:
        width, height = token.split("x", 1)
        return int(width), int(height)
    value = int(token)
    return value, value


def comfy_available(url: str = COMFY_URL, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/system_stats", timeout=timeout):
            return True
    except Exception:
        return False


def backend_status() -> dict:
    return {
        "comfy_url": COMFY_URL,
        "comfy_available": comfy_available(),
        "bfl_key": bool(BFL_API_KEY),
        "backends": BACKENDS,
    }


def get_backend(name: str, seed: int = 0):
    if name == "dummy":
        return make_backend("dummy", seed=seed)
    if name == "comfy":
        return make_backend(
            "comfy",
            seed=seed,
            server=COMFY_URL,
            input_dir=str(COMFY_INPUT),
        )
    if name == "bfl":
        if not BFL_API_KEY:
            raise RuntimeError("BFL_API_KEY is not set; use dummy or start ComfyUI")
        return make_backend("bfl", seed=seed, api_key=BFL_API_KEY)
    raise ValueError(f"Unknown backend {name!r}")


def generate_reference(
    satellite: Path,
    out_path: Path,
    backend: str = "dummy",
    view: str = "nadir",
    seed: int = 0,
    size: str = "768x768",
    altitude_m: float = 200.0,
    style_refs: Sequence[Path] = (),
) -> dict:
    width, height = parse_size(size)
    prompt = build_prompt(
        view=view,
        style_refs=len(style_refs),
        altitude_m=int(altitude_m),
        season="summer",
        lighting="clear midday sunlight",
        weather="clear",
    )
    instance = get_backend(backend, seed=seed)
    meta = instance.generate(
        satellite=satellite,
        prompt=prompt,
        out_path=out_path,
        size=(width, height),
        seed=seed,
        style_refs=style_refs,
    )
    meta["prompt"] = prompt
    meta["view"] = view
    meta["size"] = [width, height]
    return meta
