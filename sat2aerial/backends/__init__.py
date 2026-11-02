from __future__ import annotations

from .base import Backend
from .dummy import DummyBackend


def make_backend(name: str, **kwargs) -> Backend:
    key = name.lower().strip()
    if key in {"dummy", "offline", "smoke"}:
        return DummyBackend(**kwargs)
    if key in {"bfl", "api", "bfl_api"}:
        from .bfl_api import BFLBackend

        return BFLBackend(**kwargs)
    if key in {"flux3", "local", "flux3_local"}:
        from .flux3_local import Flux3LocalBackend

        return Flux3LocalBackend(**kwargs)
    if key in {"comfy", "comfyui"}:
        from .comfy import ComfyBackend

        return ComfyBackend(**kwargs)
    if key in {"sglang", "sgl"}:
        from .sglang import SGLangBackend

        return SGLangBackend(**kwargs)
    raise ValueError(
        f"Unknown backend {name!r}; use dummy, comfy, bfl, sglang, or flux3"
    )


__all__ = ["Backend", "make_backend"]
