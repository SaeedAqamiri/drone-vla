from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Sequence

import requests
from PIL import Image

from .base import Backend


class SGLangBackend(Backend):
    """FLUX.3 Klein served by SGLang's OpenAI-compatible image API."""

    name = "sglang"
    model = "black-forest-labs/flux-3-image-klein (fp8r, SGLang)"

    def __init__(
        self,
        server: str = "http://127.0.0.1:30010",
        steps: int = 16,
        timeout: float = 1800.0,
        output_format: str = "png",
        **kwargs,
    ):
        super().__init__(seed=kwargs.pop("seed", 0))
        self.server = server.rstrip("/")
        self.steps = steps
        self.timeout = timeout
        self.output_format = output_format

    def _check(self) -> None:
        try:
            response = requests.get(f"{self.server}/health", timeout=5)
            response.raise_for_status()
        except Exception as error:
            raise RuntimeError(
                f"SGLang is not reachable at {self.server}. Start it with:\n"
                "  bash /home/ava/flux3-klein/api/sglang_start.sh\n"
                f"Original error: {error}"
            ) from error

    def generate(
        self,
        satellite: Path,
        prompt: str,
        out_path: Path,
        size: tuple[int, int],
        seed: int,
        style_refs: Sequence[Path] = (),
    ) -> dict:
        started = time.time()
        self._check()
        images = [Path(satellite), *[Path(item) for item in style_refs]]
        files = []
        for path in images:
            files.append(
                (
                    "image",
                    (path.name, path.read_bytes(), "image/png"),
                )
            )
        data = {
            "prompt": prompt,
            "size": f"{int(size[0])}x{int(size[1])}",
            "num_inference_steps": str(int(self.steps)),
            "seed": str(int(seed)),
            "response_format": "b64_json",
            "output_format": self.output_format,
        }
        response = requests.post(
            f"{self.server}/v1/images/edits",
            data=data,
            files=files,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"SGLang rejected the request ({response.status_code}): "
                f"{response.text[:2000]}"
            )
        payload = response.json()
        items = payload.get("data") or []
        if not items or not items[0].get("b64_json"):
            raise RuntimeError(f"SGLang returned no b64_json image: {payload}")
        image = Image.open(io.BytesIO(base64.b64decode(items[0]["b64_json"]))).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")
        return self._finish(
            out_path,
            started,
            {
                "prompt": prompt,
                "size": list(size),
                "seed": seed,
                "style_refs": [str(item) for item in style_refs],
                "server": self.server,
                "sglang_id": payload.get("id"),
                "sglang_inference_s": payload.get("inference_time_s"),
                "peak_memory_mb": payload.get("peak_memory_mb"),
            },
        )
