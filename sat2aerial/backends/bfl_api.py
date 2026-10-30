from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from PIL import Image

from .base import Backend

DEFAULT_URL = "https://api.bfl.ai/v1/flux-3-image-klein"


class BFLBackend(Backend):
    name = "bfl"
    model = "flux-3-image-klein"

    def __init__(
        self,
        api_key: str | None = None,
        url: str = DEFAULT_URL,
        steps: int = 16,
        safety_tolerance: int = 2,
        data_uri: bool = False,
        poll_interval: float = 1.0,
        poll_timeout: float = 300.0,
        dry_run: bool = False,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.api_key = api_key or os.environ.get("BFL_API_KEY", "")
        self.url = url
        self.steps = steps
        self.safety_tolerance = safety_tolerance
        self.data_uri = data_uri
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.dry_run = dry_run

    def _encode(self, path: Path) -> str:
        raw = base64.b64encode(path.read_bytes()).decode("ascii")
        if self.data_uri:
            return f"data:image/png;base64,{raw}"
        return raw

    def _headers(self) -> dict[str, str]:
        return {
            "x-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, url: str) -> dict:
        request = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _submit(self, payload: dict) -> dict:
        try:
            return self._post(payload)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in {400, 422} and not self.data_uri:
                retry = dict(payload)
                retry["images"] = [
                    item if item.startswith("http") else f"data:image/png;base64,{item}"
                    for item in payload.get("images", [])
                ]
                try:
                    return self._post(retry)
                except urllib.error.HTTPError as second:
                    raise RuntimeError(
                        f"BFL request failed ({second.code}): "
                        f"{second.read().decode('utf-8', errors='replace')[:1000]}"
                    ) from second
            raise RuntimeError(f"BFL request failed ({error.code}): {body[:1000]}") from error

    def _poll(self, polling_url: str) -> str:
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            data = self._get(polling_url)
            status = str(data.get("status", ""))
            if status == "Ready":
                result = data.get("result") or {}
                sample = result.get("sample")
                if not sample:
                    raise RuntimeError(f"BFL task ready without sample: {data}")
                return sample
            if status in {"Error", "Task not found", "Content Moderated", "Request Moderated"}:
                raise RuntimeError(f"BFL task failed: {json.dumps(data)[:1000]}")
            time.sleep(self.poll_interval)
        raise TimeoutError(f"BFL task did not finish within {self.poll_timeout}s")

    def _download(self, url: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            blob = response.read()
        image = Image.open(io.BytesIO(blob)).convert("RGB")
        image.save(out_path, format="PNG")

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
        if not self.api_key and not self.dry_run:
            raise RuntimeError(
                "BFL_API_KEY is not set. Export it or pass --api-key. "
                "Get a key at https://dashboard.bfl.ai"
            )
        width, height = size
        images = [satellite, *style_refs]
        payload = {
            "prompt": prompt,
            "images": [self._encode(path) for path in images],
            "width": int(width),
            "height": int(height),
            "seed": int(seed),
            "steps": int(self.steps),
            "safety_tolerance": int(self.safety_tolerance),
            "output_format": "png",
        }
        if self.dry_run:
            return self._finish(
                out_path,
                started,
                {
                    "prompt": prompt,
                    "size": [width, height],
                    "seed": seed,
                    "style_refs": [str(p) for p in style_refs],
                    "dry_run": True,
                    "payload_bytes": len(json.dumps(payload)),
                },
            )
        response = self._submit(payload)
        polling_url = response.get("polling_url")
        if not polling_url:
            raise RuntimeError(f"BFL response missing polling_url: {response}")
        sample = self._poll(polling_url)
        self._download(sample, out_path)
        return self._finish(
            out_path,
            started,
            {
                "prompt": prompt,
                "size": [width, height],
                "seed": seed,
                "style_refs": [str(p) for p in style_refs],
                "task_id": response.get("id"),
                "cost": response.get("cost"),
                "output_mp": response.get("output_mp"),
            },
        )
