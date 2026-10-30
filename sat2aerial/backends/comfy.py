from __future__ import annotations

import io
import json
import math
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Sequence

from PIL import Image

from .base import Backend

DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_INPUT_DIR = "ComfyUI/input"


class ComfyBackend(Backend):
    name = "comfy"
    model = "black-forest-labs/flux-3-image-klein (int8, ComfyUI)"

    def __init__(
        self,
        server: str = DEFAULT_SERVER,
        input_dir: str = DEFAULT_INPUT_DIR,
        steps: int = 16,
        alpha: float = 16.0,
        timeout: float = 900.0,
        poll_interval: float = 2.0,
        output_prefix: str = "sat2aerial",
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.server = server.rstrip("/")
        self.input_dir = Path(input_dir)
        self.steps = steps
        self.alpha = alpha
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.output_prefix = output_prefix

    def _request(self, path: str, payload: dict | None = None, timeout: float = 60.0):
        url = self.server + path
        if payload is None:
            request = urllib.request.Request(url, method="GET")
        else:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _check(self) -> None:
        try:
            self._request("/system_stats", timeout=5.0)
        except Exception as error:
            raise RuntimeError(
                f"ComfyUI is not reachable at {self.server}. Start it with:\n"
                "  cd ComfyUI && ../env_comfy/bin/python main.py --lowvram "
                "--disable-auto-launch --listen 127.0.0.1 --port 8188\n"
                f"Original error: {error}"
            ) from error

    def _stage_image(self, path: Path, size: tuple[int, int], exact: bool) -> str:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(path).convert("RGB")
        if exact:
            image = image.resize(size, Image.LANCZOS)
        else:
            image.thumbnail((max(size), max(size)), Image.LANCZOS)
        name = f"{self.output_prefix}_{uuid.uuid4().hex[:12]}.png"
        image.save(self.input_dir / name, format="PNG")
        return name

    def _build_graph(
        self,
        names: Sequence[str],
        prompt: str,
        size: tuple[int, int],
        seed: int,
        prefix: str,
    ) -> dict:
        width, height = size
        shift = math.log(self.alpha)
        graph: dict = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "flux3-image-klein-int8-convrot.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_4b_int8_convrot.safetensors",
                    "type": "flux3",
                    "device": "default",
                },
            },
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
            "10": {
                "class_type": "EmptyFlux2LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "11": {
                "class_type": "ModelSamplingFlux",
                "inputs": {
                    "model": ["1", 0],
                    "max_shift": shift,
                    "base_shift": shift,
                    "width": width,
                    "height": height,
                },
            },
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["11", 0],
                    "seed": int(seed),
                    "steps": int(self.steps),
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["10", 0],
                    "denoise": 1.0,
                },
            },
            "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
            "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": prefix}},
        }

        positive: list = ["4", 0]
        negative: list = ["5", 0]
        node_id = 20
        for name in names:
            load_id, encode_id = str(node_id), str(node_id + 1)
            pos_ref, neg_ref = str(node_id + 2), str(node_id + 3)
            graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            graph[encode_id] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": [load_id, 0], "vae": ["3", 0]},
            }
            graph[pos_ref] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": positive, "latent": [encode_id, 0]},
            }
            graph[neg_ref] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": negative, "latent": [encode_id, 0]},
            }
            positive = [pos_ref, 0]
            negative = [neg_ref, 0]
            node_id += 4
        graph["12"]["inputs"]["positive"] = positive
        graph["12"]["inputs"]["negative"] = negative
        return graph

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
        names = [self._stage_image(path, size, exact=index == 0) for index, path in enumerate(images)]
        prefix = f"{self.output_prefix}_{uuid.uuid4().hex[:10]}"
        graph = self._build_graph(names, prompt, size, seed, prefix)
        response = self._request(
            "/prompt",
            {"prompt": graph, "client_id": f"sat2aerial-{uuid.uuid4().hex[:8]}"},
            timeout=120.0,
        )
        if response.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected the graph: {json.dumps(response)[:3000]}")
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {response}")

        deadline = time.time() + self.timeout
        outputs: dict = {}
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            history = self._request(f"/history/{prompt_id}", timeout=30.0)
            if prompt_id not in history:
                continue
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution failed: {json.dumps(status)[:3000]}")
            if status.get("completed"):
                outputs = item.get("outputs", {})
                break
        else:
            raise TimeoutError(f"ComfyUI generation timed out after {self.timeout}s")

        files = []
        for node_output in outputs.values():
            files.extend(node_output.get("images", []))
        if not files:
            raise RuntimeError(f"ComfyUI returned no images: {outputs}")
        first = files[0]
        query = urllib.parse.urlencode(
            {
                "filename": first["filename"],
                "subfolder": first.get("subfolder", ""),
                "type": first.get("type", "output"),
            }
        )
        blob = urllib.request.urlopen(f"{self.server}/view?{query}", timeout=120.0).read()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(blob)).convert("RGB").save(out_path, format="PNG")
        return self._finish(
            out_path,
            started,
            {
                "prompt": prompt,
                "size": list(size),
                "seed": seed,
                "style_refs": [str(item) for item in style_refs],
                "prompt_id": prompt_id,
                "comfy_file": first["filename"],
                "staged_inputs": names,
            },
        )
