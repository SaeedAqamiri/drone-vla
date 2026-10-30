from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from PIL import Image

from .base import Backend


class Flux3LocalBackend(Backend):
    name = "flux3"
    model = "black-forest-labs/flux-3-image-klein"

    def __init__(
        self,
        model_id: str = "black-forest-labs/flux-3-image-klein",
        variant: str = "fp8r",
        device: str = "cuda",
        steps: int = 16,
        cpu_offload: bool = True,
        seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.model_id = model_id
        self.variant = variant
        self.device = device
        self.steps = steps
        self.cpu_offload = cpu_offload
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import ComponentsManager, ModularPipeline
        except Exception as error:
            raise RuntimeError(
                "Local FLUX 3 backend needs a recent diffusers + transformers + torch.\n"
                "Install with: uv pip install -U diffusers transformers accelerate torch\n"
                f"Original error: {error}"
            ) from error

        try:
            kwargs = {}
            if self.cpu_offload:
                manager = ComponentsManager()
                manager.enable_auto_cpu_offload(device=self.device)
                kwargs["components_manager"] = manager
            pipe = ModularPipeline.from_pretrained(self.model_id, **kwargs)
            pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
            if self.variant and self.variant != "bf16":
                from diffusers import BFLQuantizationConfig, Flux3Transformer2DModel

                url = (
                    f"https://huggingface.co/{self.model_id}/blob/main/"
                    f"variants/{self.variant}/model.safetensors"
                )
                transformer = Flux3Transformer2DModel.from_single_file(
                    url,
                    config=self.model_id,
                    subfolder="transformer",
                    quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
                    dtype=torch.bfloat16,
                )
                pipe.update_components(transformer=transformer)
            if not self.cpu_offload:
                pipe.to(self.device)
            self._pipe = pipe
        except Exception as error:
            message = str(error)
            if "gated" in message.lower() or "401" in message or "403" in message:
                raise RuntimeError(
                    f"{self.model_id} is gated. Log in with `hf auth login`, accept the "
                    "FLUX Kommunity License on the model page, then retry.\n"
                    f"Original error: {message}"
                ) from error
            raise
        return self._pipe

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
        import torch

        pipe = self._load()
        images = [Image.open(path).convert("RGB") for path in (satellite, *style_refs)]
        width, height = size
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        result = pipe(
            prompt=prompt,
            image=images,
            width=int(width),
            height=int(height),
            num_inference_steps=int(self.steps),
            generator=generator,
            output="images",
        )
        image = result[0] if isinstance(result, (list, tuple)) else result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self._finish(
            out_path,
            started,
            {
                "prompt": prompt,
                "size": [width, height],
                "seed": seed,
                "style_refs": [str(p) for p in style_refs],
                "variant": self.variant,
                "cpu_offload": self.cpu_offload,
            },
        )
