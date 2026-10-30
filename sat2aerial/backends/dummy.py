from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .base import Backend, save_png


def _color_transfer(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    for channel in range(3):
        s_mean, s_std = src[..., channel].mean(), src[..., channel].std() + 1e-6
        r_mean, r_std = ref[..., channel].mean(), ref[..., channel].std() + 1e-6
        src[..., channel] = (src[..., channel] - s_mean) * (r_std / s_std) + r_mean
    return cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class DummyBackend(Backend):
    name = "dummy"
    model = "procedural-warp (NOT a generative model)"

    def __init__(self, seed: int = 0, perspective: float = 0.035, blur: float = 0.8):
        super().__init__(seed=seed)
        self.perspective = perspective
        self.blur = blur

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
        image = cv2.imread(str(satellite), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(satellite)
        width, height = size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        rng = np.random.default_rng(seed)
        margin = self.perspective * min(width, height)
        src = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        dst = src + rng.uniform(-margin, margin, size=(4, 2)).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        gain = rng.uniform(0.92, 1.08)
        bias = rng.uniform(-8, 8)
        warped = np.clip(warped.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        if style_refs:
            reference = cv2.imread(str(style_refs[0]), cv2.IMREAD_COLOR)
            if reference is not None:
                warped = _color_transfer(warped, cv2.resize(reference, (width, height)))
        if self.blur > 0:
            warped = cv2.GaussianBlur(warped, (3, 3), self.blur)
        noise = rng.normal(0, 2.0, warped.shape)
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        save_png(warped, out_path)
        return self._finish(
            out_path,
            started,
            {
                "prompt": prompt,
                "size": [width, height],
                "seed": seed,
                "style_refs": [str(p) for p in style_refs],
                "note": "procedural smoke-test backend; do not use for scientific results",
            },
        )
