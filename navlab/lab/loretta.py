"""Optional adapter for the upstream LoRetta remote-sensing matcher.

Install the upstream package separately and set NAVLAB_LORETTA_CHECKPOINT when
using locally downloaded weights. Upstream code and model have separate
licenses; the DINOv3-derived checkpoint is not covered by our repository's
license. See the upstream repository before use.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .deep import _homography_outcome, _read_rgb
from .types import MatchOutcome


class LoRettaMatcher:
    def __init__(self, *, max_dim: int = 768, min_inliers: int = 15,
                 min_ratio: float = 0.15, min_coverage: float = 0.03,
                 ransac_px: float = 5.0, confidence: float = 0.5):
        try:
            import torch
            from loretta.config import get_inference_config
            from loretta.model import LoRetta
            from loretta.inference.checkpoint import load_checkpoint, resolve_checkpoint
            from loretta.inference.image import image_to_tensor, normalized_warp_to_pixels, MODEL_SIZE
            from loretta.inference.geometry import estimate_homography
        except ImportError as error:
            raise RuntimeError("LoRetta is optional; install its official upstream package first") from error
        self.torch = torch
        self.to_tensor = image_to_tensor
        self.warp_to_pixels = normalized_warp_to_pixels
        self.model_size = MODEL_SIZE
        self.estimate_homography = estimate_homography
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = os.environ.get("NAVLAB_LORETTA_CHECKPOINT")
        self.model = LoRetta(get_inference_config(enable_amp=self.device.type == "cuda"))
        load_checkpoint(self.model, resolve_checkpoint(checkpoint))
        self.model = self.model.eval().to(self.device)
        self.max_dim = max_dim
        self.min_inliers, self.min_ratio, self.min_coverage = min_inliers, min_ratio, min_coverage
        self.ransac_px, self.confidence = ransac_px, confidence

    def match(self, ref: str | Path | np.ndarray, query: str | Path | np.ndarray) -> MatchOutcome:
        started = time.time()
        ref_name = "<array>" if isinstance(ref, np.ndarray) else str(ref)
        query_name = "<array>" if isinstance(query, np.ndarray) else str(query)
        ref_read, query_read = _read_rgb(ref, self.max_dim), _read_rgb(query, self.max_dim)
        if ref_read is None or query_read is None:
            return MatchOutcome(ref_name, query_name, "loretta", 0, 0, 0, None, None, None,
                                (time.time()-started)*1000, False, error="image read failed",
                                execution_error=True)
        ref_img, _ = ref_read
        query_img, _ = query_read
        execution_error = False
        try:
            ref_rgb = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            query_rgb = Image.fromarray(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
            ref_tensor, _ = self.to_tensor(ref_rgb, self.device)
            query_tensor, _ = self.to_tensor(query_rgb, self.device)
            batch = {"reference_image": ref_tensor, "sensed_image": query_tensor}
            with self.torch.inference_mode():
                self.model(batch)
            warp = batch["warp_1"][0].float().cpu().permute(1, 2, 0).numpy()
            matchability = batch["matchability_1"][0, 0].float().cpu().numpy()
            if not np.isfinite(warp).all() or not np.isfinite(matchability).all():
                raise ValueError("nonfinite model output")
            # Pre-sample deterministically without mutating numpy's global RNG.
            # The upstream helper skips its random sampling at <=1000 points.
            indices = np.flatnonzero(matchability.reshape(-1) >= self.confidence)
            if indices.size > 1000:
                keep = np.random.default_rng(0).choice(indices, 1000, replace=False)
                filtered = np.full_like(matchability, -1.0)
                filtered.reshape(-1)[keep] = matchability.reshape(-1)[keep]
                matchability = filtered
            geometry = self.estimate_homography(
                self.warp_to_pixels(warp, self.model_size), matchability,
                reference_size=(ref_img.shape[1], ref_img.shape[0]),
                sensed_size=(query_img.shape[1], query_img.shape[0]),
                matchability_threshold=self.confidence, max_keypoints=1000,
                ransac_threshold=self.ransac_px)
            if geometry.get("success"):
                return _homography_outcome(ref_name, query_name, "loretta",
                    geometry["reference_points"], geometry["sensed_points"],
                    ref_img.shape[:2], query_img.shape[:2], started, self.ransac_px,
                    self.min_inliers, self.min_ratio, self.min_coverage)
            error = geometry.get("reason", "geometry failed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            execution_error = True
        return MatchOutcome(ref_name, query_name, "loretta", 0, 0, 0, None, None, None,
                            (time.time()-started)*1000, False, error=error,
                            execution_error=execution_error)
