"""Optional adapter for the external REDI-Match dense matcher.

The upstream checkout and weights are kept under the ignored
``models/external/REDI-Match`` tree. Set ``NAVLAB_REDIMATCH_CHECKPOINT`` to the
downloaded outdoor checkpoint before selecting ``matcher='redi'``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .deep import _homography_outcome, _read_rgb
from .types import MatchOutcome


def _as_pil(image: str | Path | np.ndarray, max_dim: int) -> Image.Image | None:
    # Match the image scale used by the other lab matchers. The returned H is
    # therefore in resized pixel coordinates, which is what LabRunner expects.
    read = _read_rgb(image, max_dim)
    if read is None:
        return None
    image_bgr, _factor = read
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def _sample_correspondences(
    model,
    warp,
    certainty,
    ref_shape: tuple[int, int],
    query_shape: tuple[int, int],
    num_matches: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse ``ref -> query`` points in resized-input pixel units.

    REDI's symmetric warp is laid out as A→B in the first width and B→A in
    the second. Passing (ref, query) to ``model.match`` means the first half is
    directly the ref→query homography expected by ``_homography_outcome``.
    Its coordinates are normalized, so use each resized input's own shape
    when converting them to pixels; no extra C4 rotation is applied here as
    REDI already maps the correspondences back into input-image coordinates.
    """
    if warp.ndim != 4 or certainty.ndim != 3 or warp.shape[0] != 1:
        raise ValueError(
            f"unexpected REDI warp/certainty shapes: {tuple(warp.shape)}, "
            f"{tuple(certainty.shape)}"
        )
    first_width = warp.shape[2] // 2
    sampled, _ = model.sample(
        warp[0, :, :first_width, :],
        certainty[0, :, :first_width],
        num_matches,
    )
    h_ref, w_ref = ref_shape
    h_query, w_query = query_shape
    ref_points, query_points = model.to_pixel_coordinates(
        sampled, h_ref, w_ref, h_query, w_query
    )
    return (
        ref_points.detach().cpu().numpy().astype(np.float32),
        query_points.detach().cpu().numpy().astype(np.float32),
    )


class REDIMatcher:
    """REDI-Match adapter that returns the lab's standard gated homography."""

    def __init__(
        self,
        *,
        max_dim: int = 768,
        min_inliers: int = 15,
        min_ratio: float = 0.15,
        min_coverage: float = 0.03,
        ransac_px: float = 5.0,
    ):
        self.max_dim = int(max_dim)  # retained for matcher-interface parity
        self.min_inliers = int(min_inliers)
        self.min_ratio = float(min_ratio)
        self.min_coverage = float(min_coverage)
        self.ransac_px = float(ransac_px)

        try:
            import torch
        except ImportError as error:
            raise RuntimeError("REDI-Match requires PyTorch in the active Python environment") from error

        code_root = Path(
            os.environ.get(
                "NAVLAB_REDIMATCH_ROOT",
                Path(__file__).resolve().parents[2] / "models" / "external" / "REDI-Match",
            )
        ).expanduser().resolve()
        if not (code_root / "redimatch" / "models" / "model_builder.py").is_file():
            raise RuntimeError(
                "REDI-Match source is missing; clone the official repository into "
                f"{code_root} or set NAVLAB_REDIMATCH_ROOT"
            )
        sys.path.insert(0, str(code_root))
        try:
            from redimatch.models.model_builder import (
                get_model,
                install_gp16_cholesky_solver,
            )
        except ImportError as error:
            raise RuntimeError(
                "REDI-Match core dependencies are unavailable; install its PyTorch, "
                "torchvision, numpy, einops, Pillow, Kornia and OpenCV requirements"
            ) from error

        checkpoint = os.environ.get("NAVLAB_REDIMATCH_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError(
                "Set NAVLAB_REDIMATCH_CHECKPOINT to the outdoor.pth checkpoint"
            )
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise RuntimeError(f"REDI-Match checkpoint does not exist: {checkpoint_path}")

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(
            pretrained_backbone=True,
            resolution="medium",
            model_type="gp",
            freeze_encoder=True,
            exported_weights_path=None,
            symmetric=True,
            upsample_preds=False,
            attenuate_cert=False,
        ).to(self.device)
        install_gp16_cholesky_solver(self.model)
        checkpoint_data = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint_data and isinstance(checkpoint_data[key], dict):
                checkpoint_data = checkpoint_data[key]
                break
        state = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in checkpoint_data.items()
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"REDI checkpoint mismatch: {len(missing)} missing and "
                f"{len(unexpected)} unexpected keys"
            )
        self.model.eval()

        decoder = self.model.decoder
        decoder.c4_rotation_matching = True
        decoder.c4_perm_reverse = True
        decoder.c4_symmetric_reverse = "align_to_query"
        decoder.c4_hybrid = True

    def match(
        self,
        ref: str | Path | np.ndarray,
        query: str | Path | np.ndarray,
    ) -> MatchOutcome:
        started = time.time()
        ref_name = "<array>" if isinstance(ref, np.ndarray) else str(ref)
        query_name = "<array>" if isinstance(query, np.ndarray) else str(query)
        ref_image = _as_pil(ref, self.max_dim)
        query_image = _as_pil(query, self.max_dim)
        if ref_image is None or query_image is None:
            return MatchOutcome(
                ref_name, query_name, "redi", 0, 0.0, 0.0, None, None, None,
                (time.time() - started) * 1000.0, False, error="image read failed",
                execution_error=True,
            )

        ref_shape = (ref_image.height, ref_image.width)
        query_shape = (query_image.height, query_image.width)
        try:
            with self.torch.inference_mode():
                warp, certainty = self.model.match(
                    ref_image, query_image, device=self.device
                )
            src, dst = _sample_correspondences(
                self.model, warp, certainty, ref_shape, query_shape
            )
        except Exception as error:
            return MatchOutcome(
                ref_name, query_name, "redi", 0, 0.0, 0.0, None, None, None,
                (time.time() - started) * 1000.0, False,
                error=f"{type(error).__name__}: {error}",
                execution_error=True,
            )

        return _homography_outcome(
            ref_name,
            query_name,
            "redi",
            src,
            dst,
            ref_shape,
            query_shape,
            started,
            self.ransac_px,
            self.min_inliers,
            self.min_ratio,
            self.min_coverage,
        )
