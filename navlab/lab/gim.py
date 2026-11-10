"""Optional, pretrained GIM-LoFTR adapter for offline research.

The external source and checkpoint are not bundled. Uses the original GIM
RGB architecture rather than loading its weights into a different LoFTR.
"""
from __future__ import annotations

import os
import hashlib
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np

from .deep import _homography_outcome, _read_rgb
from .types import MatchOutcome


def _lower(value):
    return {str(key).lower(): _lower(item) for key, item in value.items()} if isinstance(value, dict) else value


def _prepare(image, torch, device):
    height, width = image.shape[:2]
    h8, w8 = height // 8 * 8, width // 8 * 8
    if min(h8, w8) < 16:
        raise ValueError("image too small for GIM")
    rgb = cv2.cvtColor(cv2.resize(image, (w8, h8)), cv2.COLOR_BGR2RGB)
    color = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None].to(device) / 255
    return color, (width / w8, height / h8)


class GIMLoFTRMatcher:
    def __init__(self, *, max_dim=768, min_inliers=15, min_ratio=.15,
                 min_coverage=.03, ransac_px=5.0, confidence=.5):
        import torch
        import kornia.utils

        root = Path(os.environ.get("NAVLAB_GIM_ROOT", Path(__file__).resolve().parents[2] / "models/external/gim")).resolve()
        ckpt = Path(os.environ.get("NAVLAB_GIM_LOFTR_CHECKPOINT", root / "weights/gim_loftr_50h.ckpt"))
        if not (root / "networks/loftr/loftr.py").is_file() or not ckpt.is_file():
            raise RuntimeError("GIM source or checkpoint missing; set NAVLAB_GIM_ROOT and NAVLAB_GIM_LOFTR_CHECKPOINT")
        # Upstream imports this module path from older Kornia releases. Modern
        # Kornia exports the same function from kornia.utils.
        try:
            import kornia.utils.grid  # noqa: F401
        except ImportError:
            shim = types.ModuleType("kornia.utils.grid")
            shim.create_meshgrid = kornia.utils.create_meshgrid
            sys.modules["kornia.utils.grid"] = shim
        sys.path.insert(0, str(root))
        from networks.loftr.loftr import LoFTR
        from networks.loftr.config import get_cfg_defaults
        self.torch, self.device = torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LoFTR(_lower(get_cfg_defaults())["loftr"])
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state.get("state_dict", state))
        self.model = self.model.to(self.device).eval()
        self.max_dim, self.min_inliers, self.min_ratio = int(max_dim), min_inliers, min_ratio
        self.min_coverage, self.ransac_px, self.confidence = min_coverage, ransac_px, confidence

    def match(self, ref, query):
        started = time.time()
        names = tuple("<array>" if isinstance(i, np.ndarray) else str(i) for i in (ref, query))
        try:
            a, b = _read_rgb(ref, self.max_dim), _read_rgb(query, self.max_dim)
            if a is None or b is None:
                raise ValueError("image read failed")
            ref_img, query_img = a[0], b[0]
            color0, scale0 = _prepare(ref_img, self.torch, self.device)
            color1, scale1 = _prepare(query_img, self.torch, self.device)
            data = {"color0": color0, "color1": color1,
                    "image0": color0.mean(1, keepdim=True), "image1": color1.mean(1, keepdim=True)}
            with self.torch.inference_mode():
                self.model(data)
            keep = data["mconf"] >= self.confidence
            src = data["mkpts0_f"][keep].detach().cpu().numpy() * np.asarray(scale0)
            dst = data["mkpts1_f"][keep].detach().cpu().numpy() * np.asarray(scale1)
            return _homography_outcome(*names, "gim_loftr", src, dst, ref_img.shape[:2], query_img.shape[:2],
                                       started, self.ransac_px, self.min_inliers, self.min_ratio, self.min_coverage)
        except Exception as error:
            return MatchOutcome(*names, "gim_loftr", 0, 0, 0, None, None, None,
                                (time.time() - started) * 1000, False,
                                error=f"{type(error).__name__}: {error}", execution_error=True)


class GIMDKMMatcher:
    """GIM-DKM using the author's 100-hour checkpoint and demo geometry."""

    def __init__(self, *, max_dim=768, min_inliers=15, min_ratio=.15,
                 min_coverage=.03, ransac_px=5.0):
        import torch
        root = Path(os.environ.get("NAVLAB_GIM_ROOT", Path(__file__).resolve().parents[2] / "models/external/gim")).resolve()
        ckpt = Path(os.environ.get("NAVLAB_GIM_DKM_CHECKPOINT", root / "weights/gim_dkm_100h.ckpt"))
        if not (root / "networks/dkm/models/model_zoo/DKMv3.py").is_file() or not ckpt.is_file():
            raise RuntimeError("GIM-DKM source or checkpoint missing")
        sys.path.insert(0, str(root))
        from networks.dkm.models.model_zoo.DKMv3 import DKMv3
        self.torch, self.device = torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DKMv3(weights=None, h=672, w=896)
        # The author's Lightning checkpoint contains a YACS object and cannot
        # be read by torch's weights-only unpickler. Pin its official published
        # SHA-256 before allowing the trusted upstream pickle to be loaded.
        digest = hashlib.sha256()
        with ckpt.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != "bbe056bcf1b401e8dbc0ea983e69ed168eb0a330dcee35f3d9cf53048bf69ab8":
            raise RuntimeError("GIM-DKM checkpoint SHA-256 does not match the pinned official checkpoint")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        state = {k.removeprefix("model."): v for k, v in state.items() if "encoder.net.fc" not in k}
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self.max_dim, self.min_inliers, self.min_ratio = int(max_dim), min_inliers, min_ratio
        self.min_coverage, self.ransac_px = min_coverage, ransac_px

    def match(self, ref, query):
        outcome, _, _ = self.match_with_points(ref, query)
        return outcome

    def match_with_points(self, ref, query):
        """Return the standard homography outcome and sampled correspondences.

        Points use the internally resized reference/query pixel coordinates;
        they are returned even if the planar homography gate rejects the pair.
        """
        started = time.time()
        names = tuple("<array>" if isinstance(i, np.ndarray) else str(i) for i in (ref, query))
        try:
            a, b = _read_rgb(ref, self.max_dim), _read_rgb(query, self.max_dim)
            if a is None or b is None:
                raise ValueError("image read failed")
            ref_img, query_img = a[0], b[0]
            color0, scale0 = _prepare(ref_img, self.torch, self.device)
            color1, scale1 = _prepare(query_img, self.torch, self.device)
            def pad(t):
                h, w = t.shape[-2:]
                nw, nh = max(w, int(h * 896 / 672)), max(h, int(w * 672 / 896))
                left, top = (nw-w)//2, (nh-h)//2
                return self.torch.nn.functional.pad(t, (left, nw-w-left, top, nh-h-top)), (left, top, w, h)
            pad0, bounds0 = pad(color0)
            pad1, bounds1 = pad(color1)
            with self.torch.inference_mode():
                self.torch.manual_seed(0)
                warp, certainty = self.model.match(pad0, pad1)
                samples, _ = self.model.sample(warp, certainty, 5000)
                p0, p1 = self.model.to_pixel_coordinates(samples, *pad0.shape[-2:], *pad1.shape[-2:])
            p0, p1 = p0.cpu().numpy(), p1.cpu().numpy()
            p0 -= np.asarray(bounds0[:2]); p1 -= np.asarray(bounds1[:2])
            keep = ((p0[:, 0] > 0) & (p0[:, 1] > 0) & (p0[:, 0] <= bounds0[2]-1) & (p0[:, 1] <= bounds0[3]-1)
                    & (p1[:, 0] > 0) & (p1[:, 1] > 0) & (p1[:, 0] <= bounds1[2]-1) & (p1[:, 1] <= bounds1[3]-1))
            src, dst = p0[keep] * np.asarray(scale0), p1[keep] * np.asarray(scale1)
            outcome = _homography_outcome(*names, "gim_dkm", src, dst, ref_img.shape[:2], query_img.shape[:2],
                                          started, self.ransac_px, self.min_inliers, self.min_ratio, self.min_coverage)
            return outcome, src, dst
        except Exception as error:
            outcome = MatchOutcome(*names, "gim_dkm", 0, 0, 0, None, None, None,
                                   (time.time() - started) * 1000, False,
                                   error=f"{type(error).__name__}: {error}", execution_error=True)
            return outcome, None, None
