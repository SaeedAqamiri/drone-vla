from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .types import MatchOutcome

try:
    import torch
    import kornia.feature as KF

    KORNIA_AVAILABLE = True
except Exception:
    torch = None
    KF = None
    KORNIA_AVAILABLE = False


def _device():
    if torch is not None and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_rgb(path: str | Path, max_dim: int) -> tuple[np.ndarray, float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    factor = min(1.0, max_dim / float(max(height, width)))
    if factor < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
            interpolation=cv2.INTER_AREA,
        )
    return image, factor


def _tensor(image: np.ndarray, device) -> "torch.Tensor":
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float() / 255.0
    return tensor.permute(2, 0, 1)[None].to(device)


def _homography_outcome(
    ref: str,
    query: str,
    detector: str,
    src: np.ndarray,
    dst: np.ndarray,
    ref_shape: tuple[int, int],
    query_shape: tuple[int, int],
    started: float,
    ransac_px: float,
    min_inliers: int,
    min_ratio: float,
    min_coverage: float,
) -> MatchOutcome:
    if len(src) < 4:
        return MatchOutcome(
            ref=ref,
            query=query,
            detector=detector,
            inliers=0,
            inlier_ratio=0.0,
            coverage=0.0,
            reproj=None,
            scale=None,
            rotation=None,
            latency_ms=(time.time() - started) * 1000.0,
            success=False,
        )
    src = np.float32(src)
    dst = np.float32(dst)
    homography, mask = cv2.findHomography(
        src, dst, cv2.RANSAC, ransac_px, maxIters=5000, confidence=0.999
    )
    if homography is None or mask is None:
        return MatchOutcome(
            ref=ref,
            query=query,
            detector=detector,
            inliers=0,
            inlier_ratio=0.0,
            coverage=0.0,
            reproj=None,
            scale=None,
            rotation=None,
            latency_ms=(time.time() - started) * 1000.0,
            success=False,
        )
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    ratio = inliers / float(len(src))
    inlier_dst = dst[inlier_mask]
    coverage = 0.0
    if len(inlier_dst) >= 3:
        hull = cv2.convexHull(inlier_dst)
        coverage = float(
            cv2.contourArea(hull) / float(query_shape[0] * query_shape[1])
        )
    reproj = None
    scale = None
    rotation = None
    if inliers >= 4:
        projected = cv2.perspectiveTransform(
            src[inlier_mask].reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        reproj = float(np.median(np.linalg.norm(projected - inlier_dst, axis=1)))
        affine, _ = cv2.estimateAffinePartial2D(
            src[inlier_mask],
            inlier_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_px,
        )
        if affine is not None:
            scale = float(np.hypot(affine[0, 0], affine[0, 1]))
            rotation = float(
                np.degrees(np.arctan2(affine[0, 1], affine[0, 0]))
            )
    success = (
        inliers >= min_inliers
        and ratio >= min_ratio
        and coverage >= min_coverage
    )
    return MatchOutcome(
        ref=ref,
        query=query,
        detector=detector,
        inliers=inliers,
        inlier_ratio=ratio,
        coverage=coverage,
        reproj=reproj,
        scale=scale,
        rotation=rotation,
        latency_ms=(time.time() - started) * 1000.0,
        success=success,
        homography=homography,
    )


class DeepMatcher:
    def __init__(
        self,
        kind: str = "loftr",
        *,
        max_dim: int = 768,
        min_inliers: int = 15,
        min_ratio: float = 0.15,
        min_coverage: float = 0.03,
        ransac_px: float = 5.0,
        confidence: float = 0.5,
        num_features: int = 2048,
    ):
        if not KORNIA_AVAILABLE:
            raise RuntimeError(
                "Deep matchers need kornia. Install with: "
                "uv pip install --python env_flight/bin/python kornia"
            )
        self.kind = kind.lower().strip()
        self.max_dim = int(max_dim)
        self.min_inliers = int(min_inliers)
        self.min_ratio = float(min_ratio)
        self.min_coverage = float(min_coverage)
        self.ransac_px = float(ransac_px)
        self.confidence = float(confidence)
        self.num_features = int(num_features)
        self.device = _device()
        self._model = None
        self._aux = None
        if self.kind == "loftr":
            self._model = KF.LoFTR(pretrained="outdoor").to(self.device).eval()
        elif self.kind in {"lightglue", "superpoint_lightglue", "disk_lightglue"}:
            self._aux = KF.DISK.from_pretrained("depth").to(self.device).eval()
            self._model = KF.LightGlue("disk").to(self.device).eval()
        else:
            raise ValueError(f"Unknown deep matcher {kind!r}")

    def match(self, ref: str | Path, query: str | Path) -> MatchOutcome:
        started = time.time()
        ref_image = _read_rgb(ref, self.max_dim)
        query_image = _read_rgb(query, self.max_dim)
        if ref_image is None or query_image is None:
            return MatchOutcome(
                ref=str(ref),
                query=str(query),
                detector=self.kind,
                inliers=0,
                inlier_ratio=0.0,
                coverage=0.0,
                reproj=None,
                scale=None,
                rotation=None,
                latency_ms=(time.time() - started) * 1000.0,
                success=False,
                error="image read failed",
            )
        ref_image_array, _ = ref_image
        query_image_array, _ = query_image
        try:
            with torch.inference_mode():
                if self.kind == "loftr":
                    ref_tensor = _tensor(ref_image_array, self.device).mean(
                        dim=1, keepdim=True
                    )
                    query_tensor = _tensor(query_image_array, self.device).mean(
                        dim=1, keepdim=True
                    )
                    output = self._model(
                        {"image0": ref_tensor, "image1": query_tensor}
                    )
                    kp_ref = output["keypoints0"].detach().cpu().numpy()
                    kp_query = output["keypoints1"].detach().cpu().numpy()
                    confidence = output.get("confidence")
                    if confidence is not None:
                        keep = (
                            confidence.detach().cpu().numpy() >= self.confidence
                        )
                        kp_ref = kp_ref[keep]
                        kp_query = kp_query[keep]
                    src, dst = kp_ref, kp_query
                else:
                    ref_tensor = _tensor(ref_image_array, self.device)
                    query_tensor = _tensor(query_image_array, self.device)
                    features_ref = self._aux(
                        ref_tensor,
                        self.num_features,
                        pad_if_not_divisible=True,
                    )[0]
                    features_query = self._aux(
                        query_tensor,
                        self.num_features,
                        pad_if_not_divisible=True,
                    )[0]
                    image0 = {
                        "keypoints": features_ref.keypoints[None],
                        "descriptors": features_ref.descriptors[None],
                        "image_size": torch.tensor(
                            ref_tensor.shape[-2:][::-1], device=self.device
                        ).view(1, 2),
                    }
                    image1 = {
                        "keypoints": features_query.keypoints[None],
                        "descriptors": features_query.descriptors[None],
                        "image_size": torch.tensor(
                            query_tensor.shape[-2:][::-1], device=self.device
                        ).view(1, 2),
                    }
                    output = self._model({"image0": image0, "image1": image1})
                    matches = output["matches"][0]
                    valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
                    matches = matches[valid]
                    kp_ref = features_ref.keypoints
                    kp_query = features_query.keypoints
                    src = kp_ref[matches[:, 0]].detach().cpu().numpy()
                    dst = kp_query[matches[:, 1]].detach().cpu().numpy()
        except Exception as error:
            return MatchOutcome(
                ref=str(ref),
                query=str(query),
                detector=self.kind,
                inliers=0,
                inlier_ratio=0.0,
                coverage=0.0,
                reproj=None,
                scale=None,
                rotation=None,
                latency_ms=(time.time() - started) * 1000.0,
                success=False,
                error=f"{type(error).__name__}: {error}",
            )
        return _homography_outcome(
            str(ref),
            str(query),
            self.kind,
            src,
            dst,
            ref_image_array.shape[:2],
            query_image_array.shape[:2],
            started,
            self.ransac_px,
            self.min_inliers,
            self.min_ratio,
            self.min_coverage,
        )
