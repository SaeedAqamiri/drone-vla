"""Conservative registration of generated references to their source maps.

These checks are necessary, not sufficient: a single projective transform
cannot detect all local hallucinations or make generated texture factual.
"""
from __future__ import annotations

from dataclasses import asdict
import cv2
import numpy as np

from .matching import match_arrays


def register_generated_to_source(
    source: np.ndarray,
    generated: np.ndarray,
    *,
    min_inliers: int = 40,
    min_coverage: float = 0.15,
    max_reproj_px: float = 2.5,
    max_shift_fraction: float = 0.05,
) -> tuple[np.ndarray | None, dict]:
    """Return a source-aligned generated BGR image and transparent diagnostics.

    Matching and thresholds are measured in generated-image pixels after
    resizing the source to the generated image's dimensions.
    """
    if source is None or generated is None or source.size == 0 or generated.size == 0:
        return None, {"accepted": False, "reason": "unreadable_image"}
    h, w = generated.shape[:2]
    source = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    outcome, matrix = match_arrays(
        generated, source, mode="gray", max_dim=max(h, w),
        min_inliers=min_inliers, min_ratio=0.2, min_coverage=min_coverage,
        ransac_px=3.0,
    )
    diag = asdict(outcome)
    diag["accepted"] = False
    if matrix is None or not outcome.success or outcome.median_reproj is None:
        diag["reason"] = "insufficient_correspondences"
        return None, diag
    if outcome.median_reproj > max_reproj_px:
        diag["reason"] = "reprojection_error"
        return None, diag
    # Check the entire image, not just the centre. Large translation, scaling,
    # perspective or rotation changes are not a trustworthy style-only edit.
    grid = np.float32([[x, y] for x in (0, w/2, w-1) for y in (0, h/2, h-1)])
    mapped = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(mapped).all():
        diag["reason"] = "nonfinite_transform"
        return None, diag
    shift = np.linalg.norm(mapped - grid, axis=1)
    diag["max_grid_shift_px"] = float(np.max(shift))
    diag["homography_generated_to_source"] = matrix.tolist()
    if diag["max_grid_shift_px"] > max_shift_fraction * max(h, w):
        diag["reason"] = "excessive_transform"
        return None, diag
    diag["accepted"] = True
    diag["reason"] = "ok"
    return cv2.warpPerspective(generated, matrix, (w, h)), diag
