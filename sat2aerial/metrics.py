from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _read(path: str | Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), low, high)


def aligned_edge_iou(
    ref_path: str | Path,
    target_path: str | Path,
    homography: np.ndarray | None,
    tolerance_px: int = 3,
) -> float | None:
    if homography is None:
        return None
    target = _read(target_path)
    origin = _read(ref_path)
    if target is None or origin is None:
        return None
    height, width = target.shape[:2]
    warped = cv2.warpPerspective(origin, homography, (width, height), flags=cv2.INTER_LINEAR)
    ref_edges = _edges(warped)
    target_edges = _edges(target)
    kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8)
    ref_dilated = cv2.dilate(ref_edges, kernel)
    target_dilated = cv2.dilate(target_edges, kernel)
    intersection = np.logical_and(ref_dilated > 0, target_dilated > 0).sum()
    union = np.logical_or(ref_dilated > 0, target_dilated > 0).sum()
    return float(intersection / union) if union else None


def color_stats(ref_path: str | Path, target_path: str | Path) -> dict[str, float | None]:
    ref = _read(ref_path)
    target = _read(target_path)
    if ref is None or target is None:
        return {"lab_mean_dist": None, "hist_corr": None}
    target = cv2.resize(target, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_dist = float(np.linalg.norm(ref_lab.mean(axis=(0, 1)) - target_lab.mean(axis=(0, 1))))
    correlations = []
    for channel in range(3):
        hist_ref = cv2.calcHist([ref_lab[..., channel].astype(np.uint8)], [0], None, [64], [0, 256])
        hist_target = cv2.calcHist([target_lab[..., channel].astype(np.uint8)], [0], None, [64], [0, 256])
        cv2.normalize(hist_ref, hist_ref)
        cv2.normalize(hist_target, hist_target)
        correlations.append(float(cv2.compareHist(hist_ref, hist_target, cv2.HISTCMP_CORREL)))
    return {"lab_mean_dist": mean_dist, "hist_corr": float(np.mean(correlations))}


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return np.outer(kernel, kernel)


def ssim(ref_path: str | Path, target_path: str | Path) -> float | None:
    ref = _read(ref_path)
    target = _read(target_path)
    if ref is None or target is None:
        return None
    target = cv2.resize(target, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float64)
    kernel = _gaussian_kernel()
    mu_x = cv2.filter2D(x, -1, kernel)
    mu_y = cv2.filter2D(y, -1, kernel)
    sigma_x = cv2.filter2D(x * x, -1, kernel) - mu_x * mu_x
    sigma_y = cv2.filter2D(y * y, -1, kernel) - mu_y * mu_y
    sigma_xy = cv2.filter2D(x * y, -1, kernel) - mu_x * mu_y
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(np.mean(numerator / denominator))


def evaluate_pair(
    ref_path: str | Path,
    target_path: str | Path,
    homography: np.ndarray | None = None,
    tolerance_px: int = 3,
) -> dict[str, float | None]:
    colors = color_stats(ref_path, target_path)
    return {
        "aligned_edge_iou": aligned_edge_iou(ref_path, target_path, homography, tolerance_px),
        "ssim": ssim(ref_path, target_path),
        **colors,
    }
