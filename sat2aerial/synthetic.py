from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def _fractal(rng: np.random.Generator, size: int) -> np.ndarray:
    field = np.zeros((size, size), np.float32)
    for sigma, weight in ((1.5, 0.20), (6.0, 0.30), (24.0, 0.30), (96.0, 0.20)):
        noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
        field += weight * cv2.GaussianBlur(noise, (0, 0), sigma)
    field -= field.min()
    field /= max(field.max(), 1e-6)
    return field


def make_satellite(rng: np.random.Generator, size: int = 768) -> np.ndarray:
    texture = _fractal(rng, size)
    base = np.zeros((size, size, 3), np.float32)
    base[..., 0] = 55 + 35 * texture
    base[..., 1] = 95 + 55 * texture
    base[..., 2] = 55 + 30 * texture

    for _ in range(rng.integers(2, 5)):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        axes = (int(rng.integers(size // 10, size // 4)), int(rng.integers(size // 10, size // 4)))
        cv2.ellipse(base, center, axes, int(rng.integers(0, 180)), 0, 360, (60, 140, 70), -1)

    step = int(size / rng.integers(4, 7))
    offset_x = int(rng.integers(0, step))
    offset_y = int(rng.integers(0, step))
    roads = []
    for axis in (0, 1):
        start = offset_x if axis == 0 else offset_y
        for position in range(start, size, step):
            width = int(rng.integers(7, 15))
            color = int(rng.integers(95, 130))
            if axis == 0:
                cv2.line(base, (position, 0), (position, size), (color, color, color), width)
                roads.append(("v", position, width))
            else:
                cv2.line(base, (0, position), (size, position), (color, color, color), width)
                roads.append(("h", position, width))

    for axis, position, width in roads:
        for other in range(offset_y if axis == "v" else offset_x, size, step):
            cell = step - width
            if cell < 30:
                continue
            for _ in range(int(rng.integers(1, 3))):
                bw = int(rng.integers(max(12, cell // 4), max(16, cell - 6)))
                bh = int(rng.integers(max(12, cell // 4), max(16, cell - 6)))
                if axis == "v":
                    x = position + width // 2 + int(rng.integers(3, max(4, cell - bw)))
                    y = other + int(rng.integers(3, max(4, cell - bh)))
                else:
                    x = other + int(rng.integers(3, max(4, cell - bw)))
                    y = position + width // 2 + int(rng.integers(3, max(4, cell - bh)))
                if x + bw >= size or y + bh >= size:
                    continue
                shadow = int(rng.integers(6, 14))
                cv2.rectangle(base, (x + shadow, y + shadow), (x + bw + shadow, y + bh + shadow), (35, 35, 35), -1)
                tone = int(rng.integers(130, 205))
                cv2.rectangle(base, (x, y), (x + bw, y + bh), (tone, tone, min(255, tone + 8)), -1)
                roof = base[y + 2 : y + bh - 2, x + 2 : x + bw - 2]
                if roof.size:
                    roof += rng.normal(0, 6, roof.shape).astype(np.float32)

    water_x = int(rng.integers(size // 2, size - 60))
    water_y = int(rng.integers(size // 2, size - 60))
    polygon = np.array(
        [
            [water_x, water_y],
            [min(size - 1, water_x + int(rng.integers(120, 260))), water_y],
            [min(size - 1, water_x + int(rng.integers(120, 260))), min(size - 1, water_y + int(rng.integers(90, 200)))],
            [water_x, min(size - 1, water_y + int(rng.integers(90, 200)))],
        ],
        np.int32,
    )
    cv2.fillPoly(base, [polygon], (150, 105, 60))

    base += rng.normal(0, 5, base.shape).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def make_aerial(satellite: np.ndarray, rng: np.random.Generator, strength: float = 0.04) -> np.ndarray:
    height, width = satellite.shape[:2]
    margin = strength * min(height, width)
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst = src + rng.uniform(-margin, margin, size=(4, 2)).astype(np.float32)
    dst[2] += rng.uniform(-margin, margin, 2) * 1.6
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        satellite, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    gamma = float(rng.uniform(0.85, 1.15))
    lookup = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    warped = cv2.LUT(warped, lookup)
    gains = rng.uniform(0.9, 1.1, size=3).astype(np.float32)
    warped = np.clip(warped.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    warped = cv2.GaussianBlur(warped, (3, 3), float(rng.uniform(0.4, 1.1)))
    warped = np.clip(warped.astype(np.float32) + rng.normal(0, 3.0, warped.shape), 0, 255).astype(np.uint8)
    return warped


def generate_demo(root: str | Path, sites: int = 6, size: int = 768, seed: int = 0) -> Path:
    root = Path(root)
    pairs_dir = root / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(sites):
        rng = np.random.default_rng(seed + index)
        site_id = f"site_{index:02d}"
        site_dir = pairs_dir / site_id
        site_dir.mkdir(parents=True, exist_ok=True)
        satellite = make_satellite(rng, size=size)
        satellite_path = site_dir / "satellite.png"
        cv2.imwrite(str(satellite_path), satellite)
        aerial_paths = []
        for frame in range(2):
            strength = 0.03 if frame == 0 else 0.05
            aerial = make_aerial(satellite, rng, strength=strength)
            aerial_path = site_dir / f"aerial_{frame:02d}.png"
            cv2.imwrite(str(aerial_path), aerial)
            aerial_paths.append(aerial_path)
        rows.append(
            {
                "site": site_id,
                "satellite": str(satellite_path.relative_to(root)),
                "aerial": ";".join(str(path.relative_to(root)) for path in aerial_paths),
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["site", "satellite", "aerial"])
        writer.writeheader()
        writer.writerows(rows)
    return root
