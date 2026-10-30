from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from sat2aerial.matching import draw_matches, match_images
from sat2aerial.metrics import evaluate_pair


def match_pair(
    ref: Path,
    target: Path,
    mode: str = "edges",
    max_dim: int = 1024,
    min_inliers: int = 15,
    min_ratio: float = 0.15,
    min_coverage: float = 0.03,
) -> dict:
    result, homography = match_images(
        ref,
        target,
        mode=mode,
        max_dim=max_dim,
        min_inliers=min_inliers,
        min_ratio=min_ratio,
        min_coverage=min_coverage,
    )
    fidelity = evaluate_pair(ref, target, homography)
    return {"match": result.as_dict(), "fidelity": fidelity, "success": result.success}


def save_overlay(ref: Path, target: Path, out: Path, mode: str = "edges") -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    return draw_matches(ref, target, out, mode=mode)


def save_montage(images: Iterable[Path], labels: Iterable[str], out: Path, height: int = 480) -> bool:
    panels = []
    for path, label in zip(images, labels):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return False
        scale = height / float(image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), height),
            interpolation=cv2.INTER_AREA,
        )
        bar = np.zeros((28, resized.shape[1], 3), np.uint8)
        cv2.putText(
            bar,
            label,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(np.vstack([bar, resized]))
    if not panels:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out), np.hstack(panels)))


def best_match(target: Path, refs: list[Path], **kwargs) -> tuple[Path | None, dict | None]:
    best_ref: Path | None = None
    best: dict | None = None
    for ref in refs:
        result = match_pair(ref, target, **kwargs)
        if best is None or (
            result["match"]["inliers"],
            result["match"]["coverage"],
        ) > (
            best["match"]["inliers"],
            best["match"]["coverage"],
        ):
            best_ref = ref
            best = result
    return best_ref, best
