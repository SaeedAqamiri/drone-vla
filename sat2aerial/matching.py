from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import MatchResult

FLANN_INDEX_KDTREE = 1


def _read(path: str | Path, max_dim: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    scale = max_dim / float(max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _prepare(image: np.ndarray, mode: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if mode == "edges":
        median = float(np.median(gray))
        low = int(max(0, 0.66 * median))
        high = int(min(255, 1.33 * median))
        gray = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), low, high)
        gray = cv2.dilate(gray, np.ones((3, 3), np.uint8), iterations=1)
    elif mode != "gray":
        raise ValueError(f"Unknown match mode {mode!r}")
    return gray


def _detector(name: str = "sift"):
    key = name.lower().strip()
    if key == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(
                nfeatures=5000, contrastThreshold=0.005, edgeThreshold=15
            )
        return cv2.ORB_create(nfeatures=5000)
    if key == "orb":
        return cv2.ORB_create(nfeatures=5000)
    if key == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"Unknown detector {name!r}")


def _matcher(name: str = "sift"):
    if name.lower().strip() in {"orb", "akaze"}:
        return cv2.BFMatcher(cv2.NORM_HAMMING)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    height, width = shape
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / float(width * height))


def match_images(
    ref_path: str | Path,
    target_path: str | Path,
    mode: str = "edges",
    max_dim: int = 1024,
    ratio: float = 0.75,
    ransac_px: float = 5.0,
    min_inliers: int = 15,
    min_ratio: float = 0.15,
    min_coverage: float = 0.03,
    detector: str = "sift",
) -> tuple[MatchResult, np.ndarray | None]:
    ref_image = _read(ref_path, max_dim)
    target_image = _read(target_path, max_dim)
    empty = MatchResult(
        ref=str(ref_path),
        target=str(target_path),
        mode=mode,
        keypoints_ref=0,
        keypoints_target=0,
        good_matches=0,
        inliers=0,
        inlier_ratio=0.0,
        median_reproj=None,
        scale=None,
        rotation_deg=None,
        coverage=0.0,
        success=False,
        detector=detector,
    )
    if ref_image is None or target_image is None:
        return empty, None

    ref_gray = _prepare(ref_image, mode)
    target_gray = _prepare(target_image, mode)
    extractor = _detector(detector)
    kp_ref, des_ref = extractor.detectAndCompute(ref_gray, None)
    kp_target, des_target = extractor.detectAndCompute(target_gray, None)
    if des_ref is None or des_target is None or len(kp_ref) < 4 or len(kp_target) < 4:
        empty.keypoints_ref = len(kp_ref)
        empty.keypoints_target = len(kp_target)
        return empty, None

    knn = _matcher(detector).knnMatch(des_ref, des_target, k=2)
    good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < 4:
        empty.keypoints_ref = len(kp_ref)
        empty.keypoints_target = len(kp_target)
        empty.good_matches = len(good)
        return empty, None

    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_target[m.trainIdx].pt for m in good])
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px, maxIters=5000, confidence=0.999)
    if homography is None or mask is None:
        empty.keypoints_ref = len(kp_ref)
        empty.keypoints_target = len(kp_target)
        empty.good_matches = len(good)
        return empty, None

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / float(len(good)) if good else 0.0
    inlier_src = src[inlier_mask]
    inlier_dst = dst[inlier_mask]

    reproj = None
    scale = None
    rotation = None
    if inliers >= 4:
        projected = cv2.perspectiveTransform(inlier_src.reshape(-1, 1, 2), homography).reshape(-1, 2)
        errors = np.linalg.norm(projected - inlier_dst, axis=1)
        reproj = float(np.median(errors))
        affine, _ = cv2.estimateAffinePartial2D(
            inlier_src,
            inlier_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_px,
        )
        if affine is not None:
            scale = float(np.hypot(affine[0, 0], affine[0, 1]))
            rotation = float(np.degrees(np.arctan2(affine[0, 1], affine[0, 0])))

    coverage = _coverage(inlier_dst, target_gray.shape[:2])
    result = MatchResult(
        ref=str(ref_path),
        target=str(target_path),
        mode=mode,
        keypoints_ref=len(kp_ref),
        keypoints_target=len(kp_target),
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        median_reproj=reproj,
        scale=scale,
        rotation_deg=rotation,
        coverage=coverage,
        success=inliers >= min_inliers and inlier_ratio >= min_ratio and coverage >= min_coverage,
        detector=detector,
    )
    return result, homography


def warp_to_target(ref_path: str | Path, target_shape: tuple[int, int], homography: np.ndarray) -> np.ndarray | None:
    image = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = target_shape
    return cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_LINEAR)


def draw_matches(
    ref_path: str | Path,
    target_path: str | Path,
    out_path: str | Path,
    mode: str = "edges",
    max_dim: int = 1024,
    ratio: float = 0.75,
) -> bool:
    ref_image = _read(ref_path, max_dim)
    target_image = _read(target_path, max_dim)
    if ref_image is None or target_image is None:
        return False
    ref_gray = _prepare(ref_image, mode)
    target_gray = _prepare(target_image, mode)
    detector = _detector()
    kp_ref, des_ref = detector.detectAndCompute(ref_gray, None)
    kp_target, des_target = detector.detectAndCompute(target_gray, None)
    if des_ref is None or des_target is None:
        return False
    knn = _matcher().knnMatch(des_ref, des_target, k=2)
    good = [pair[0] for pair in knn if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if not good:
        return False
    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
    dst = np.float32([kp_target[m.trainIdx].pt for m in good])
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0, maxIters=5000, confidence=0.999)
    if homography is None or mask is None:
        return False
    inlier_mask = mask.ravel().astype(int)
    canvas = cv2.drawMatches(
        ref_image,
        kp_ref,
        target_image,
        kp_target,
        good,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        matchesMask=inlier_mask.tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), canvas))
