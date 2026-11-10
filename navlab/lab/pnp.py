"""Experimental 2D-to-3D camera pose from pre-established correspondences.

Coordinates must share one metric map frame. A successful numerical solve is
not a calibrated confidence bound or proof that the correspondences are true.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PoseEstimate:
    success: bool
    reason: str
    rotation_world_to_camera: np.ndarray | None = None
    translation_world_to_camera: np.ndarray | None = None
    camera_center_world: np.ndarray | None = None
    inliers: int = 0
    inlier_ratio: float = 0.0
    query_coverage: float = 0.0
    median_reprojection_px: float | None = None


def estimate_pose(world_points, image_points, intrinsics, image_shape, *, reproj_px=6.0,
                  min_inliers=12, min_inlier_ratio=.2, min_coverage=.03) -> PoseEstimate:
    world = np.asarray(world_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or image.shape != (len(world), 2):
        raise ValueError("expected N x 3 world points and N x 2 image points")
    if (camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0
            or camera[1, 1] <= 0 or abs(np.linalg.det(camera)) < 1e-12):
        raise ValueError("invalid camera intrinsics")
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError("invalid image shape")
    if (not np.isfinite(reproj_px) or reproj_px <= 0 or min_inliers < 6
            or not 0 <= min_inlier_ratio <= 1 or not 0 <= min_coverage <= 1):
        raise ValueError("invalid PnP acceptance thresholds")
    valid = np.isfinite(world).all(axis=1) & np.isfinite(image).all(axis=1)
    world, image = world[valid], image[valid]
    if len(world) < max(6, min_inliers):
        return PoseEstimate(False, "too_few_correspondences")
    cv2.setRNGSeed(0)
    try:
        success, rvec, tvec, indices = cv2.solvePnPRansac(
            world, image, camera, None, iterationsCount=10000,
            reprojectionError=float(reproj_px), confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or indices is None:
            return PoseEstimate(False, "pnp_failed")
        indices = indices.ravel()
        count = len(indices)
        ratio = count / len(world)
        hull = cv2.convexHull(image[indices].astype(np.float32)) if count >= 3 else None
        coverage = cv2.contourArea(hull) / float(image_shape[0] * image_shape[1]) if hull is not None else 0.0
        if count < min_inliers or ratio < min_inlier_ratio or coverage < min_coverage:
            return PoseEstimate(False, "geometry_gate", inliers=count, inlier_ratio=ratio, query_coverage=coverage)
        rvec, tvec = cv2.solvePnPRefineLM(world[indices], image[indices], camera, None, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        center = -rotation.T @ tvec.reshape(3)
        xyz_camera = world[indices] @ rotation.T + tvec.reshape(1, 3)
        if not np.isfinite(center).all() or np.mean(xyz_camera[:, 2] > 0) < .9:
            return PoseEstimate(False, "invalid_camera_geometry", inliers=count, inlier_ratio=ratio, query_coverage=coverage)
        projected, _ = cv2.projectPoints(world[indices], rvec, tvec, camera, None)
        residual = np.linalg.norm(projected.reshape(-1, 2) - image[indices], axis=1)
        return PoseEstimate(True, "ok", rotation, tvec.reshape(3), center, count, ratio,
                            coverage, float(np.median(residual)))
    except cv2.error as error:
        return PoseEstimate(False, f"opencv_error: {error}")
