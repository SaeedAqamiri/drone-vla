import unittest
import cv2
import numpy as np

from navlab.lab.pnp import estimate_pose


class PoseTest(unittest.TestCase):
    def test_known_camera_with_outliers(self):
        rng = np.random.default_rng(42)
        world = np.column_stack((rng.uniform(-20, 20, 120), rng.uniform(-15, 15, 120), rng.uniform(0, 3, 120)))
        rotation_vec = np.array([3.08, .03, .04], dtype=float)
        translation = np.array([2., 3., 70.])
        intrinsics = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
        projected, _ = cv2.projectPoints(world, rotation_vec, translation, intrinsics, None)
        image = projected.reshape(-1, 2)
        image[:10] = rng.uniform([0, 0], [640, 480], (10, 2))
        pose = estimate_pose(world, image, intrinsics, (480, 640))
        self.assertTrue(pose.success, pose.reason)
        rotation, _ = cv2.Rodrigues(rotation_vec)
        truth = -rotation.T @ translation
        np.testing.assert_allclose(pose.camera_center_world, truth, atol=.05)
        self.assertGreaterEqual(pose.inliers, 100)

    def test_rejects_few_points_and_bad_intrinsics(self):
        world = np.zeros((5, 3))
        pixels = np.zeros((5, 2))
        self.assertEqual(estimate_pose(world, pixels, np.eye(3), (480, 640)).reason, "too_few_correspondences")
        with self.assertRaises(ValueError):
            estimate_pose(world, pixels, np.zeros((3, 3)), (480, 640))


if __name__ == "__main__":
    unittest.main()
