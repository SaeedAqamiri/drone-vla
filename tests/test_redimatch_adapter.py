import time
import unittest

import cv2
import numpy as np
import torch

from navlab.lab.deep import _homography_outcome
from navlab.lab.redi import _as_pil, _sample_correspondences


class _FakeREDI:
    def sample(self, matches, certainty, num):
        self.sampled_shape = tuple(matches.shape)
        self.certainty_shape = tuple(certainty.shape)
        return matches.reshape(-1, 4), certainty.reshape(-1)

    @staticmethod
    def to_pixel_coordinates(coords, h_ref, w_ref, h_query, w_query):
        ref = torch.stack(
            (w_ref / 2 * (coords[..., 0] + 1), h_ref / 2 * (coords[..., 1] + 1)),
            dim=-1,
        )
        query = torch.stack(
            (w_query / 2 * (coords[..., 2] + 1), h_query / 2 * (coords[..., 3] + 1)),
            dim=-1,
        )
        return ref, query


class REDIMatcherAdapterTest(unittest.TestCase):
    def test_first_symmetric_half_is_ref_to_query_and_center_projects_back(self):
        # Original images have different aspect ratios and both exceed max_dim.
        ref_image = _as_pil(np.zeros((160, 200, 3), dtype=np.uint8), max_dim=100)
        query_image = _as_pil(np.zeros((180, 120, 3), dtype=np.uint8), max_dim=100)
        self.assertEqual(ref_image.size, (100, 80))
        self.assertEqual(query_image.size, (67, 100))
        ref_shape = (ref_image.height, ref_image.width)
        query_shape = (query_image.height, query_image.width)
        ref_points = np.array(
            [[x, y] for y in (8, 40, 72) for x in (10, 50, 90)],
            dtype=np.float32,
        )
        query_points = ref_points * np.array([0.4, 0.8]) + np.array([10, 10])

        # A three-row, three-column first (A->B) half with a deliberately
        # unrelated second (B->A) half proves the adapter reads the right side.
        warp = torch.zeros((1, 3, 6, 4), dtype=torch.float32)
        certainty = torch.ones((1, 3, 6), dtype=torch.float32)
        for index, (ref, query) in enumerate(zip(ref_points, query_points)):
            row, col = divmod(index, 3)
            warp[0, row, col] = torch.tensor(
                [
                    2 * ref[0] / ref_shape[1] - 1,
                    2 * ref[1] / ref_shape[0] - 1,
                    2 * query[0] / query_shape[1] - 1,
                    2 * query[1] / query_shape[0] - 1,
                ]
            )
            warp[0, row, col + 3] = torch.tensor([0.9, 0.9, -0.9, -0.9])

        model = _FakeREDI()
        src, dst = _sample_correspondences(
            model, warp, certainty, ref_shape, query_shape, num_matches=100
        )
        self.assertEqual(model.sampled_shape, (3, 3, 4))
        np.testing.assert_allclose(src, ref_points, atol=1e-4)
        np.testing.assert_allclose(dst, query_points, atol=1e-4)

        outcome = _homography_outcome(
            "ref.png", "query.png", "redi", src, dst,
            ref_shape, query_shape, time.time(), 1.0, 4, 0.1, 0.0,
        )
        self.assertTrue(outcome.success)
        self.assertGreaterEqual(outcome.inliers, 8)
        # LabRunner inverts this ref->query H to localize the query center.
        query_center = np.array([[[query_shape[1] / 2, query_shape[0] / 2]]], dtype=np.float32)
        projected_ref_center = cv2.perspectiveTransform(
            query_center, np.linalg.inv(outcome.homography)
        )[0, 0]
        np.testing.assert_allclose(projected_ref_center, [58.75, 50], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
