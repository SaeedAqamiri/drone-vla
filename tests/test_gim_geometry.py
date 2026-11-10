import unittest

import numpy as np
import torch

from navlab.lab.gim import GIMDKMMatcher


class FakeDKM:
    def match(self, a, b):
        return None, None

    def sample(self, warp, certainty, n):
        xy = torch.tensor([[x, y, x, y] for y in (-.7, -.3, .1, .5) for x in (-.7, -.3, .1, .5)])
        return xy, None

    def to_pixel_coordinates(self, matches, h0, w0, h1, w1):
        return ((matches[:, :2] + 1) * torch.tensor([w0/2, h0/2]),
                (matches[:, 2:] + 1) * torch.tensor([w1/2, h1/2]))


class GIMGeometryTest(unittest.TestCase):
    def test_identity_survives_padding_and_resizing(self):
        model = GIMDKMMatcher.__new__(GIMDKMMatcher)
        model.torch, model.device, model.model = torch, torch.device("cpu"), FakeDKM()
        model.max_dim, model.min_inliers, model.min_ratio = 768, 4, .15
        model.min_coverage, model.ransac_px = .03, 3
        image = np.zeros((41, 65, 3), np.uint8)
        out = model.match(image, image)
        self.assertFalse(out.execution_error, out.error)
        self.assertTrue(out.success)
        h = out.homography / out.homography[2, 2]
        np.testing.assert_allclose(h, np.eye(3), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
