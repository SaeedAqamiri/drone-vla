import unittest
from unittest.mock import patch

import numpy as np

from navlab.lab.landmarks import LandmarkVerifiedMatcher, anchor_support
from navlab.lab.types import MatchOutcome


class FakeDense:
    def __init__(self, ref, query):
        self.ref, self.query = ref, query
    def match_with_points(self, ref, query):
        outcome = MatchOutcome('', '', 'test', len(self.ref), .9, .5, 0, 1, 0, 1, True,
                               homography=np.eye(3))
        return outcome, self.ref, self.query


class LandmarkTest(unittest.TestCase):
    def test_supported_anchors_require_spatial_and_geometric_agreement(self):
        anchors = np.array([[30,30], [100,30], [170,30]], np.float32)
        ref = np.repeat(anchors, 8, axis=0)
        query = ref.copy()
        query[-8:, 0] += 25
        counts, supported = anchor_support(anchors, ref, query, np.eye(3))
        self.assertEqual(counts, [8,8,0])
        self.assertEqual(supported, 2)
        image = np.zeros((200,200,3), np.uint8)
        with patch('navlab.lab.landmarks.map_anchors', return_value=anchors):
            matcher = LandmarkVerifiedMatcher(FakeDense(ref, query), min_supported=2)
            outcome = matcher.match(image, image)
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.landmark_supported, 2)
            matcher = LandmarkVerifiedMatcher(FakeDense(ref, query), min_supported=3)
            self.assertFalse(matcher.match(image, image).success)

if __name__ == '__main__': unittest.main()
