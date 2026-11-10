import unittest

from scripts.audit_visloc_label_alignment import motion_components


class LabelAlignmentAuditTest(unittest.TestCase):
    def test_along_track_cross_track_and_stationary(self):
        speed, along, cross, equivalent = motion_components(20, 7, (0, 0), (100, 0), 10)
        self.assertEqual((speed, along, cross, equivalent), (10, 20, 7, 2))
        speed, along, cross, equivalent = motion_components(20, 7, (5, 5), (5, 5), 10)
        self.assertEqual((speed, along, cross, equivalent), (0, None, None, None))
        self.assertEqual(motion_components(1, 1, (0, 0), (10, 10), 0), (0, None, None, None))

if __name__ == '__main__': unittest.main()
