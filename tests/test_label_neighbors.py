import unittest
from scripts.analyze_label_neighbors import nearest_index

class NearestLabelDiagnosticTest(unittest.TestCase):
    def test_window_and_index_offset(self):
        points=[(0,0),(10,0),(20,0),(30,0)]
        self.assertEqual(nearest_index(points,29,0,0,2),(2,9))
        self.assertEqual(nearest_index(points,29,0,0,3),(3,1))
        with self.assertRaises(ValueError): nearest_index(points,0,0,-1,1)

if __name__=='__main__': unittest.main()
