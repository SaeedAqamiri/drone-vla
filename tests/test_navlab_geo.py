import unittest

from navlab.geo import (
    distance_to_polyline_m,
    route_summary,
    route_summary_multi,
    tile_bounds_along_route,
    tile_bounds_multi,
)


class NavlabGeoTest(unittest.TestCase):
    def test_route_summary(self):
        summary = route_summary(35.6892, 51.3890, 35.7000, 51.4200, 2.0, 0.5)
        self.assertGreater(summary["length_km"], 3.0)
        self.assertLess(summary["length_km"], 4.0)
        self.assertGreater(summary["area_km2"], 5.0)
        self.assertGreater(summary["tile_count_estimate"], 0)
        self.assertEqual(summary["tile_ground_m"], 512.0)

    def test_tiles(self):
        tiles = tile_bounds_along_route(
            35.6892, 51.3890, 35.7000, 51.4200, tile_ground_m=512, limit=4
        )
        self.assertGreaterEqual(len(tiles), 2)
        self.assertLessEqual(len(tiles), 4)
        for tile in tiles:
            bounds = tile["bounds"]
            self.assertLess(bounds["min_lat"], bounds["max_lat"])
            self.assertLess(bounds["min_lon"], bounds["max_lon"])


    def test_waypoint_route(self):
        points = [(35.6892, 51.3890), (35.6895, 51.3895), (35.6898, 51.3900)]
        summary = route_summary_multi(points, 0.3, 0.5)
        self.assertGreater(summary["length_km"], 0.05)
        self.assertGreater(len(summary["polygon"]), 3)
        self.assertGreaterEqual(len(tile_bounds_multi(points, 512, 4)), 2)
        self.assertLess(distance_to_polyline_m(points, 35.6895, 51.3895), 5.0)


if __name__ == "__main__":
    unittest.main()
