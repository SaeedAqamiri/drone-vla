import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rasterio
from affine import Affine

from navlab.imagery import inspect_image
from navlab.lab.synthetic import make_queries, make_scene
from navlab.lab.tiles import TileStore, build_tiles_along_route


class LabTilesTest(unittest.TestCase):
    def test_rotated_gsd_and_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.tif"
            width, height = 64, 64
            lon0, lat0 = 50.0, 26.0
            step = 5e-6
            transform = Affine(0.0, -step, lon0, step, 0.0, lat0)
            data = np.zeros((3, height, width), np.uint8)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=3,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as dataset:
                dataset.write(data)
            info = inspect_image(path)
            self.assertTrue(info["rotated"])
            self.assertIsNotNone(info["gsd_xy_m"])
            self.assertGreater(info["gsd_xy_m"][0], 0.0)
            self.assertGreater(info["gsd_xy_m"][1], 0.0)

    def test_build_tiles_along_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = make_scene(root, size=1024, seed=2, gsd_m=0.5)
            queries = make_queries(
                root,
                scene,
                count=5,
                seed=2,
                altitude_m=250.0,
                query_size=(512, 342),
            )
            route = [(query.lon, query.lat) for query in queries]
            build_tiles_along_route(
                scene["path"],
                root / "tiles",
                route,
                tile_ground_m=500.0,
                tile_px=512,
                stride_m=150.0,
            )
            store = TileStore(root / "tiles")
            self.assertGreater(store.size, 0)
            self.assertAlmostEqual(store.gsd, 500.0 / 512.0, places=6)
            for tile in store.tiles:
                image = cv2.imread(str(store.path_for(tile)))
                self.assertIsNotNone(image)
                self.assertEqual(image.shape[:2], (512, 512))
                self.assertGreater(tile["bounds"]["max_east"], tile["bounds"]["min_east"])


if __name__ == "__main__":
    unittest.main()
