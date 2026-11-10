from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from sat2aerial.orthofly import (
    Camera,
    Effects,
    FlythroughConfig,
    GroundMap,
    Pose,
    build_poses,
    camera_axes,
    ground_homography,
    line_path,
    procedural_map,
    project,
    run_flythrough,
    snake_path,
)


class GeometryTests(unittest.TestCase):
    def test_axes_are_orthonormal(self) -> None:
        right, down, forward = camera_axes(37.0, 12.0, -8.0)
        for axis in (right, down, forward):
            self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=9)
        self.assertAlmostEqual(float(right @ down), 0.0, places=9)
        self.assertAlmostEqual(float(right @ forward), 0.0, places=9)
        self.assertAlmostEqual(float(down @ forward), 0.0, places=9)

    def test_nadir_looks_down_with_east_right(self) -> None:
        camera = Camera(width=640, height=640, hfov_deg=60.0)
        pose = Pose(x=0.0, y=0.0, alt=100.0)
        east = project(pose, camera, 10.0, 0.0)
        north = project(pose, camera, 0.0, 10.0)
        focal = camera.focal_px
        self.assertIsNotNone(east)
        self.assertIsNotNone(north)
        assert east is not None and north is not None
        self.assertAlmostEqual(east[0], 320.0 + focal * 10.0 / 100.0, places=6)
        self.assertAlmostEqual(east[1], 320.0, places=6)
        # +v in the image points south, so a point to the north lands above centre
        self.assertAlmostEqual(north[0], 320.0, places=6)
        self.assertAlmostEqual(north[1], 320.0 - focal * 10.0 / 100.0, places=6)

    def test_homography_agrees_with_projection(self) -> None:
        camera = Camera(width=512, height=384, hfov_deg=73.0)
        pose = Pose(x=12.0, y=-7.0, alt=80.0, yaw_deg=123.0, pitch_deg=15.0, roll_deg=4.0)
        matrix = ground_homography(pose, camera)
        for x, y in ((-10.0, 5.0), (0.0, 0.0), (18.0, -9.0), (25.0, 12.0)):
            expected = project(pose, camera, x, y)
            self.assertIsNotNone(expected)
            assert expected is not None
            p = matrix @ np.array([x, y, 1.0])
            self.assertAlmostEqual(p[0] / p[2], expected[0], places=6)
            self.assertAlmostEqual(p[1] / p[2], expected[1], places=6)


class MapTests(unittest.TestCase):
    def _map(self) -> GroundMap:
        # 400x400 px at 1 m/px -> 400 m square, indexed so position is readable.
        rows, cols = np.mgrid[0:400, 0:400]
        image = np.dstack([(cols % 256).astype(np.uint8), (rows % 256).astype(np.uint8), np.full_like(cols, 90, dtype=np.uint8)])
        return GroundMap.from_array(image, gsd_m=1.0, x0=-200.0, y0=200.0)

    def test_local_pixel_roundtrip(self) -> None:
        ground = self._map()
        for x, y in ((-150.0, 150.0), (0.0, 0.0), (37.0, -81.0)):
            col, row = ground.local_to_pixel(x, y)
            self.assertAlmostEqual(col, x + 200.0, places=6)
            self.assertAlmostEqual(row, 200.0 - y, places=6)
            bx, by = ground.pixel_to_local(col, row)
            self.assertAlmostEqual(bx, x, places=6)
            self.assertAlmostEqual(by, y, places=6)

    def test_render_centre_is_map_centre(self) -> None:
        ground = self._map()
        camera = Camera(width=64, height=64, hfov_deg=60.0)
        pose = Pose(x=37.0, y=-81.0, alt=100.0)
        frame = ground.render(pose, camera)
        self.assertEqual(frame.shape, (64, 64, 3))
        col, row = ground.local_to_pixel(pose.x, pose.y)
        expected = ground.image[int(round(row)), int(round(col))]
        np.testing.assert_allclose(frame[32, 32].astype(int), expected.astype(int), atol=3)

    def test_image_matrix_matches_projection(self) -> None:
        ground = self._map()
        camera = Camera(width=256, height=256, hfov_deg=60.0)
        pose = Pose(x=10.0, y=20.0, alt=120.0, yaw_deg=45.0, pitch_deg=10.0)
        matrix = ground.image_matrix(pose, camera)
        for x, y in ((-5.0, 30.0), (25.0, 10.0), (10.0, 20.0)):
            col, row = ground.local_to_pixel(x, y)
            p = matrix @ np.array([col, row, 1.0])
            expected = project(pose, camera, x, y)
            assert expected is not None
            self.assertAlmostEqual(p[0] / p[2], expected[0], places=5)
            self.assertAlmostEqual(p[1] / p[2], expected[1], places=5)

    def test_crop_is_centred(self) -> None:
        ground = self._map()
        crop = ground.crop(30.0, -40.0, 64)
        self.assertEqual(crop.shape, (64, 64, 3))
        col, row = ground.local_to_pixel(30.0, -40.0)
        np.testing.assert_allclose(crop[32, 32].astype(int), ground.image[int(row), int(col)].astype(int), atol=1)


class PathTests(unittest.TestCase):
    def test_line_length_and_spacing(self) -> None:
        cfg = FlythroughConfig(out=Path("/tmp/unused"), path_kind="line", frames=61, length_m=300.0, heading_deg=90.0)
        poses = build_poses(cfg)
        self.assertEqual(len(poses), 61)
        self.assertAlmostEqual(poses[0].x, 0.0, places=6)
        self.assertAlmostEqual(poses[-1].x, 300.0, places=6)
        self.assertAlmostEqual(poses[-1].y, 0.0, places=6)
        spacing = math.dist((poses[0].x, poses[0].y), (poses[1].x, poses[1].y))
        self.assertAlmostEqual(spacing, 5.0, places=6)
        # heading east -> yaw 90
        self.assertAlmostEqual(poses[0].yaw_deg, 90.0, places=3)

    def test_snake_has_expected_vertices(self) -> None:
        points = snake_path(100.0, 0.0, legs=3, leg_spacing_m=50.0)
        self.assertEqual(len(points), 6)
        self.assertAlmostEqual(points[1][1], 100.0, places=6)
        self.assertAlmostEqual(points[2][0], 50.0, places=6)

    def test_line_path_endpoint(self) -> None:
        points = line_path(10.0, 0.0)
        self.assertAlmostEqual(points[-1][0], 0.0, places=6)
        self.assertAlmostEqual(points[-1][1], 10.0, places=6)


class FlythroughTests(unittest.TestCase):
    def test_end_to_end_procedural(self) -> None:
        ground = procedural_map(size=512, gsd_m=1.0, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            pairs = Path(tmp) / "pairs"
            cfg = FlythroughConfig(
                out=out,
                frames=6,
                camera=Camera(width=128, height=128, hfov_deg=60.0),
                altitude_m=60.0,
                length_m=100.0,
                ref_size_px=96,
                effects=Effects(blur_sigma=0.0, noise_sigma=0.0, gamma_jitter=0.0, exposure_jitter=0.0, vignette=0.0, exposure_ms=0.0),
                pairs_dir=pairs,
            )
            meta = run_flythrough(ground, cfg)

            self.assertEqual(len(list((out / "frames").glob("*.png"))), 6)
            self.assertTrue((out / "preview.png").is_file())
            self.assertTrue((out / "meta.json").is_file())
            self.assertAlmostEqual(meta["distance_m"], 100.0, places=3)

            with (out / "poses.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertIn("lat", rows[0])
            self.assertAlmostEqual(float(rows[0]["x_m"]), 0.0, places=4)
            self.assertAlmostEqual(float(rows[-1]["x_m"]), 100.0, places=4)
            gsd = 2.0 * 60.0 * math.tan(math.radians(30.0)) / 128
            self.assertAlmostEqual(float(rows[0]["gsd_m"]), gsd, places=3)

            self.assertEqual(len(list((pairs / "satellite").glob("*.png"))), 6)
            self.assertEqual(len(list((pairs / "aerial").glob("*.png"))), 6)
            satellite = cv2.imread(str(pairs / "satellite" / "frame_0000.png"))
            self.assertEqual(satellite.shape[:2], (96, 96))

    @unittest.skipIf(__import__("importlib").util.find_spec("rasterio") is None, "rasterio missing")
    def test_geotiff_roundtrip(self) -> None:
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "site.tif"
            width = height = 200
            gsd = 0.5
            # UTM 33N, 0.5 m pixels, north-up
            transform = from_origin(400_000.0, 5_000_000.0, gsd, gsd)
            rows, cols = np.mgrid[0:height, 0:width]
            data = np.dstack([(cols % 256).astype(np.uint8), (rows % 256).astype(np.uint8), np.full((height, width), 120, np.uint8)])
            with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs="EPSG:32633", transform=transform) as dst:
                dst.write(np.transpose(data, (2, 0, 1)))

            ground = GroundMap.from_geotiff(path)
            self.assertAlmostEqual(ground.gsd_m, gsd, places=6)
            self.assertAlmostEqual(ground.origin_lat, 45.0, delta=0.3)
            self.assertAlmostEqual(ground.origin_lon, 13.7, delta=0.3)

            for x, y in ((0.0, 0.0), (20.0, -10.0), (-30.0, 15.0)):
                col, row = ground.local_to_pixel(x, y)
                bx, by = ground.pixel_to_local(col, row)
                self.assertAlmostEqual(bx, x, places=4)
                self.assertAlmostEqual(by, y, places=4)

            lat, lon = ground.latlon(0.0, 0.0)
            self.assertAlmostEqual(lat, ground.origin_lat, places=5)
            self.assertAlmostEqual(lon, ground.origin_lon, places=5)

            camera = Camera(width=128, height=128, hfov_deg=60.0)
            pose = Pose(x=0.0, y=0.0, alt=20.0)
            frame = ground.render(pose, camera)
            self.assertEqual(frame.shape, (128, 128, 3))
            self.assertGreater(int(frame.std()), 5)


@unittest.skipIf(__import__("importlib").util.find_spec("rasterio") is None, "rasterio missing")
class GeoTiffCrsTests(unittest.TestCase):
    """Real archives are often geographic (degrees) or rotated, not north-up UTM."""

    @staticmethod
    def _write(path: Path, transform, crs: str, width: int, height: int) -> None:
        import rasterio

        from sat2aerial.synthetic import make_satellite

        image = make_satellite(np.random.default_rng(5), size=max(width, height))[:height, :width]
        with rasterio.open(
            path, "w", driver="GTiff", width=width, height=height, count=3,
            dtype="uint8", crs=crs, transform=transform,
        ) as dst:
            dst.write(np.transpose(image, (2, 0, 1)))

    def test_render_is_centred_on_the_pose_for_a_geotiff(self) -> None:
        """Regression: the renderer used to centre frames at 2*pose on georeferenced rasters."""
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "utm.tif"
            width = height = 1600
            gsd = 0.3
            # smooth gradients: a sub-pixel shift changes the value by <<1 level,
            # so the centre pixel is a precise test of the geometry
            rows, cols = np.mgrid[0:height, 0:width]
            image = np.dstack(
                [
                    (cols * 255 // width).astype(np.uint8),
                    (rows * 255 // height).astype(np.uint8),
                    np.full((height, width), 120, np.uint8),
                ]
            )
            import rasterio

            with rasterio.open(
                path, "w", driver="GTiff", width=width, height=height, count=3,
                dtype="uint8", crs="EPSG:32633",
                transform=from_origin(400_000.0, 5_000_000.0, gsd, gsd),
            ) as dst:
                dst.write(np.transpose(image, (2, 0, 1)))
            ground = GroundMap.from_geotiff(path)

            camera = Camera(width=256, height=256, hfov_deg=60.0)
            for x, y in ((0.0, 0.0), (60.0, -40.0), (150.0, 90.0)):
                pose = Pose(x=x, y=y, alt=120.0)
                frame = ground.render(pose, camera)
                col, row = ground.local_to_pixel(x, y)
                expected = ground.image[int(round(row)), int(round(col))]
                np.testing.assert_allclose(
                    frame[128, 128].astype(int), expected.astype(int), atol=4,
                    err_msg=f"frame centre is not the pose at ({x}, {y})",
                )

    def test_geographic_degrees(self) -> None:
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geo.tif"
            # 0.3 m/px near 45 N: 2.7e-6 deg lon, 2.7e-6 deg lat
            self._write(path, from_origin(13.40, 45.00, 2.7e-6, 2.7e-6), "EPSG:4326", 300, 300)
            ground = GroundMap.from_geotiff(path)
            # square pixels in degrees are anisotropic in metres at 45 N
            self.assertAlmostEqual(ground.gsd_xy_m[0], 0.3 * math.cos(math.radians(45.0)), delta=0.02)
            self.assertAlmostEqual(ground.gsd_xy_m[1], 0.3, delta=0.02)
            self.assertAlmostEqual(ground.gsd_m, 0.2565, delta=0.02)

            for x, y in ((0.0, 0.0), (20.0, -12.0), (-25.0, 18.0)):
                col, row = ground.local_to_pixel(x, y)
                bx, by = ground.pixel_to_local(col, row)
                self.assertAlmostEqual(bx, x, delta=0.05)
                self.assertAlmostEqual(by, y, delta=0.05)

    def test_rotated_raster(self) -> None:
        from rasterio.transform import Affine

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rot.tif"
            transform = Affine.translation(400_000.0, 5_000_000.0) @ Affine.rotation(17.0) @ Affine.scale(0.5, -0.5)
            self._write(path, transform, "EPSG:32633", 400, 400)
            ground = GroundMap.from_geotiff(path)
            self.assertAlmostEqual(ground.gsd_m, 0.5, places=6)

            for x, y in ((0.0, 0.0), (30.0, 10.0), (60.0, -40.0)):
                col, row = ground.local_to_pixel(x, y)
                bx, by = ground.pixel_to_local(col, row)
                self.assertAlmostEqual(bx, x, delta=0.05)
                self.assertAlmostEqual(by, y, delta=0.05)

            camera = Camera(width=96, height=96, hfov_deg=60.0)
            frame = ground.render(Pose(x=30.0, y=10.0, alt=40.0, yaw_deg=70.0), camera)
            self.assertGreater(int(frame.std()), 5)


if __name__ == "__main__":
    unittest.main()
