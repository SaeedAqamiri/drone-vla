from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import import_uav_visloc  # noqa: E402
from sat2aerial.localize import LocalizeConfig, evaluate  # noqa: E402


def _make_flight(root: Path, size: int = 1600, gsd: float = 0.5, frames: int = 3) -> Path:
    site = root / "01"
    (site / "drone").mkdir(parents=True)
    transform = from_origin(400_000.0, 5_000_000.0, gsd, gsd)
    from sat2aerial.synthetic import make_satellite

    data = make_satellite(np.random.default_rng(11), size=size)
    tif = site / "satellite01.tif"
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs="EPSG:32633",
        transform=transform,
    ) as dst:
        dst.write(np.transpose(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), (2, 0, 1)))

    to_latlon = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
    records = []
    for index in range(frames):
        col = 300 + index * 120
        row = 400 + index * 120
        crop = data[row - 200 : row + 200, col - 200 : col + 200]
        name = f"01_{index + 1:04d}.jpg"
        cv2.imwrite(str(site / "drone" / name), crop)
        x, y = transform @ (col, row)
        lon, lat = to_latlon.transform(x, y)
        records.append({"filename": name, "latitude": lat, "longitude": lon, "height": 120.0, "phi": 30.0})

    with (site / "01.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "latitude", "longitude", "height", "phi"])
        writer.writeheader()
        writer.writerows(records)
    return site


class ImportUavVislocTests(unittest.TestCase):
    def test_convert_and_localize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = _make_flight(root)
            out = root / "converted"
            meta = import_uav_visloc.convert_flight(
                site=site,
                out=out,
                ref_meters=300.0,
                ref_size=512,
                ref_aspect=0.0,
                query_size=0,
                max_frames=0,
                offset_m=(0.0, 0.0),
                jpeg_quality=95,
                frame_gsd_arg="0.5",
            )

            self.assertEqual(meta["frames"], 3)
            self.assertAlmostEqual(
                meta["map"]["gsd_m"],
                meta["map"]["ref_meters"] / meta["map"]["ref_size_px"],
                places=6,
            )
            self.assertEqual(len(list((out / "pairs" / "satellite").glob("*.jpg"))), 3)
            self.assertEqual(len(list((out / "pairs" / "aerial").glob("*.jpg"))), 3)

            with (out / "poses.csv").open(encoding="utf-8") as handle:
                poses = list(csv.DictReader(handle))
            self.assertEqual(len(poses), 3)
            self.assertIn("ref_center_x_m", poses[0])
            # the converted flight must be origin-relative, so the first frame is not at 0,0
            self.assertTrue(any(abs(float(p["x_m"])) > 1e-6 or abs(float(p["y_m"])) > 1e-6 for p in poses))

            summary = evaluate(
                LocalizeConfig(
                    run=out,
                    out=out / "eval",
                    mode="gray",
                    detector="sift",
                    max_dim=1024,
                )
            )
            self.assertEqual(summary["frames"], 3)
            self.assertGreaterEqual(summary["fix_rate"], 0.99)
            self.assertIsNotNone(summary["error_m"]["median"])
            self.assertLess(summary["error_m"]["median"], 4.0)
            self.assertTrue((out / "eval" / "results.csv").is_file())
            saved = json.loads((out / "eval" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["detector"], "sift")

    def test_stride_measures_the_gsd_before_downsampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = _make_flight(root, frames=4)
            out = root / "strided"
            meta = import_uav_visloc.convert_flight(
                site=site,
                out=out,
                ref_meters=300.0,
                ref_size=256,
                ref_aspect=0.0,
                query_size=0,
                max_frames=0,
                offset_m=(0.0, 0.0),
                jpeg_quality=95,
                frame_gsd_arg="auto",
                frame_stride=4,
            )
            # four frames 120 px apart were strided to one, but the GSD must
            # still come from the consecutive pairs
            self.assertEqual(meta["frames"], 1)
            self.assertAlmostEqual(meta["map"]["frame_gsd_m"], 0.5, delta=0.02)

    def test_offset_moves_the_reference_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = _make_flight(root, frames=1)
            out = root / "shifted"
            meta = import_uav_visloc.convert_flight(
                site=site,
                out=out,
                ref_meters=300.0,
                ref_size=512,
                ref_aspect=0.0,
                query_size=0,
                max_frames=0,
                offset_m=(40.0, -25.0),
                jpeg_quality=95,
                frame_gsd_arg="0.5",
            )
            self.assertEqual(meta["offset_m"], [40.0, -25.0])
            with (out / "poses.csv").open(encoding="utf-8") as handle:
                pose = next(csv.DictReader(handle))
            self.assertAlmostEqual(
                float(pose["ref_center_x_m"]) - float(pose["x_m"]), 40.0, places=3
            )
            self.assertAlmostEqual(
                float(pose["ref_center_y_m"]) - float(pose["y_m"]), -25.0, places=3
            )


class ImportUavVislocV2Tests(unittest.TestCase):
    @staticmethod
    def _make_flight(root: Path, gsd: float = 0.2, tile: int = 200, step: int = 100) -> tuple[Path, np.ndarray]:
        """A v2-style flight: named chips + a drone track cropped from the same map."""
        site = root / "07"
        (site / "satellite-tile").mkdir(parents=True)
        (site / "UAV-image").mkdir(parents=True)

        from sat2aerial.synthetic import make_satellite

        mosaic = make_satellite(np.random.default_rng(7), size=tile + 6 * step)
        offsets = list(range(0, 6 * step + 1, step))
        tile_rows, tile_cols = [], []
        for row in offsets:
            for col in offsets:
                name = f"{row}_{col}.jpg"
                cv2.imwrite(str(site / "satellite-tile" / name), mosaic[row : row + tile, col : col + tile])
                tile_rows.append({"patchname": name})
                tile_cols.append((row, col))

        lat0, lon0 = 40.0, 116.0
        import math

        m_lat, m_lon = import_uav_visloc.meters_per_degree(lat0)

        def to_latlon(col_px: float, row_px: float) -> tuple[float, float]:
            east = (col_px - (0 + tile / 2.0)) * gsd
            north = ((0 + tile / 2.0) - row_px) * gsd
            return lat0 + north / m_lat, lon0 + east / m_lon

        records = []
        for (row, col), tile_row in zip(tile_cols, tile_rows):
            lat, lon = to_latlon(col + tile / 2.0, row + tile / 2.0)
            tile_row["latitude"] = f"{lat:.12f}"
            tile_row["longitude"] = f"{lon:.12f}"

        with (site / "07-satellite-tile-coordinates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["patchname", "latitude", "longitude"])
            writer.writeheader()
            writer.writerows(tile_rows)

        for index, col in enumerate((300, 450, 600)):
            row = 300
            half = 80
            frame = mosaic[row - half : row + half, col - half : col + half]
            name = f"07_{index + 1:04d}.JPG"
            cv2.imwrite(str(site / "UAV-image" / name), frame)
            lat, lon = to_latlon(col, row)
            records.append(
                {"filename": name, "lat": f"{lat:.12f}", "lon": f"{lon:.12f}", "height": "200.0"}
            )
        with (site / "07-UAV-coordinates.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "lat", "lon", "height"])
            writer.writeheader()
            writer.writerows(records)
        return site, mosaic

    def test_tile_names_without_a_prefix(self) -> None:
        self.assertEqual(import_uav_visloc._tile_rc(Path("0_135.jpg")), (0, 135))
        self.assertEqual(import_uav_visloc._tile_rc(Path("satellite07_135_270.jpg")), (135, 270))

    def test_crop_is_rectangular_and_clipped(self) -> None:
        image = np.full((100, 120, 3), 7, np.uint8)
        crop = import_uav_visloc._crop_array(image, 5.0, 8.0, 40, 20)
        self.assertEqual(crop.shape, (20, 40, 3))
        # centred at col 5 with width 40 -> the first 15 columns are outside
        self.assertTrue(np.all(crop[2:, 14] == 0))
        self.assertTrue(np.all(crop[2:, 15] == 7))
        # centred at row 8 with height 20 -> the first two rows are outside
        self.assertTrue(np.all(crop[0, 15:] == 0))
        self.assertTrue(np.all(crop[2, 15:] == 7))

    def test_v2_convert_keeps_the_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site, _ = self._make_flight(root)
            out = root / "converted"
            meta = import_uav_visloc.convert_flight_v2(
                site=site,
                out=out,
                ref_size=0,
                ref_meters=0.0,
                ref_aspect=0.0,
                query_size=0,
                max_frames=0,
                offset_m=(0.0, 0.0),
                jpeg_quality=95,
                frame_gsd_arg="0.2",
                strip_shift_arg="off",
            )
            self.assertEqual(meta["layout"], "uav_visloc_v2")
            self.assertEqual(meta["frames"], 3)
            self.assertEqual(meta["warnings"], [])
            self.assertAlmostEqual(meta["map"]["mosaic_gsd_m"], 0.2, places=4)
            self.assertEqual(meta["map"]["ref_size_py"], 184)
            self.assertGreater(meta["map"]["ref_meters"], 2 * meta["map"]["ref_meters_y"])

            with (out / "poses.csv").open(encoding="utf-8") as handle:
                poses = list(csv.DictReader(handle))
            self.assertEqual(len(poses), 3)
            # the track sits exactly on the chip grid, so the map pixels are known
            self.assertAlmostEqual(float(poses[0]["map_col"]), 300.0, delta=1.0)
            self.assertAlmostEqual(float(poses[0]["map_row"]), 300.0, delta=1.0)

            reference = cv2.imread(str(out / "pairs" / "satellite" / "07_0001.jpg"))
            self.assertEqual(reference.shape[:2], (184, meta["map"]["ref_size_px"]))

            summary = evaluate(
                LocalizeConfig(
                    run=out,
                    out=out / "eval",
                    matcher="ncc",
                    min_score=0.2,
                    max_dim=1024,
                )
            )
            self.assertEqual(summary["frames"], 3)
            self.assertGreaterEqual(summary["fix_rate"], 0.99)
            self.assertLess(summary["error_m"]["median"], 2.0)
            self.assertLess(summary["bias_m"]["magnitude"], 2.0)

    def test_strip_shift_auto_centres_the_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site, mosaic = self._make_flight(root)
            out = root / "centred"
            meta = import_uav_visloc.convert_flight_v2(
                site=site,
                out=out,
                ref_size=256,
                ref_meters=0.0,
                ref_aspect=0.0,
                query_size=0,
                max_frames=0,
                offset_m=(0.0, 0.0),
                jpeg_quality=95,
                frame_gsd_arg="0.2",
                strip_shift_arg="auto",
            )
            with (out / "poses.csv").open(encoding="utf-8") as handle:
                poses = list(csv.DictReader(handle))
            # the track row is now the middle of the mosaic
            centre = mosaic.shape[0] / 2.0
            for pose in poses:
                self.assertAlmostEqual(float(pose["map_row"]), centre, delta=1.0)
            self.assertNotAlmostEqual(meta["strip_shift_m"], 0.0, places=3)


class CorrelationMatcherTests(unittest.TestCase):
    def test_finds_a_known_offset(self) -> None:
        from sat2aerial.localize import _correlate
        from sat2aerial.synthetic import make_satellite

        big = make_satellite(np.random.default_rng(3), size=400)
        top, left, height, width = 70, 130, 120, 160
        template = big[top : top + height, left : left + width]
        score, centre_x, centre_y = _correlate(big, template)
        self.assertGreater(score, 0.95)
        self.assertAlmostEqual(centre_x, left + width / 2.0, delta=1.5)
        self.assertAlmostEqual(centre_y, top + height / 2.0, delta=1.5)

    def test_large_reference_uses_the_pyramid(self) -> None:
        from sat2aerial.localize import _correlate
        from sat2aerial.synthetic import make_satellite

        big = make_satellite(np.random.default_rng(5), size=1600)
        top, left, height, width = 700, 900, 240, 300
        template = big[top : top + height, left : left + width]
        score, centre_x, centre_y = _correlate(big, template)
        self.assertGreater(score, 0.85)
        self.assertAlmostEqual(centre_x, left + width / 2.0, delta=4.0)
        self.assertAlmostEqual(centre_y, top + height / 2.0, delta=4.0)

    def test_rotation_search_recovers_a_rotated_frame(self) -> None:
        import cv2

        from sat2aerial.localize import _correlate
        from sat2aerial.synthetic import make_satellite

        big = make_satellite(np.random.default_rng(9), size=500)
        top, left, height, width = 100, 120, 160, 200
        template = big[top : top + height, left : left + width]
        rotated = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
        angles = tuple(float(value) for value in range(0, 180, 5))
        score, centre_x, centre_y = _correlate(big, rotated, angles=angles)
        self.assertGreater(score, 0.8)
        self.assertAlmostEqual(centre_x, left + width / 2.0, delta=6.0)
        self.assertAlmostEqual(centre_y, top + height / 2.0, delta=6.0)

    def test_fit_crops_the_centre_band(self) -> None:
        from sat2aerial.localize import _fit_to_reference

        query = np.zeros((400, 600, 3), np.uint8)
        query[100:300, :, :] = 9
        fitted = _fit_to_reference((200, 400), query)
        self.assertEqual(fitted.shape[:2], (200, 400))
        # the crop is centred, so it keeps the middle rows of the marked band
        self.assertTrue(np.all(fitted[:, :, 0] == 9))


if __name__ == "__main__":
    unittest.main()
