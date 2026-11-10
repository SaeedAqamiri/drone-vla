from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import make_geo_world  # noqa: E402
from sat2aerial.synthetic import make_satellite  # noqa: E402

SIZE = 1200  # 600 m of source map, so a 400 m world fits inside with no border
GSD = 0.5
ORIGIN_X = 400_000.0
ORIGIN_Y = 5_000_000.0


def _center_latlon() -> tuple[float, float]:
    from pyproj import Transformer

    centre_x = ORIGIN_X + SIZE * GSD / 2.0
    centre_y = ORIGIN_Y - SIZE * GSD / 2.0
    lon, lat = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True).transform(centre_x, centre_y)
    return float(lat), float(lon)


def _make_ortho(path: Path) -> None:
    image = make_satellite(np.random.default_rng(23), size=SIZE)
    # marker at ENU (+60, +60): NE quadrant, near the SW corner of tile_1_1
    centre_col = SIZE // 2 + int(round(60 / GSD))
    centre_row = SIZE // 2 - int(round(60 / GSD))
    # magenta does not occur in the procedural city, so it is unambiguous
    cv2.circle(image, (centre_col, centre_row), 12, (255, 0, 255), -1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=3,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, GSD, GSD),
    ) as dst:
        # GeoTIFF bands are R, G, B — cv2 arrays are B, G, R
        dst.write(np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (2, 0, 1)))


def _marker_pixel(image: np.ndarray) -> tuple[int, int]:
    score = image[:, :, 0].astype(np.int16) - image[:, :, 1].astype(np.int16)  # B - G
    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(row), int(col)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": rows}), encoding="utf-8")


class GeoWorldTests(unittest.TestCase):
    def test_tiles_are_georeferenced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ortho = tmp_path / "site.tif"
            _make_ortho(ortho)
            lat0, lon0 = _center_latlon()
            out = tmp_path / "geo"

            code = make_geo_world.main(
                [
                    "--ortho", str(ortho),
                    "--name", "demo",
                    "--out", str(out),
                    "--extent-m", "400",
                    "--tile-m", "200",
                    "--tile-px", "256",
                    "--center-lat", f"{lat0:.9f}",
                    "--center-lon", f"{lon0:.9f}",
                    "--clean",
                ]
            )
            self.assertEqual(code, 0)

            meta = json.loads((out / "demo" / "meta.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(meta["extent_m"], 400.0)
            self.assertAlmostEqual(meta["ground_gsd_m"], 200.0 / 256.0, places=6)
            self.assertAlmostEqual(meta["origin"]["lat"], lat0, places=6)
            self.assertAlmostEqual(meta["origin"]["lon"], lon0, places=6)
            self.assertAlmostEqual(meta["coverage"], 1.0, places=3)

            world = (out / "demo" / "worlds" / "demo.sdf").read_text(encoding="utf-8")
            world_lat = float(re.search(r"<latitude_deg>([-\d.]+)</latitude_deg>", world).group(1))
            world_lon = float(re.search(r"<longitude_deg>([-\d.]+)</longitude_deg>", world).group(1))
            self.assertAlmostEqual(world_lat, lat0, places=8)
            self.assertAlmostEqual(world_lon, lon0, places=8)
            self.assertIn("<world_frame_orientation>ENU</world_frame_orientation>", world)
            self.assertIn("model://geo_demo", world)
            self.assertIn("model://iris_down_cam", world)
            self.assertNotIn("geo_demo_buildings", world)

            model = (out / "demo" / "models" / "geo_demo" / "model.sdf").read_text(encoding="utf-8")
            self.assertEqual(model.count("<visual name="), 4)

            env = (out / "demo" / "env.sh").read_text(encoding="utf-8")
            self.assertIn("GZ_SIM_RESOURCE_PATH", env)
            self.assertIn("geo/demo/models", env.replace("\\", "/"))

            # The marker at ENU (+60, +60) must land in the north-east tile, near
            # its south-west corner: row in the lower half, column in the left half.
            ne = cv2.imread(str(out / "demo" / "models" / "geo_demo" / "textures" / "tile_1_1.png"))
            padded = 256 + 2 * make_geo_world.TILE_PAD_PX
            self.assertEqual(ne.shape[:2], (padded, padded))
            row, col = _marker_pixel(ne)
            self.assertGreater(row, 128)
            self.assertLess(col, 128)

            sw = cv2.imread(str(out / "demo" / "models" / "geo_demo" / "textures" / "tile_0_0.png"))
            sw_score = sw[:, :, 0].astype(np.int16) - sw[:, :, 1].astype(np.int16)
            ne_score = ne[:, :, 0].astype(np.int16) - ne[:, :, 1].astype(np.int16)
            self.assertGreater(int(ne_score.max()), 180)
            self.assertLess(int(sw_score.max()), 180)

    def test_buildings_are_extruded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ortho = tmp_path / "site.tif"
            _make_ortho(ortho)

            # Two ~20 m rectangles a few metres north-east of the origin.
            lat0, lon0 = _center_latlon()
            dlat = 20.0 / 111_320.0
            dlon = 20.0 / (111_320.0 * math.cos(math.radians(lat0)))
            footprints = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"building": "yes", "height": "20"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [lon0, lat0],
                                [lon0 + dlon, lat0],
                                [lon0 + dlon, lat0 + dlat],
                                [lon0, lat0 + dlat],
                                [lon0, lat0],
                            ]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"building": "yes", "building:levels": "3"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [lon0 + 2 * dlon, lat0 + 2 * dlat],
                                [lon0 + 3 * dlon, lat0 + 2 * dlat],
                                [lon0 + 3 * dlon, lat0 + 3 * dlat],
                                [lon0 + 2 * dlon, lat0 + 3 * dlat],
                                [lon0 + 2 * dlon, lat0 + 2 * dlat],
                            ]],
                        },
                    },
                ],
            }
            buildings = tmp_path / "buildings.geojson"
            _write_rows(buildings, footprints["features"])
            out = tmp_path / "geo"

            code = make_geo_world.main(
                [
                    "--ortho", str(ortho),
                    "--name", "bld",
                    "--out", str(out),
                    "--extent-m", "400",
                    "--tile-m", "200",
                    "--tile-px", "128",
                    "--center-lat", str(lat0),
                    "--center-lon", str(lon0),
                    "--buildings", str(buildings),
                    "--clean",
                ]
            )
            self.assertEqual(code, 0)

            meta = json.loads((out / "bld" / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["buildings"]["enabled"])
            self.assertEqual(meta["buildings"]["count"], 2)

            world = (out / "bld" / "worlds" / "bld.sdf").read_text(encoding="utf-8")
            self.assertIn("model://geo_bld_buildings", world)

            obj = (out / "bld" / "models" / "geo_bld_buildings" / "meshes" / "buildings.obj").read_text(encoding="utf-8")
            vertices = [line for line in obj.splitlines() if line.startswith("v ")]
            faces = [line for line in obj.splitlines() if line.startswith("f ")]
            self.assertEqual(len(vertices), 16)  # 2 prisms x 4 corners x (bottom + top)
            self.assertEqual(len(faces), 20)  # 2 prisms x (4 sides x 2 + 2 roof triangles)
            heights = [float(line.split()[3]) for line in vertices]
            self.assertAlmostEqual(max(heights), 20.0, places=3)
            # the second building uses building:levels = 3 -> 9 m
            self.assertTrue(any(abs(value - 9.0) < 0.01 for value in heights))

    def test_buildings_outside_extent_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ortho = tmp_path / "site.tif"
            _make_ortho(ortho)
            lat0, lon0 = _center_latlon()
            far_lat, far_lon = lat0 + 0.2, lon0 + 0.2
            far = {
                "type": "Feature",
                "properties": {"height": "12"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [far_lon, far_lat],
                        [far_lon + 0.0001, far_lat],
                        [far_lon + 0.0001, far_lat + 0.0001],
                        [far_lon, far_lat + 0.0001],
                        [far_lon, far_lat],
                    ]],
                },
            }
            buildings = tmp_path / "far.geojson"
            _write_rows(buildings, [far])
            out = tmp_path / "geo"
            make_geo_world.main(
                [
                    "--ortho", str(ortho), "--name", "far", "--out", str(out),
                    "--extent-m", "200", "--tile-m", "200", "--tile-px", "64",
                    "--center-lat", str(lat0), "--center-lon", str(lon0),
                    "--buildings", str(buildings), "--clean",
                ]
            )
            meta = json.loads((out / "far" / "meta.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["buildings"]["enabled"])
            world = (out / "far" / "worlds" / "far.sdf").read_text(encoding="utf-8")
            self.assertNotIn("geo_far_buildings", world)


if __name__ == "__main__":
    unittest.main()
