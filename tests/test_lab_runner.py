import tempfile
import unittest
from pathlib import Path

from navlab.lab.runner import LabRunner, RunnerConfig
from navlab.lab.synthetic import make_queries, make_scene
from navlab.lab.tiles import TileStore, build_tiles_along_route


class LabRunnerTest(unittest.TestCase):
    def test_synthetic_sequence_localizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = make_scene(root, size=1536, seed=4, gsd_m=0.5)
            queries = make_queries(
                root,
                scene,
                count=6,
                seed=4,
                altitude_m=300.0,
                query_size=(768, 512),
                start_fraction=0.25,
                end_fraction=0.7,
            )
            build_tiles_along_route(
                scene["path"],
                root / "tiles",
                [(query.lon, query.lat) for query in queries],
                tile_ground_m=600.0,
                tile_px=768,
                stride_m=200.0,
            )
            store = TileStore(root / "tiles")
            config = RunnerConfig(
                matcher="sift",
                mode="gray",
                base_radius=150.0,
                max_radius=1000.0,
                max_jump_m=300.0,
                scale_tolerance=1.0,
                max_candidates=8,
                bootstrap_candidates=12,
            )
            result = LabRunner(store, queries, config).run()
            self.assertGreaterEqual(result.summary["accepted"], 4)
            self.assertIsNotNone(result.summary["p95_error_m"])
            self.assertLessEqual(result.summary["p95_error_m"], 40.0)


if __name__ == "__main__":
    unittest.main()
