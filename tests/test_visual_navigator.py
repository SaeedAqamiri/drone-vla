import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from navlab.lab.candidates import Prior
from navlab.lab.navigator import VisualNavigator
from navlab.lab.runner import RunnerConfig
from navlab.lab.synthetic import make_queries, make_scene
from navlab.lab.tiles import TileStore, build_tiles_along_route
from navlab.lab.types import MatchOutcome


class FakeMatcher:
    def __init__(self, matrix=None):
        self.matrix = np.eye(3) if matrix is None else matrix

    def match(self, ref, frame):
        assert isinstance(frame, np.ndarray)
        return MatchOutcome(str(ref), "<array>", "fake", 50, .9, .5, .1, 1., 0., 1., True,
                            homography=self.matrix)


class RejectMatcher:
    def match(self, ref, frame):
        return MatchOutcome(str(ref), "<array>", "fake", 0, 0, 0, None, None, None, 1, False)


class ErrorMatcher:
    def match(self, ref, frame):
        return MatchOutcome(str(ref), "<array>", "fake", 0, 0, 0, None, None, None, 1,
                            False, error="OutOfMemoryError: test", execution_error=True)


class PartialErrorMatcher(FakeMatcher):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def match(self, ref, frame):
        self.calls += 1
        return super().match(ref, frame) if self.calls == 1 else ErrorMatcher().match(ref, frame)


class RecordingMatcher(FakeMatcher):
    def match(self, ref, frame):
        self.shape = frame.shape
        return super().match(ref, frame)


class RotatedMatcher(FakeMatcher):
    def match(self, ref, frame):
        if frame[-1, -1, 0] != 255:
            return RejectMatcher().match(ref, frame)
        return super().match(ref, frame)


class VisualNavigatorTest(unittest.TestCase):
    def test_incremental_confirmation_and_out_of_bounds_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = make_scene(root, size=1024, seed=2, gsd_m=.5)
            queries = make_queries(root, scene, count=3, seed=2, altitude_m=250,
                                   query_size=(512, 512))
            build_tiles_along_route(scene["path"], root / "tiles",
                [(q.lon, q.lat) for q in queries], tile_ground_m=400, tile_px=512, stride_m=400)
            store = TileStore(root / "tiles")
            store.tiles = store.tiles[:1]
            store._east = np.array([store.tiles[0]["east"]])
            store._north = np.array([store.tiles[0]["north"]])
            tile = store.tiles[0]
            prior = Prior(east=tile["east"], north=tile["north"], timestamp=0, sigma=5, valid=True)
            config = RunnerConfig(max_dim=512, scale_tolerance=0, confirmation=3,
                                  confirmation_radius_m=2, base_radius=30)
            nav = VisualNavigator(store, prior, config=config, matcher=FakeMatcher())
            frame = np.zeros((512, 512, 3), np.uint8)
            self.assertEqual(nav.process_frame(frame, 1).reason, "pending_confirmation")
            self.assertEqual(nav.process_frame(frame, 2).reason, "pending_confirmation")
            result = nav.process_frame(frame, 3)
            self.assertTrue(result.accepted)
            self.assertAlmostEqual(result.east_m, tile["east"], delta=1)
            with self.assertRaises(ValueError):
                nav.process_frame(frame, 3)
            bad = np.array([[1., 0, 10000], [0, 1., 0], [0, 0, 1.]])
            nav = VisualNavigator(store, prior, config=config, matcher=FakeMatcher(bad))
            self.assertFalse(nav.process_frame(frame, 1).accepted)
            nav = VisualNavigator(store, prior, config=config,
                                  matcher=FakeMatcher(np.zeros((3, 3))))
            self.assertFalse(nav.process_frame(frame, 1).accepted)
            nav = VisualNavigator(store, prior, config=config, matcher=RejectMatcher())
            first = nav.process_frame(frame, 1, motion_delta_m=(10, -3), motion_sigma_m=1)
            second = nav.process_frame(frame, 2, motion_delta_m=(10, -3), motion_sigma_m=1)
            self.assertFalse(first.accepted)
            self.assertAlmostEqual(second.prior_east_m, tile["east"] + 20)
            self.assertAlmostEqual(second.prior_north_m, tile["north"] - 6)
            self.assertGreater(second.prior_sigma_m, 5)
            nav = VisualNavigator(store, prior, config=config, matcher=ErrorMatcher())
            failed = nav.process_frame(frame, 1)
            self.assertFalse(failed.accepted)
            self.assertEqual(failed.reason, "matcher_error")
            self.assertIn("OutOfMemoryError", failed.execution_errors[0])
            config.confirmation = 1
            # An earlier valid candidate is not a valid ranking when a later
            # candidate could not be evaluated due to an execution error.
            store.tiles.append({**store.tiles[0], "id": "second_tile"})
            store._east = np.append(store._east, tile["east"])
            store._north = np.append(store._north, tile["north"])
            nav = VisualNavigator(store, prior, config=config, matcher=PartialErrorMatcher())
            result = nav.process_frame(frame, 1)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, "matcher_error")
            store.tiles.pop()
            store._east = store._east[:1]
            store._north = store._north[:1]
            nav = VisualNavigator(store, prior, config=config, matcher=RotatedMatcher(), rotations=(0, 180))
            frame[0, 0, 0] = 255
            result = nav.process_frame(frame, 1)
            self.assertTrue(result.accepted)
            self.assertEqual(result.query_rotation_deg, 180)

            shifted = Prior(east=tile["east"] + 20, north=tile["north"],
                            timestamp=0, sigma=5, valid=True)
            config.max_jump_m, config.jump_sigma_scale = 5, 1
            nav = VisualNavigator(store, shifted, config=config, matcher=FakeMatcher())
            self.assertFalse(nav.process_frame(frame, 1).accepted)
            config.jump_sigma_scale = 4
            nav = VisualNavigator(store, shifted, config=config, matcher=FakeMatcher())
            self.assertTrue(nav.process_frame(frame, 1).accepted)
            config.max_jump_m, config.jump_sigma_scale = 250, 2

            recorder = RecordingMatcher()
            nav = VisualNavigator(store, prior, config=config, matcher=recorder)
            # Reference has 400 / 512 metres per pixel; half that value for
            # the input image should resize it to half of each dimension.
            result = nav.process_frame(frame, 1, camera_gsd_m=(400 / 512) / 2)
            self.assertTrue(result.accepted)
            self.assertEqual(recorder.shape[:2], (256, 256))
            with self.assertRaisesRegex(ValueError, "GSD"):
                nav.process_frame(frame, 2, camera_gsd_m=0)

    def test_unchecked_generated_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = make_scene(root, size=512, seed=2, gsd_m=.5)
            queries = make_queries(root, scene, count=3, seed=2, altitude_m=250,
                                   query_size=(512, 512))
            build_tiles_along_route(scene["path"], root / "tiles",
                [(q.lon, q.lat) for q in queries], tile_ground_m=400, tile_px=512, stride_m=400)
            store = TileStore(root / "tiles")
            ref = root / "refs.json"
            ref.write_text(json.dumps({"tiles": [{**store.tiles[0], "kind": "generated"}]}))
            prior = Prior(east=0, north=0, timestamp=0, valid=True)
            with self.assertRaisesRegex(ValueError, "geometry validation"):
                VisualNavigator(store, prior, references=ref, matcher=FakeMatcher())


if __name__ == "__main__":
    unittest.main()
