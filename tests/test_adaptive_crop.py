import tempfile
import unittest
from pathlib import Path

import numpy as np

from navlab.lab.adaptive_crop import AdaptiveCrop
from navlab.lab.candidates import Prior
from navlab.lab.navigator import VisualNavigator
from navlab.lab.runner import RunnerConfig
from navlab.lab.synthetic import make_queries, make_scene
from navlab.lab.tiles import TileStore, build_tiles_along_route
from navlab.lab.types import MatchOutcome


class IdentityMatcher:
    def match(self, reference, frame):
        assert isinstance(reference, np.ndarray)
        return MatchOutcome('<map>', '<frame>', 'test', 50, .9, .5, .1, 1, 0, 1, True,
                            homography=np.eye(3))


class AdaptiveCropTest(unittest.TestCase):
    def test_projected_crop_georeferencing_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = make_scene(root, size=1024, seed=2, gsd_m=.5)
            queries = make_queries(root, scene, count=2, seed=2, altitude_m=250,
                                   query_size=(512, 512))
            build_tiles_along_route(scene['path'], root/'tiles', [(q.lon,q.lat) for q in queries],
                                    tile_ground_m=400, tile_px=512, stride_m=400)
            store = TileStore(root/'tiles')
            center = store.to_local(store.manifest['center_lon'], store.manifest['center_lat'])
            cropper = AdaptiveCrop(store, footprint_width_m=100, uncertainty_scale=2,
                                   max_side_m=300, pixels=512)
            try:
                tile = cropper.crop(*center, 10, (512,512))
                self.assertIsNotNone(tile)
                self.assertAlmostEqual(tile['side_m'], 100 * 2**.5 + 40)
                self.assertAlmostEqual((tile['bounds']['min_east']+tile['bounds']['max_east'])/2, center[0])
                self.assertEqual(tile['_image'].shape, (512,512,3))
                self.assertGreater(tile['valid_fraction'], .99)
                self.assertIsNone(cropper.crop(*center, 100, (512,512)))
                cropper.horizontal_fov_deg = 90
                by_altitude = cropper.crop(*center, 10, (512,512), altitude_agl_m=50)
                self.assertAlmostEqual(by_altitude['side_m'], 100 * 2**.5 + 40)
                self.assertTrue(cropper.last_diagnostic['uses_altitude_agl'])
                cropper.horizontal_fov_deg = None
                self.assertIsNone(cropper.crop(center[0]+10000,center[1], 10, (512,512)))
                prior = Prior(east=center[0], north=center[1], timestamp=0, sigma=10, valid=True)
                nav = VisualNavigator(store, prior, config=RunnerConfig(max_dim=512, scale_tolerance=0),
                                      matcher=IdentityMatcher(), cropper=cropper)
                result = nav.process_frame(np.zeros((512,512,3),np.uint8), 1)
                self.assertTrue(result.accepted)
                self.assertAlmostEqual(result.east_m, center[0], delta=.01)
                self.assertAlmostEqual(result.north_m, center[1], delta=.01)
                self.assertEqual(len(result.candidate_bounds), 1)
                class SpyCropper:
                    sigma_growth_mps = 3.0
                    def crop(self, east, north, sigma, shape, **kwargs):
                        self.sigma = sigma
                        return None
                spy = SpyCropper()
                nav = VisualNavigator(store, prior, config=RunnerConfig(max_dim=512),
                                      matcher=IdentityMatcher(), cropper=spy)
                self.assertEqual(nav.process_frame(np.zeros((512,512,3),np.uint8), 5).reason, 'no_candidates')
                self.assertEqual(spy.sigma, 25.0)
                class ErrorCropper(SpyCropper):
                    def crop(self, *args, **kwargs):
                        raise OSError('map unavailable')
                nav = VisualNavigator(store, prior, config=RunnerConfig(max_dim=512),
                                      matcher=IdentityMatcher(), cropper=ErrorCropper())
                result = nav.process_frame(np.zeros((512,512,3),np.uint8), 5)
                self.assertEqual(result.reason, 'crop_error')
                self.assertIn('map unavailable', result.execution_errors[0])
                class MissingCenterCropper(SpyCropper):
                    def crop(self, *args, **kwargs):
                        return {**tile, '_valid_mask': np.zeros((512,512), np.uint8)}
                nav = VisualNavigator(store, prior, config=RunnerConfig(max_dim=512),
                                      matcher=IdentityMatcher(), cropper=MissingCenterCropper())
                self.assertFalse(nav.process_frame(np.zeros((512,512,3),np.uint8), 5).accepted)
            finally:
                cropper.close()


if __name__ == '__main__':
    unittest.main()
