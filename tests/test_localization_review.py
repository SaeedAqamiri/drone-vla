import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.localization_review import Catalog, replay_candidates, safe_path


class DummyStore:
    def to_local(self, lon, lat):
        return lon, lat
    def candidates(self, x, y, radius, limit, kind):
        return [{'id':'tile','bounds':{'min_east':-100,'max_east':100,
                                       'min_north':-100,'max_north':100}}]


class ReviewTest(unittest.TestCase):
    def test_reconstruction_fails_closed_on_unreproducible_prior(self):
        store = DummyStore()
        summary = {'config':{'initial_prior':{'lat':0,'lon':0,'timestamp':0,'vx':1,'vy':0,
                                             'sigma_m':5},'max_candidates':1}}
        rows=[{'query_id':'q1','timestamp':'1','prior_east':'1','prior_north':'0',
               'candidates':'1','best_tile':'tile','accepted':'True',
               'best_east':'1','best_north':'0'},
              {'query_id':'q2','timestamp':'2','prior_east':'2','prior_north':'0',
               'candidates':'1','best_tile':'tile','accepted':'False'}]
        self.assertEqual(len(replay_candidates(rows,summary,store,{},{})),2)
        rows[1]['prior_east']='20'
        self.assertEqual(replay_candidates(rows,summary,store,{},{}),{})
        self.assertIsNone(safe_path('/etc/passwd'))

    def test_ortholoc_displays_actual_scaled_input_and_map_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.npz'
            np.savez(path,image_query=np.zeros((80,100,3),np.uint8),image_dop=np.zeros((40,50,3),np.uint8))
            data={'success':True,'query_scale':.25,'horizontal_error_m':2.,'matcher':'sift',
                  'map_shape':[40,50,3], 'map_scale_m_per_px':[.2,-.2]}
            cat=Catalog.__new__(Catalog)
            cat.runs={'sample':{'type':'ortholoc','rows':[{'query_id':'q','sample':str(path),'data':data}]}}
            frame=cat.frame('sample',0)
            self.assertEqual(frame['threshold_m'],5)
            self.assertEqual(frame['map_size_m'],[10.,8.])
            self.assertEqual(cat.image('sample',0,'query').shape,(20,25,3))
            self.assertEqual(cat.image('sample',0,'map').shape,(40,50,3))

if __name__=='__main__': unittest.main()
