import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import scripts.localization_review as review
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

    def test_flux_requires_explicit_source_and_real_flux_model(self):
        with tempfile.TemporaryDirectory(dir=review.ROOT) as tmp:
            root=Path(tmp); store=root/'store'; store.mkdir()
            source=store/'source.png'; generated=store/'out.png'
            cv2.imwrite(str(source),np.full((4,4,3),40,np.uint8))
            cv2.imwrite(str(generated),np.full((4,4,3),120,np.uint8))
            manifest={'store':str(store),'backend':'sglang','tiles':[
                {'id':'generated1','source_tile':'source1','source':str(source),
                 'image':'out.png','meta':{'model':'flux-3-image-klein'}},
                {'id':'fake','source_tile':'source2','source':str(source),
                 'image':'out.png','meta':{'model':'procedural dummy'}},
            ]}
            (root/'references_nadir.json').write_text(json.dumps(manifest))
            with patch.object(review,'OUTPUT',root):
                cat=Catalog()
            run={'type':'uav','store':type('Store',(),{'root':store})(),
                 'tiles':{'source1':{'id':'source1'},
                          'generated1':{'id':'generated1','source_tile':'source1'},
                          'source2':{'id':'source2'}}}
            self.assertIsNotNone(cat.flux_for(run,{'id':'source1','dynamic':False}))
            self.assertIsNotNone(cat.flux_for(run,{'id':'generated1','dynamic':False}))
            self.assertIsNone(cat.flux_for(run,{'id':'source2','dynamic':False}))
            self.assertIsNone(cat.flux_for(run,{'id':'source1','dynamic':True}))
            pair=cat.flux_for(run,{'id':'source1','dynamic':False})
            self.assertEqual(int(cv2.imread(str(pair['source']))[0,0,0]),40)
            self.assertEqual(int(cv2.imread(str(pair['image']))[0,0,0]),120)

if __name__=='__main__': unittest.main()
