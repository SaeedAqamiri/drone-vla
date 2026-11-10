#!/usr/bin/env python3
"""Diagnostic map-only visual anchors for previously recorded GIM matches.

The anchors are distinctive SIFT corners on the satellite map, not semantic
building detections. They are selected without query images or ground truth.
For each recorded best tile we re-run GIM and count geographically separated
anchors supported by homography-consistent correspondences. The outcome is
retrospective only: this script does not change or validate flight decisions.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navlab.lab.deep import _read_rgb
from navlab.lab.remote_gim import RemoteGIMMatcher
from navlab.lab.tiles import TileStore
from navlab.lab.landmarks import map_anchors, anchor_support



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids',nargs='+',default=['02_0072','03_0031','06_0180'])
    parser.add_argument('--out',default='outputs/navlab/decision_examples/landmark_diagnostic.json')
    parser.add_argument('--study-root',default='outputs/navlab/level_holdout')
    parser.add_argument('--run-name',default='gim_dkm_c2_cov10_768')
    args=parser.parse_args()
    matcher=RemoteGIMMatcher('ava@172.16.20.117',min_coverage=.1)
    results=[]
    try:
        for query_id in args.ids:
            region=query_id[:2]; study=ROOT/args.study_root/region
            rows=list(csv.DictReader((study/args.run_name/'frames.csv').open()))
            row=next(r for r in rows if r['query_id']==query_id)
            if row['accepted']!='True':
                raise ValueError(f'{query_id} is not a recorded accepted fix')
            query=next(q for q in json.loads((study/'queries.json').read_text())['queries'] if q['id']==query_id)
            store=TileStore(study/'tiles')
            tile=next(t for t in store.tiles if t['id']==row['best_tile'])
            ref=_read_rgb(store.path_for(tile),768)[0]
            image=cv2.imread(query['path'],cv2.IMREAD_COLOR)
            angle=int(row['match_query_rotation_deg'])
            if angle:
                image=cv2.rotate(image,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
            anchors=map_anchors(ref)
            outcome,ref_pts,query_pts=matcher.match_with_points(store.path_for(tile),image)
            if outcome.execution_error or outcome.homography is None:
                raise RuntimeError(outcome.error)
            support, supported = anchor_support(anchors, ref_pts, query_pts, outcome.homography)
            results.append({'query_id':query_id,'recorded_error_m':float(row['error_m']),
                            'reference_tile':row['best_tile'],'map_only_anchors':len(anchors),
                            'supported_anchors_at_least_8_points':supported,
                            'anchor_support_counts':support,'anchor_coordinates_768px':anchors.tolist(),
                            'recomputed_inliers':outcome.inliers,'explanation':
                            'SIFT map corners; GIM points within 25 resized-map pixels and 5 query reprojection pixels. Not semantically identified objects; dense points are correlated.'})
            print(query_id,'label error',row['error_m'],'anchors',results[-1]['supported_anchors_at_least_8_points'],flush=True)
    finally:
        matcher.close()
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
