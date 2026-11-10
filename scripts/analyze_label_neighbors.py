#!/usr/bin/env python3
"""Retrospective nearest-label diagnostic; never use as online navigation.

Searches multiple labels including future ones, so selected residuals are
optimistically biased and are not validation or proof of a timing error.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navlab.lab.sequences import uav_visloc_queries
from navlab.lab.tiles import TileStore


def nearest_index(points, x, y, index, window):
    if not points or index < 0 or index >= len(points) or window < 0:
        raise ValueError('invalid sequence, index or window')
    indices=range(max(0,index-window),min(len(points),index+window+1))
    selected=min(indices,key=lambda j:math.hypot(points[j][0]-x,points[j][1]-y))
    return selected, math.hypot(points[selected][0]-x,points[selected][1]-y)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'data/raw/uav_visloc/UAV_VisLoc_dataset')
    p.add_argument('--window',type=int,default=12)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.window<0: p.error('window must be nonnegative')
    audit=json.loads(args.audit.read_text()); studies=[]
    for study in audit['studies']:
        path=Path(study['study']);region=path.name
        store=TileStore(path/'tiles')
        queries=uav_visloc_queries(args.dataset,region,near_nadir_deg=180)
        positions=[store.to_local(q.lon,q.lat) for q in queries]
        by_id={q.id:i for i,q in enumerate(queries)}
        results=[]
        for frame in study['frames']:
            if not frame['matched'] or frame['query_id'] not in by_id: continue
            i=by_id[frame['query_id']]
            j,error=nearest_index(positions,frame['east'],frame['north'],i,args.window)
            results.append({'query_id':frame['query_id'],'original_label_error_m':frame['error_m'],
                            'nearest_label_id':queries[j].id,'nearest_label_index_offset':j-i,
                            'nearest_label_time_offset_s':queries[j].timestamp-queries[i].timestamp,
                            'nearest_label_distance_m':error})
        studies.append({'study':str(path),'frames':results,'summary':{'matched_frames':len(results),
                        'median_original_error_m':statistics.median(x['original_label_error_m'] for x in results) if results else None,
                        'median_nearest_label_distance_m':statistics.median(x['nearest_label_distance_m'] for x in results) if results else None,
                        'median_nearest_index_offset':statistics.median(x['nearest_label_index_offset'] for x in results) if results else None}})
    result={'note':'ORACLE diagnostic: map selection and nearest neighbouring labels use ground truth, including future labels. Searching 25 labels biases the minimum. No fixed timing correction or mission probability is established.',
            'neighbor_window_frames':args.window,'studies':studies}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    for s in studies: print(s['study'],s['summary'])

if __name__=='__main__': main()
