#!/usr/bin/env python3
"""Oracle-only diagnostic of image-centre versus UAV-VisLoc labels.

Reference labels are deliberately used to select nearby map tiles. This is
NOT a navigation benchmark, must never be used online, and cannot estimate
mission success. Signed along-track offsets are descriptive, not a clock fix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navlab.lab.remote_gim import RemoteGIMMatcher
from navlab.lab.runner import LabRunner, RunnerConfig
from navlab.lab.sequences import load_queries
from navlab.lab.tiles import TileStore


def motion_components(dx, dy, start_xy, end_xy, elapsed_s):
    """Resolve an offset along a measured label track, not an inferred clock bias."""
    if elapsed_s <= 0:
        return 0.0, None, None, None
    vx=(end_xy[0]-start_xy[0])/elapsed_s
    vy=(end_xy[1]-start_xy[1])/elapsed_s
    speed=math.hypot(vx,vy)
    if speed <= 0:
        return 0.0, None, None, None
    along=(dx*vx+dy*vy)/speed
    cross=(-dx*vy+dy*vx)/speed
    return speed, along, cross, along/speed


def run_study(study, matcher, *, limit):
    store=TileStore(study/'tiles')
    queries,_=load_queries(study/'queries.json')
    queries=queries[:limit]
    truth=[store.to_local(q.lon,q.lat) for q in queries]
    helper=LabRunner(store,[],RunnerConfig(max_dim=768,scale_tolerance=0))
    frames=[]
    for index,query in enumerate(queries):
        image=cv2.imread(str(query.path),cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(query.path)
        # Oracle only: in deployment this true location is unavailable.
        x,y=truth[index]
        tiles=store.candidates(x,y,0,limit=2)
        candidates=[]
        for angle in (0,90,180,270):
            rotated=image if angle==0 else cv2.rotate(image,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
            for tile in tiles:
                out=matcher.match(store.path_for(tile),rotated)
                if out.execution_error: raise RuntimeError(out.error)
                if not out.success or out.homography is None: continue
                east,north=helper._refine_position(tile,out,rotated.shape[:2])
                b=tile['bounds']
                if not (math.isfinite(east) and math.isfinite(north) and b['min_east']<=east<=b['max_east'] and b['min_north']<=north<=b['max_north']): continue
                candidates.append((helper._score(out),tile,out,angle,east,north))
            if candidates: break
        row={'query_id':query.id,'timestamp':query.timestamp,'oracle_tile_ids':[t['id'] for t in tiles],
             'truth_east':x,'truth_north':y,'matched':bool(candidates)}
        if candidates:
            _,tile,out,angle,east,north=max(candidates,key=lambda a:a[0])
            dx,dy=east-x,north-y
            a=max(0,index-1);b=min(len(queries)-1,index+1)
            dt=queries[b].timestamp-queries[a].timestamp
            speed,along,cross,equivalent_time=motion_components(dx,dy,truth[a],truth[b],dt)
            row.update({'best_tile':tile['id'],'east':east,'north':north,'error_m':math.hypot(dx,dy),
                        'east_offset_m':dx,'north_offset_m':dy,'along_track_m':along,'cross_track_m':cross,
                        'label_speed_mps':speed,'equivalent_along_track_time_s':equivalent_time,
                        'inliers':out.inliers,'coverage':out.coverage,'rotation_deg':angle})
        frames.append(row)
        print(f'{study.name} {query.id}: {row.get("error_m", "no match")}',flush=True)
    usable=[r for r in frames if r['matched']]
    def median(name):
        values=[r[name] for r in usable if r.get(name) is not None]
        return float(np.median(values)) if values else None
    return {'study':str(study),'frames':frames,'summary':{'frames':len(frames),'matched':len(usable),
            'median_error_m':median('error_m'),'median_east_offset_m':median('east_offset_m'),
            'median_north_offset_m':median('north_offset_m'),'median_along_track_m':median('along_track_m'),
            'median_cross_track_m':median('cross_track_m'),
            'median_equivalent_along_track_time_s':median('equivalent_along_track_time_s')}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--remote-gim-host',default='ava@172.16.20.117')
    p.add_argument('--limit',type=int,default=10)
    args=p.parse_args()
    if args.limit<1: p.error('limit must be positive')
    cv2.setNumThreads(2); t=time.monotonic()
    matcher=RemoteGIMMatcher(args.remote_gim_host,min_coverage=.1)
    try: results=[run_study(Path(s),matcher,limit=args.limit) for s in args.studies]
    finally: matcher.close()
    payload={'note':'ORACLE ground truth chooses map tiles. Descriptive retrospective geometry only; not a navigation or flight-success test. Along-track-equivalent time is not proof of timestamp error.',
             'elapsed_s':time.monotonic()-t,'studies':results}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2))
    print(json.dumps([r['summary'] for r in results],indent=2))

if __name__=='__main__': main()
