#!/usr/bin/env python3
"""Render two recorded visual-localization decisions; correspondences are re-run.

Candidate IDs are reconstructed from the logged prior and replayed uncertainty.
Frame outcomes and coordinates come from the recorded benchmark, not the re-run.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from navlab.lab.candidates import Prior
from navlab.lab.deep import _read_rgb
from navlab.lab.remote_gim import RemoteGIMMatcher
from navlab.lab.tiles import TileStore

ROOT = Path('outputs/navlab/level_holdout')
OUT = Path('outputs/navlab/decision_examples')
OUT.mkdir(parents=True, exist_ok=True)
CASES = [('02', 1), ('03', 0), ('06', 6)]


def decision(region, selected):
    study = ROOT / region
    run = study / 'gim_dkm_c2_cov10_768'
    init = json.loads((study / 'initial_prior.json').read_text())
    queries = {q['id']: q for q in json.loads((study / 'queries.json').read_text())['queries']}
    rows = list(csv.DictReader((run / 'frames.csv').open()))
    store = TileStore(study / 'tiles')
    east, north = store.to_local(init['lon'], init['lat'])
    prior = Prior(east=east, north=north, vx=init['vx'], vy=init['vy'],
                  sigma=init['sigma_m'], timestamp=init['timestamp'], valid=True)
    for i, r in enumerate(rows):
        timestamp = float(r['timestamp'])
        x, y, sigma = prior.predict(timestamp)
        assert abs(x-float(r['prior_east'])) < .01 and abs(y-float(r['prior_north'])) < .01
        radius = min(800., 60. + 3. * sigma)
        tiles = store.candidates(x, y, radius, limit=2)
        assert len(tiles) == int(r['candidates'])
        if i == selected:
            assert r['best_tile'] in [t['id'] for t in tiles]
            return r, queries[r['query_id']], store, tiles, radius
        if r['accepted'] == 'True':
            prior.update(float(r['best_east']), float(r['best_north']), timestamp,
                         sigma=6., max_speed=80.)
        else:
            prior.decay(timestamp)
    raise IndexError(selected)


def write_text(img, text, org, scale=.65, color=(235, 235, 235), thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_marker(img, x, y, color, label, radius=7, label_offset=(10, -9)):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (x,y), radius+2, (0,0,0), 4, cv2.LINE_AA)
        cv2.circle(img, (x,y), radius, color, 3, cv2.LINE_AA)
        label_pos = (x+label_offset[0], y+label_offset[1])
        write_text(img, label, label_pos, .55, (0,0,0), 4)
        write_text(img, label, label_pos, .55, color, 2)


def render(matcher, region, index):
    r, query, store, candidates, radius = decision(region, index)
    chosen = next(t for t in candidates if t['id'] == r['best_tile'])
    other = next(t for t in candidates if t['id'] != r['best_tile'])
    outcome, rp, qp = matcher.match_with_points(store.path_for(chosen), query['path'])
    if outcome.execution_error or outcome.homography is None:
        raise RuntimeError(outcome.error)
    query_img = _read_rgb(query['path'],768)[0]
    best_img = _read_rgb(store.path_for(chosen),768)[0]
    other_img = _read_rgb(store.path_for(other),768)[0]
    # The returned points are in 768px-resized image coordinates. Overlay a
    # small spatially distributed subset of homography-consistent re-run pairs.
    projected = cv2.perspectiveTransform(np.float32(rp).reshape(-1,1,2), outcome.homography).reshape(-1,2)
    keep = np.flatnonzero(np.linalg.norm(projected-qp,axis=1) < 3)
    selected = []
    for j in keep:
        pt = qp[j]
        if all(np.linalg.norm(pt-qp[k]) > 105 for k in selected):
            selected.append(j)
        if len(selected) == 12:
            break
    query_height, query_width = query_img.shape[:2]
    image_corners = np.float32([[[0, 0], [query_width-1, 0],
                                 [query_width-1, query_height-1], [0, query_height-1]]])
    corners_in_tile = cv2.perspectiveTransform(image_corners, np.linalg.inv(outcome.homography))[0]
    corners_m = corners_in_tile * (500.0 / best_img.shape[1])
    edges_m = [float(np.linalg.norm(corners_m[(i+1)%4] - corners_m[i])) for i in range(4)]
    footprint_m2 = abs(float(cv2.contourArea(corners_m)))
    original = cv2.imread(query['path'], cv2.IMREAD_COLOR)
    original_pixels = original.shape[0] * original.shape[1]
    h, w = 900, 2190
    canvas = np.full((h,w,3), (25,27,32), np.uint8)
    write_text(canvas, f"UAV-VisLoc {r['query_id']}  |  recorded result: {float(r['error_m']):.1f} m error", (30,45), .95, (255,255,255), 2)
    write_text(canvas, 'DRONE IMAGE', (30,98), .72, (230,230,230), 2)
    write_text(canvas, 'SELECTED SATELLITE TILE', (760,98), .72, (230,230,230), 2)
    write_text(canvas, 'OTHER CANDIDATE TILE', (1490,98), .72, (230,230,230), 2)
    # Preserve proportions and maintain a common tile size.
    def put_image(image, x, y, maxw=680, maxh=650):
        factor = min(maxw/image.shape[1],maxh/image.shape[0])
        ww, hh = round(image.shape[1]*factor),round(image.shape[0]*factor)
        resized = cv2.resize(image,(ww,hh),interpolation=cv2.INTER_AREA)
        canvas[y:y+hh,x:x+ww] = resized
        return factor
    qx,qy=30,130
    bx,by=760,130
    ox,oy=1490,130
    qscale=put_image(query_img,qx,qy)
    bscale=put_image(best_img,bx,by)
    oscale=put_image(other_img,ox,oy)
    # Render lines before markers; colored dots are re-run point examples.
    colors=[(0,220,255),(255,175,0),(180,255,50),(250,100,250),(60,140,255),(240,240,80)]
    for k,j in enumerate(selected):
        color=colors[k%len(colors)]
        a=tuple(np.round(qp[j]*qscale+[qx,qy]).astype(int))
        b=tuple(np.round(rp[j]*bscale+[bx,by]).astype(int))
        cv2.line(canvas,a,b,color,1,cv2.LINE_AA)
        for p in (a,b):
            cv2.circle(canvas,p,6,(0,0,0),3,cv2.LINE_AA)
            cv2.circle(canvas,p,4,color,-1,cv2.LINE_AA)
    for tile,tx,ty,scale in [(chosen,bx,by,bscale),(other,ox,oy,oscale)]:
        bounds=tile['bounds']; px_per_m=768*scale/500.
        for keyx,keyy,color,label,marker_radius,offset in [('prior_east','prior_north',(255,210,10),'P',12,(12,-12)),('best_east','best_north',(40,40,255),'E',7,(10,-9)),('truth_east','truth_north',(30,255,30),'T',5,(-20,18))]:
            x=round(tx+(float(r[keyx])-bounds['min_east'])*px_per_m)
            y=round(ty+(bounds['max_north']-float(r[keyy]))*px_per_m)
            if tx <= x < tx+768*scale and ty <= y < ty+768*scale:
                draw_marker(canvas,x,y,color,label,marker_radius,offset)
    write_text(canvas, f"{chosen['id']}  |  500 m x 500 m", (760,825), .62)
    write_text(canvas, f"{other['id']}  |  500 m x 500 m", (1490,825), .62)
    write_text(canvas, 'T=dataset position  E=estimated image center  P=predicted position', (30,865), .65, (240,240,240), 2)
    a,b=candidates
    ab,bb=a['bounds'],b['bounds']
    overlap=max(0,min(ab['max_east'],bb['max_east'])-max(ab['min_east'],bb['min_east']))*max(0,min(ab['max_north'],bb['max_north'])-max(ab['min_north'],bb['min_north']))
    union=500*500*2-overlap
    path=OUT / f'{region}_{r["query_id"]}_decision.jpg'
    cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
    return {'query':r['query_id'],'figure':str(path),'recorded_error_m':float(r['error_m']),
            'recorded_inliers':int(r['match_inliers']),'recorded_coverage':float(r['match_coverage']),
            'recorded_prior_error_m':float(r['prior_error_m']),'recorded_latency_ms':float(r['latency_ms']),
            'candidate_ids_reconstructed':[t['id'] for t in candidates], 'selection_radius_m':radius,
            'tile_area_m2':250000,'union_area_m2':union,'rerun_inliers':outcome.inliers,
            'displayed_correspondences':len(selected),'altitude_metadata_m':query['altitude'],
            'estimated_image_footprint': {
                'area_m2':footprint_m2, 'edge_lengths_m':edges_m,
                'original_camera_gsd_geometric_mean_m_per_pixel':(footprint_m2/original_pixels)**.5,
                'native_map_gsd_m_per_pixel':store.manifest['native_gsd_m'],
                'map_after_match_resize_m_per_pixel':500.0 / best_img.shape[1],
                'corners_in_resized_map_pixels':corners_in_tile.tolist(),
                'note':'Planar homography estimate; extrapolated wherever corners fall outside map tile; not camera calibration.'}}


if __name__ == '__main__':
    matcher=RemoteGIMMatcher('ava@172.16.20.117',min_coverage=.1)
    try:
        examples=[render(matcher,*case) for case in CASES]
    finally:
        matcher.close()
    (OUT/'examples.json').write_text(json.dumps(examples,indent=2))
    print(json.dumps(examples,indent=2))
