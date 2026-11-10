#!/usr/bin/env python3
"""Local, read-only browser for recorded visual-localization experiments.

Run from the repository root with env_flight/bin/python. The server binds only
127.0.0.1. A GIM correspondence visualization is an optional re-run, and is
never represented as the recorded matching decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
from pathlib import Path
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
import sys

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navlab.lab.candidates import Prior
from navlab.lab.deep import _read_rgb
from navlab.lab.tiles import TileStore

STATIC = ROOT / 'navlab/review'
OUTPUT = ROOT / 'outputs/navlab'


def safe_path(path):
    p = Path(path)
    p = (ROOT / p if not p.is_absolute() else p).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        return None
    return p


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def flag(value):
    return str(value).lower() in ('true', '1')


def query_index():
    index = {}
    for path in OUTPUT.rglob('*queries*.json'):
        try:
            payload = json.loads(path.read_text())
            for item in payload.get('queries', []):
                if safe_path(item.get('path', '')):
                    index.setdefault(item['id'], {})[item['path']] = item
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
    return index


def get_queries(run_path, all_queries, store):
    # Closest preparation file takes precedence over global IDs; this matters
    # when synthetic datasets reuse identifiers such as syn_0000.
    for ancestor in [run_path.parent, *run_path.parents]:
        if not ancestor.is_relative_to(OUTPUT):
            break
        p = ancestor / 'queries.json'
        if p.is_file():
            try:
                data = json.loads(p.read_text())
                return {q['id']: q for q in data.get('queries', [])}
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
    resolved = {}
    # Synthetic datasets may reuse the same identifier. Prefer the one inside
    # the same prepared scene as this run's map; never guess on a tied score.
    for key, items in all_queries.items():
        choices = list(items.values())
        if len(choices) == 1:
            resolved[key] = choices[0]
            continue
        scored = []
        for item in choices:
            path = safe_path(item.get('path', ''))
            if path is None:
                continue
            score = len(Path(__import__('os').path.commonpath([str(path), str(store.root.resolve())])).parts)
            scored.append((score, item))
        scored.sort(key=lambda x:x[0], reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            resolved[key] = scored[0][1]
    return resolved


def replay_candidates(rows, summary, store, query_map, details):
    """Reconstruct only when timestamped prior reproduces recorded inputs."""
    cfg = summary.get('config', {})
    seed = cfg.get('initial_prior')
    if (not seed or cfg.get('oracle_motion') or cfg.get('visual_propagation')
            or summary.get('references')):
        return {}
    try:
        east, north = store.to_local(seed['lon'], seed['lat'])
        prior = Prior(east=east, north=north, vx=seed.get('vx', 0), vy=seed.get('vy', 0),
                      sigma=seed.get('sigma_m', 30), timestamp=seed['timestamp'], valid=True)
        found = {}
        for row in rows:
            t = float(row['timestamp']); prediction = prior.predict(t)
            x, y, sigma = prediction
            if number(row.get('prior_east')) is None or abs(x-float(row['prior_east'])) > .03 or abs(y-float(row['prior_north'])) > .03:
                return {}
            idx = row['query_id']
            if cfg.get('adaptive_footprint_width_m') is not None:
                old = details.get(idx, {}).get('bounds')
                if old is None and row.get('best_tile') == 'adaptive_crop':
                    q = query_map.get(idx); img = cv2.imread(str(safe_path(q['path']))) if q and safe_path(q.get('path','')) else None
                    if img is not None:
                        h,w=img.shape[:2]; growth=cfg.get('crop_sigma_growth_mps',0)
                        crop_sigma=sigma+growth*max(0,t-float(prior.timestamp))
                        side=cfg['adaptive_footprint_width_m']*math.hypot(w,h)/w + 2*cfg.get('crop_uncertainty_scale',2)*crop_sigma
                        old={'min_east':x-side/2,'max_east':x+side/2,'min_north':y-side/2,'max_north':y+side/2}
                if old is not None:
                    found[idx]=[{'id':'adaptive_crop','bounds':old,'dynamic':True}]
            else:
                radius=min(cfg.get('max_radius',800),cfg.get('base_radius',60)+cfg.get('sigma_scale',3)*sigma)
                cand=store.candidates(x,y,radius,limit=cfg.get('max_candidates',8),kind=cfg.get('reference_kind'))
                if len(cand)==int(row.get('candidates') or 0) and (not row.get('best_tile') or row['best_tile'] in [c['id'] for c in cand]):
                    found[idx]=[{'id':c['id'],'bounds':c['bounds'],'dynamic':False} for c in cand]
            if flag(row.get('accepted')) and number(row.get('best_east')) is not None:
                prior.update(float(row['best_east']),float(row['best_north']),t,
                             sigma=cfg.get('fix_sigma',6),max_speed=cfg.get('max_speed_mps',80))
            else:
                prior.decay(t)
        return found
    except (KeyError, ValueError, TypeError, OverflowError):
        return {}


class Catalog:
    def __init__(self):
        self.runs = {}
        qindex = query_index()
        for frames_path in OUTPUT.rglob('frames.csv'):
            try:
                summary = json.loads((frames_path.parent/'summary.json').read_text())
                store_path = safe_path(Path(summary['store']['root'])/'manifest.json')
                if not store_path:
                    continue
                store = TileStore(store_path.parent)
                rows = list(csv.DictReader(frames_path.open()))
                queries = get_queries(frames_path, qindex, store)
                ref_tiles = summary.get('references') or {}
                tiles = {str(t['id']):t for t in store.tiles}
                tiles.update({str(t['id']):t for t in ref_tiles.get('tiles', [])})
                detail_path=frames_path.parent/'crop_details.json'
                details={d['query_id']:d for d in json.loads(detail_path.read_text()).get('frames',[])} if detail_path.exists() else {}
                candidates=replay_candidates(rows,summary,store,queries,details)
                ident=str(frames_path.parent.relative_to(ROOT))
                seq=summary.get('extra',{}).get('sequence',{})
                region=seq.get('map') or (rows[0]['query_id'][:2] if rows and rows[0]['query_id'][:2].isdigit() else 'سایر')
                self.runs[ident]={'id':ident,'type':'uav','region':str(region),'rows':rows,'queries':queries,
                                  'summary':summary,'tiles':tiles,'store':store,'candidates':candidates,'details':details}
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
        # OrthoLoC outputs are individual JSON files, not frames.csv runs.
        for path in (OUTPUT/'ortholoc').glob('*.json'):
            try:
                data=json.loads(path.read_text()); sample=safe_path(data.get('sample',''))
                if not sample or not data.get('matcher') or 'success' not in data:
                    continue
                ident=f"ortholoc/{data['matcher']}"
                run=self.runs.setdefault(ident,{'id':ident,'type':'ortholoc','region':'OrthoLoC',
                                                'rows':[],'summary':{'config':{'matcher':data['matcher']}}})
                run['rows'].append({'query_id':path.stem,'sample':str(sample),'data':data})
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                continue
        for run in self.runs.values():
            if run['type']=='ortholoc': run['rows'].sort(key=lambda x:x['query_id'])

    def list(self):
        result=[]
        for r in self.runs.values():
            cfg=r['summary'].get('config',{})
            result.append({'id':r['id'],'region':r['region'],'type':r['type'],
                           'matcher':cfg.get('matcher','?'),'count':len(r['rows']),
                           'oracle_motion':bool(cfg.get('oracle_motion')),
                           'adaptive':cfg.get('adaptive_footprint_width_m') is not None,
                           'landmark_gate':cfg.get('landmark_min_supported',0)})
        return sorted(result,key=lambda r:(r['region'],r['id']))

    def frame(self, run_id, index):
        run=self.runs[run_id]; row=run['rows'][index]
        if run['type']=='ortholoc':
            d=row['data']; error=number(d.get('horizontal_error_m')); accepted=bool(d['success'])
            return {'query_id':row['query_id'],'accepted':accepted,'error_m':error,'threshold_m':5,
                    'reason':d.get('reason'),'inliers':d.get('inliers'),'latency_ms':number(d.get('elapsed_s'))*1000 if number(d.get('elapsed_s')) is not None else None,
                    'note':d.get('note'),'candidates':[], 'query_scale':number(d.get('query_scale')) or 1.0,
                    'map_size_m':[abs(d['map_scale_m_per_px'][0])*d['map_shape'][1],
                                  abs(d['map_scale_m_per_px'][1])*d['map_shape'][0]],
                    'ortholoc':True}
        accepted=flag(row.get('accepted')); err=number(row.get('error_m'))
        best=row.get('best_tile') or None
        candidates=run['candidates'].get(row['query_id'],[])
        if not candidates and best and best in run['tiles']:
            candidates=[{'id':best,'bounds':run['tiles'][best]['bounds'],'dynamic':False}]
        candidate_text='بازسازی‌شده از حالت ذخیره‌شده' if row['query_id'] in run['candidates'] else 'فقط قطعهٔ برتر ثبت شده است'
        q=run['queries'].get(row['query_id'])
        return {'query_id':row['query_id'],'accepted':accepted,'matched':flag(row.get('matched')),
                'error_m':err,'prior_error_m':number(row.get('prior_error_m')),
                'threshold_m':number(run['summary'].get('config',{}).get('max_error_m')) or 30,
                'reason':row.get('reason'),'inliers':number(row.get('match_inliers')),
                'coverage':number(row.get('match_coverage')),'latency_ms':number(row.get('latency_ms')),
                'landmark_supported':number(row.get('match_landmark_supported')),
                'rotation':int(number(row.get('match_query_rotation_deg')) or 0),
                'candidate_count':int(number(row.get('candidates')) or 0),
                'candidates':candidates,'candidate_note':candidate_text,'best_tile':best,
                'truth':[number(row.get('truth_east')),number(row.get('truth_north'))],
                'prior':[number(row.get('prior_east')),number(row.get('prior_north'))],
                'estimate':[number(row.get('best_east')),number(row.get('best_north'))],
                'altitude_metadata_m':q.get('altitude') if q else None,
                'query_available':bool(q and safe_path(q.get('path',''))),
                'crop_diagnostic':run['details'].get(row['query_id'],{}).get('crop_diagnostic'),
                'ortholoc':False}

    def image(self, run_id, index, kind, tile_id=None, angle=0):
        run=self.runs[run_id]; row=run['rows'][index]
        if run['type']=='ortholoc':
            with np.load(row['sample'],allow_pickle=False) as data:
                image=cv2.cvtColor(data['image_query' if kind=='query' else 'image_dop'],cv2.COLOR_RGB2BGR)
                if kind=='query':
                    scale=number(row['data'].get('query_scale')) or 1.0
                    if scale != 1.0:
                        image=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))),interpolation=cv2.INTER_AREA)
        elif kind=='query':
            q=run['queries'].get(row['query_id']); path=safe_path(q.get('path','')) if q else None
            if not path: raise FileNotFoundError('query not found')
            image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        elif kind=='map':
            cand=next((x for x in self.frame(run_id,index)['candidates'] if x['id']==tile_id),None)
            if cand is None: raise FileNotFoundError('map candidate not found')
            if cand['dynamic']:
                b=cand['bounds']; src=safe_path(run['store'].manifest['source'])
                if not src: raise FileNotFoundError('satellite source missing')
                with rasterio.open(src) as dataset:
                    transform=from_bounds(b['min_east'],b['min_north'],b['max_east'],b['max_north'],1024,1024)
                    with WarpedVRT(dataset,crs=run['store'].crs,transform=transform,width=1024,height=1024,resampling=Resampling.bilinear) as vrt:
                        image=cv2.cvtColor(np.moveaxis(vrt.read([1,2,3]),0,-1),cv2.COLOR_RGB2BGR)
            else:
                tile=run['tiles'][tile_id]; path=safe_path(run['store'].root/tile['image'])
                if not path: raise FileNotFoundError('recorded tile image missing')
                image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        else:
            raise ValueError('bad image kind')
        if image is None: raise ValueError('image unreadable')
        if kind=='query' and angle:
            image=cv2.rotate(image,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        return image


CATALOG = None
CATALOG_LOCK = threading.Lock()
WORKER_LOCK = threading.Lock()
GIM_HOST = None


def catalog(refresh=False):
    global CATALOG
    with CATALOG_LOCK:
        if CATALOG is None or refresh: CATALOG=Catalog()
        return CATALOG


def rematch(cat,run_id,index,tile_id):
    run=cat.runs[run_id]; frame=cat.frame(run_id,index)
    matcher=run['summary'].get('config',{}).get('matcher')
    if matcher not in ('gim_dkm','sift','orb','akaze'):
        raise ValueError('Visual re-match is not implemented for this matcher')
    angle=frame.get('rotation',0)
    query=cat.image(run_id,index,'query',angle=angle); ref=cat.image(run_id,index,'map',tile_id)
    if matcher == 'gim_dkm':
        if not GIM_HOST: raise ValueError('No GIM host configured')
        with WORKER_LOCK:
            # Release the shared GPU after each explicit visual re-run.
            from navlab.lab.remote_gim import RemoteGIMMatcher
            worker=RemoteGIMMatcher(GIM_HOST)
            try:
                out,rp,qp=worker.match_with_points(ref,query)
                if out.execution_error:
                    raise RuntimeError(out.error)
            finally:
                worker.close()
        rsmall=_read_rgb(ref,768)[0]; qsmall=_read_rgb(query,768)[0]
        homography=out.homography
    else:
        from sat2aerial.matching import _resize, _prepare, _detector, _matcher
        cfg=run['summary'].get('config',{})
        size=int(cfg.get('max_dim',1024)); mode=cfg.get('mode','gray')
        rsmall=_resize(ref,size); qsmall=_resize(query,size)
        if run['type']=='ortholoc':
            detector=cv2.SIFT_create(nfeatures=7000)
            kq,dq=detector.detectAndCompute(cv2.cvtColor(qsmall,cv2.COLOR_BGR2GRAY),None)
            kr,dr=detector.detectAndCompute(cv2.cvtColor(rsmall,cv2.COLOR_BGR2GRAY),None)
            knn=cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq,dr,k=2) if dq is not None and dr is not None and len(dr)>=2 else []
            good=[a for pair in knn if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
            rp=np.array([kr[m.trainIdx].pt for m in good],np.float32).reshape(-1,2)
            qp=np.array([kq[m.queryIdx].pt for m in good],np.float32).reshape(-1,2)
        else:
            detector=_detector(matcher)
            kr,dr=detector.detectAndCompute(_prepare(rsmall,mode),None)
            kq,dq=detector.detectAndCompute(_prepare(qsmall,mode),None)
            knn=_matcher(matcher).knnMatch(dr,dq,k=2) if dr is not None and dq is not None and len(dq)>=2 else []
            good=[a for pair in knn if len(pair)==2 for a,b in [pair] if a.distance<float(cfg.get('ratio',.75))*b.distance]
            rp=np.array([kr[m.queryIdx].pt for m in good],np.float32).reshape(-1,2)
            qp=np.array([kq[m.trainIdx].pt for m in good],np.float32).reshape(-1,2)
        homography=None
        if len(rp)>=4:
            cv2.setRNGSeed(int(cfg.get('rng_seed',0)))
            homography,_=cv2.findHomography(rp,qp,cv2.RANSAC,float(cfg.get('ransac_px',5)),maxIters=5000,confidence=.999)
    if run['type']=='ortholoc':
        # Raw, tentative correspondences: PnP's 3D inliers are not stored.
        keep=range(len(rp)); label='این‌ها نقاط پیشنهادی بازاجرا هستند؛ نقاط پذیرفته‌شدهٔ PnP ذخیره نشده‌اند.'
    else:
        if homography is None:
            return {'pairs':[],'rotation':angle,'rerun_inliers':0,'note':'در بازاجرا تبدیل هندسی پیدا نشد.'}
        mapped=cv2.perspectiveTransform(np.float32(rp).reshape(-1,1,2),homography).reshape(-1,2)
        keep=np.flatnonzero(np.linalg.norm(mapped-qp,axis=1)<3)
        label='نقاط برای نمایش دوباره محاسبه شده‌اند؛ تصمیم و خطا مربوط به اجرای ذخیره‌شده‌اند.'
    chosen=[]
    for i in keep:
        if all(np.linalg.norm(qp[i]-qp[j])>90 for j in chosen): chosen.append(i)
        if len(chosen)>=12: break
    pairs=[{'query':[float(qp[i][0]/qsmall.shape[1]),float(qp[i][1]/qsmall.shape[0])],
            'map':[float(rp[i][0]/rsmall.shape[1]),float(rp[i][1]/rsmall.shape[0])]} for i in chosen]
    return {'pairs':pairs,'rotation':angle,'rerun_inliers':len(keep),'note':label}


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        try:
            u=urlsplit(self.path); p=parse_qs(u.query); cat=catalog()
            if u.path=='/api/catalog': return self.send_json(catalog(p.get('refresh',['0'])[0]=='1').list())
            if u.path=='/api/frames':
                run=p['run'][0]; return self.send_json([cat.frame(run,i) for i in range(len(cat.runs[run]['rows']))])
            if u.path=='/api/image':
                run=p['run'][0]; idx=int(p['index'][0]); kind=p['kind'][0]; tile=p.get('tile',[None])[0]; angle=int(p.get('angle',['0'])[0])
                if angle not in (0,90,180,270): raise ValueError('invalid angle')
                img=cat.image(run,idx,kind,tile,angle)
                factor=min(1.0,1400/max(img.shape[:2])); img=cv2.resize(img,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA) if factor<1 else img
                ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,88])
                if not ok: raise ValueError('encoding failed')
                raw=buf.tobytes(); self.send_response(200); self.send_header('Content-Type','image/jpeg'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw); return
            name={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}.get(u.path)
            if name:
                raw=(STATIC/name).read_bytes(); self.send_response(200); self.send_header('Content-Type',mimetypes.guess_type(name)[0]+'; charset=utf-8'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw); return
            self.send_json({'error':'not found'},404)
        except (KeyError,IndexError,ValueError,TypeError,FileNotFoundError) as e:
            self.send_json({'error':str(e)},400)
        except Exception as e:
            self.send_json({'error':f'{type(e).__name__}: {e}'},500)

    def do_POST(self):
        try:
            if urlsplit(self.path).path!='/api/rematch': return self.send_json({'error':'not found'},404)
            origin=self.headers.get('Origin','')
            host=self.headers.get('Host','')
            if origin and origin not in (f'http://{host}',f'https://{host}'): return self.send_json({'error':'origin denied'},403)
            length=int(self.headers.get('Content-Length','0'))
            if length<=0 or length>10000: raise ValueError('bad request length')
            body=json.loads(self.rfile.read(length)); result=rematch(catalog(),body['run'],int(body['index']),body['tile'])
            self.send_json(result)
        except (KeyError,IndexError,ValueError,TypeError,json.JSONDecodeError) as e: self.send_json({'error':str(e)},400)
        except Exception as e: self.send_json({'error':f'{type(e).__name__}: {e}'},500)


def main():
    global GIM_HOST
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--gim-host',default='ava@172.16.20.117')
    args=parser.parse_args(); GIM_HOST=args.gim_host
    count=len(catalog().runs)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Localization review: http://127.0.0.1:{args.port}/  ({count} runs)',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()

if __name__=='__main__': main()
