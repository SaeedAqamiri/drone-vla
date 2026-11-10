#!/usr/bin/env python3
"""Offline, per-frame localization scores grouped by recorded aircraft attitude.

Reported pitch/roll are metadata, not a calibrated camera viewing angle.
Attitude must be present in the original CSV; an imputed zero is not a reading.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np


def available_attitudes(root: Path, region: str) -> dict[str, bool]:
    path=root/region/f'{region}.csv'
    with path.open(newline='',encoding='utf-8-sig') as f:
        return {str(row.get('filename','')).strip():all(str(row.get(k) or '').strip()
                for k in ('Omega','Kappa')) for row in csv.DictReader(f)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',nargs=2,action='append',metavar=('STUDY','RESULT'),required=True)
    p.add_argument('--root',type=Path,default=Path('data/raw/uav_visloc/UAV_VisLoc_dataset'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();buckets={};cache={}
    for study_s,result_s in args.run:
        study,result=Path(study_s),Path(result_s)
        queries={q['id']:q for q in json.loads((study/'queries.json').read_text())['queries']}
        with (result/'frames.csv').open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                q=queries[row['query_id']]; region=q['id'].split('_')[0]
                if region not in cache:cache[region]=available_attitudes(args.root,region)
                if not cache[region].get(Path(q['path']).name,False): label='missing'
                else:
                    meta=q['meta'];angle=max(abs(float(meta['pitch_deg'])),abs(float(meta['roll_deg'])))
                    label='<3' if angle<3 else '3-<6' if angle<6 else '6-<10' if angle<10 else '>=10'
                bucket=buckets.setdefault(label,{'frames':0,'accepted':0,'within_30m':0,'errors_m':[],'regions':{}})
                bucket['frames']+=1;bucket['regions'][region]=bucket['regions'].get(region,0)+1
                if row['accepted']=='True' and row['error_m']:
                    e=float(row['error_m'])
                    if math.isfinite(e):
                        bucket['accepted']+=1;bucket['errors_m'].append(e);bucket['within_30m']+=e<=30
    for b in buckets.values():
        e=b.pop('errors_m'); b['median_accepted_m']=float(np.median(e)) if e else None
        b['p95_accepted_m']=float(np.percentile(e,95)) if e else None
    result={'definition':'max(abs(recorded Omega),abs(recorded Kappa)); camera attitude not established',
            'limitation':'Offline descriptive grouped frames, non-independent and selected runs; not a flight certification.',
            'runs':[{'study':s,'result':r} for s,r in args.run], 'buckets':buckets}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
