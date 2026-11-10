#!/usr/bin/env python3
"""Summarize a strict frame-level retrospective proxy, NOT flight success.

Each recorded segment counts as passing only if every selected frame yielded a
fix within the evaluation threshold. Missing frames fail this proxy. Results
are not independent flights and cannot be converted into a mission probability.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def summarize(path: Path, threshold: float, expected_frames: int):
    with path.open(newline='') as file:
        rows=list(csv.DictReader(file))
    if len(rows)!=expected_frames:
        raise ValueError(f'{path}: expected {expected_frames} frames, got {len(rows)}')
    if any(not row.get('timestamp') for row in rows):
        raise ValueError(f'{path}: missing frame timestamp')
    timestamps=[float(r['timestamp']) for r in rows]
    if any(not math.isfinite(x) for x in timestamps) or any(a>=b for a,b in zip(timestamps,timestamps[1:])):
        raise ValueError(f'{path}: invalid timestamp sequence')
    missing=0; over=0; within=0
    for r in rows:
        if r.get('accepted','').lower()!='true' or not r.get('error_m'):
            missing+=1
            continue
        error=float(r['error_m'])
        if not math.isfinite(error) or error<0:
            raise ValueError(f'{path}: invalid error')
        if error<=threshold: within+=1
        else: over+=1
    return {'source':str(path),'frames':len(rows),'duration_s':timestamps[-1]-timestamps[0],
            'within_threshold':within,'accepted_over_threshold':over,'no_accepted_error':missing,
            'all_frames_within_threshold':within==len(rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames',type=Path,nargs='+')
    p.add_argument('--threshold-m',type=float,default=30)
    p.add_argument('--expected-frames',type=int,default=10)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not math.isfinite(args.threshold_m) or args.threshold_m<=0 or args.expected_frames<2:
        p.error('threshold must be positive and expected-frames at least two')
    segments=[summarize(x,args.threshold_m,args.expected_frames) for x in args.frames]
    result={'note':'Strict retrospective image-frame proxy; not simulated or measured destination-arrival probability. Short correlated, selected trajectories; not independent missions.',
            'threshold_m':args.threshold_m,'segments':segments,
            'all_frames_passed_segments':sum(x['all_frames_within_threshold'] for x in segments),
            'segment_count':len(segments)}
    text=json.dumps(result,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text+'\n')
    print(text)

if __name__=='__main__': main()
