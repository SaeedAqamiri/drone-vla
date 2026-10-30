"""Capture camera frames from a Gazebo camera topic (system python3 + gz bindings).

Writes frames to <outdir>/frames.npy (T,256,256,3 uint8) and timestamps to
<outdir>/frame_ts.csv while running; stops after T frames or SIGINT.

Usage:
    python3 gz_frame_capture.py --topic /world/default/.../image --frames 300 --outdir ...
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import numpy as np

from gz.msgs10 import image_pb2

import gz.transport13 as gztransport

done = False


def main():
    global done
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--outdir", required=True)
    p.add_argument("--trigger", type=str, default=None, help="wait until this file exists, then capture")
    p.add_argument("--done", type=str, default=None, help="keep capturing until this file exists")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, lambda *_: setattr(sys.modules[__name__], "done", True))

    node = gztransport.Node()
    frames = []
    ts = []
    state = {"fmt": None}

    def cb(raw_msg, _msg_info):
        msg = image_pb2.Image()
        msg.ParseFromString(raw_msg)
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        channels = max(1, len(arr) // (msg.width * msg.height))
        arr = arr.reshape(msg.height, msg.width, channels)
        if channels >= 3:
            arr = arr[:, :, :3]
        else:
            arr = np.repeat(arr, 3, axis=2)
        frames.append(arr)
        ts.append(time.time())
        if len(frames) >= args.frames:
            state["done"] = True

    node.subscribe_raw(args.topic, cb, "gz.msgs.Image", gztransport.SubscribeOptions())
    print(f"subscribed to {args.topic}, waiting for frames ...", flush=True)

    own_start = time.time()
    if args.trigger:
        print(f"waiting for trigger file {args.trigger} ...", flush=True)
        collect = False
    else:
        collect = True
    t0 = time.time()
    while time.time() - t0 < 600 and not done:
        if args.done and Path(args.done).exists():
            break
        if len(frames) >= args.frames:
            time.sleep(0.05)
            continue
        if not collect:
            if Path(args.trigger).exists() and Path(args.trigger).stat().st_mtime > own_start:
                collect = True
                frames.clear()
                ts.clear()
                print("trigger received, capturing ...", flush=True)
            time.sleep(0.05)
            continue
        time.sleep(0.02)

    if not frames:
        print("no frames received", flush=True)
        sys.exit(1)

    arr = np.stack(frames).astype(np.uint8)
    np.save(outdir / "frames.npy", arr)
    with open(outdir / "frame_ts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"])
        w.writerows([[t] for t in ts])
    print(f"captured {len(arr)} frames {arr.shape} -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
