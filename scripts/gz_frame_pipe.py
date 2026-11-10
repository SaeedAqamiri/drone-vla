#!/usr/bin/env python3
"""Publish the newest Gazebo camera frame to a file the vision loop can read.

Runs under the **system** python3 (gz bindings). The vision loop
(`scripts/geo_vision_loop.py`, project venv) reads the file and treats a changed
mtime as a new frame. Files instead of IPC because the two need different
interpreters (gz bindings vs rasterio/pyproj).

    /usr/bin/python3 scripts/gz_frame_pipe.py --topic /probe/image \\
        --out /dev/shm/geo_frame.jpg --max-hz 10

The write is atomic (temp file + rename), so the reader never sees a partial
JPEG.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from gz.msgs10 import image_pb2
import gz.transport13 as gztransport

STOP = {"value": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-hz", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    state = {"frame": None, "written": 0}

    def callback(raw_msg, _info):
        message = image_pb2.Image()
        message.ParseFromString(raw_msg)
        array = np.frombuffer(message.data, dtype=np.uint8)
        channels = max(1, len(array) // (message.width * message.height))
        array = array.reshape(message.height, message.width, channels)
        if channels >= 3:
            array = array[:, :, :3]
        else:
            array = np.repeat(array, 3, axis=2)
        state["frame"] = array.copy()

    node = gztransport.Node()
    node.subscribe_raw(args.topic, callback, "gz.msgs.Image", gztransport.SubscribeOptions())

    signal.signal(signal.SIGINT, lambda *_: STOP.update(value=True))
    period = 1.0 / max(args.max_hz, 0.2)
    last = 0.0
    print(f"piping {args.topic} -> {out}", flush=True)

    while not STOP["value"]:
        now = time.time()
        if state["frame"] is not None and now - last >= period:
            ok, buffer = cv2.imencode(".jpg", state["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            if ok:
                tmp.write_bytes(buffer.tobytes())
                os.replace(tmp, out)
                state["written"] += 1
                last = now
        time.sleep(0.005)

    print(f"wrote {state['written']} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
