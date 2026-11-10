#!/usr/bin/env python3
"""Save the first message on a Gazebo image topic to a ``.npy`` file.

Runs under the **system** python3 (it needs ``gz.transport13`` bindings, which
the project venvs do not have). ``scripts/check_geo_world.py`` shells out to it.

Usage:
    /usr/bin/python3 scripts/gz_capture_once.py --topic /probe/image \\
        --out /tmp/capture.npy --timeout 30
"""

import argparse
import time
from pathlib import Path

import numpy as np

from gz.msgs10 import image_pb2
import gz.transport13 as gztransport


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--size", type=int, default=0, help="resize to NxN (0 keeps native)")
    args = parser.parse_args()

    node = gztransport.Node()
    state: dict = {"frame": None}

    def callback(raw_msg, _info):
        if state["frame"] is not None:
            return
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

    node.subscribe_raw(args.topic, callback, "gz.msgs.Image", gztransport.SubscribeOptions())

    started = time.time()
    while state["frame"] is None and time.time() - started < args.timeout:
        time.sleep(0.05)

    if state["frame"] is None:
        print(f"TIMEOUT waiting for {args.topic}", flush=True)
        return 2

    frame = state["frame"]
    if args.size:
        import cv2

        frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, frame)
    print(f"saved {out} {frame.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
