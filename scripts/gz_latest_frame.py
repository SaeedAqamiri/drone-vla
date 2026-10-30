"""Publish the latest Gazebo camera frame as latest.jpg for the NavLab UI.

Run with the system Python that has the gz bindings:

    /usr/bin/python3 scripts/gz_latest_frame.py --topic auto \
        --out outputs/navlab/camera --fps 5
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from PIL import Image

from gz.msgs10 import image_pb2
import gz.transport13 as gztransport

stop = False
lock = threading.Lock()
last_write = 0.0
frame_count = 0


def _handler(*_args) -> None:
    global stop
    stop = True


def discover_topic() -> str | None:
    try:
        output = subprocess.check_output(["gz", "topic", "-l"], text=True, timeout=8)
    except Exception:
        return None
    candidates = [
        line.strip()
        for line in output.splitlines()
        if line.strip().endswith("/image") and "camera" in line.lower()
    ]
    return candidates[0] if candidates else None


def write_frame(rgb: np.ndarray, out: Path) -> None:
    global last_write, frame_count
    now = time.time()
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / "latest.jpg"
    meta_path = out / "latest.json"
    tmp = out / "latest.jpg.tmp"
    Image.fromarray(rgb).save(tmp, format="JPEG", quality=85)
    os.replace(tmp, image_path)
    frame_count += 1
    last_write = now
    payload = {
        "timestamp": now,
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "count": frame_count,
    }
    meta_tmp = out / "latest.json.tmp"
    meta_tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(meta_tmp, meta_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="auto")
    parser.add_argument("--out", default="outputs/navlab/camera")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    topic = args.topic
    if topic == "auto":
        topic = discover_topic()
        if not topic:
            print("no camera topic found; is Gazebo running?", flush=True)
            return 1
    print(f"subscribing to {topic}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    period = 1.0 / max(args.fps, 1.0)

    def callback(raw_msg, _info) -> None:
        global last_write
        if stop:
            return
        now = time.time()
        if now - last_write < period:
            return
        message = image_pb2.Image()
        message.ParseFromString(raw_msg)
        width, height = int(message.width), int(message.height)
        if width <= 0 or height <= 0:
            return
        data = np.frombuffer(message.data, dtype=np.uint8)
        channels = len(data) // (width * height)
        if channels <= 0:
            return
        frame = data.reshape(height, width, channels)
        if channels == 1:
            frame = np.repeat(frame, 3, axis=2)
        frame = frame[:, :, :3]
        with lock:
            write_frame(frame, out)

    node = gztransport.Node()
    node.subscribe_raw(topic, callback, "gz.msgs.Image", gztransport.SubscribeOptions())

    while not stop:
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
