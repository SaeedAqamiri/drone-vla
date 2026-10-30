"""Write the Gazebo model ground-truth pose to truth.json for NavLab."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from gz.msgs10 import pose_v_pb2
import gz.transport13 as gztransport

stop = False
last_write = 0.0


def _handler(*_args) -> None:
    global stop
    stop = True


def discover_topic() -> str | None:
    try:
        output = subprocess.check_output(["gz", "topic", "-l"], text=True, timeout=8)
    except Exception:
        return None
    for line in output.splitlines():
        if line.strip().endswith("/pose/info"):
            return line.strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="auto")
    parser.add_argument("--model", default="x500_mono_cam_down_0")
    parser.add_argument("--out", default="outputs/navlab/truth/truth.json")
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    topic = args.topic
    if topic == "auto":
        topic = discover_topic()
        if not topic:
            print("no pose topic found", flush=True)
            return 1
    print(f"truth topic {topic}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / max(args.hz, 1.0)

    def callback(raw_msg, _info) -> None:
        global last_write
        now = time.time()
        if now - last_write < period:
            return
        message = pose_v_pb2.Pose_V()
        message.ParseFromString(raw_msg)
        for pose in message.pose:
            if args.model not in pose.name:
                continue
            q = pose.orientation
            yaw_enu = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            yaw_ned = math.pi / 2.0 - yaw_enu
            while yaw_ned > math.pi:
                yaw_ned -= 2.0 * math.pi
            while yaw_ned < -math.pi:
                yaw_ned += 2.0 * math.pi
            payload = {
                "timestamp": now,
                "model": pose.name,
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
                "yaw": yaw_ned,
            }
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, out)
            last_write = now
            return

    node = gztransport.Node()
    node.subscribe_raw(topic, callback, "gz.msgs.Pose_V", gztransport.SubscribeOptions())
    while not stop:
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
