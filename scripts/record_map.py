"""Record a visual keyframe map from NavLab's live camera + Gazebo truth."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

CAMERA = Path("outputs/navlab/camera/latest.jpg")
TRUTH = Path("outputs/navlab/truth/truth.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--stop-file", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"z0": None, "frames": []}
    last_mtime = 0.0
    started = time.time()
    period = 1.0 / max(args.hz, 1.0)
    stop_file = Path(args.stop_file) if args.stop_file else None

    while time.time() - started < args.duration:
        if stop_file and stop_file.exists():
            break
        try:
            if CAMERA.exists() and TRUTH.exists():
                mtime = CAMERA.stat().st_mtime
                if mtime != last_mtime:
                    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
                    index = len(manifest["frames"]) + 1
                    dest = frames_dir / f"{index:06d}.jpg"
                    shutil.copyfile(CAMERA, dest)
                    if manifest["z0"] is None:
                        manifest["z0"] = truth["z"]
                    manifest["frames"].append(
                        {
                            "frame": str(dest),
                            "t": truth.get("timestamp", time.time()),
                            "x": truth["x"],
                            "y": truth["y"],
                            "z": truth["z"],
                        }
                    )
                    tmp = out / "manifest.tmp"
                    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    (out / "manifest.json").write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )
                    last_mtime = mtime
        except Exception as error:
            print(f"record error: {error}", flush=True)
        time.sleep(period)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"recorded {len(manifest['frames'])} frames -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
