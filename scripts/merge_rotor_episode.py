"""Merge Gazebo camera frames + MAVSDK action log into a FLUX 3 Action 'rotor' episode.

Aligns each action (by its absolute capture time) with the nearest camera frame,
keeping a strictly increasing frame index, then writes the rotor-format NPZ:
frames (T,256,256,3) uint8 + action (T,4) float32 in [-1,1].

Usage:
    python merge_rotor_episode.py --frames-dir run1 --actions run1/actions.csv \
        --steps 300 --task "take off, fly forward, and hover" --out ep_0000.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--actions", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    frames = np.load(Path(args.frames_dir) / "frames.npy").astype(np.uint8)
    frame_ts = pd.read_csv(Path(args.frames_dir) / "frame_ts.csv")["t"].to_numpy()
    actions = pd.read_csv(args.actions)

    n = min(args.steps, len(actions))
    act = actions[["fwd", "lat", "up", "yaw"]].to_numpy(dtype=np.float32)[:n]
    t_abs = actions["t_abs"].to_numpy()[:n]

    idx = np.searchsorted(frame_ts, t_abs)
    idx = np.clip(idx, 0, len(frames) - 1)
    # ensure strictly increasing frame picks
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = min(idx[i - 1] + 1, len(frames) - 1)
    sel = frames[idx]

    assert sel.shape[1:] == (256, 256, 3), sel.shape
    assert act.shape[1] == 4
    assert np.isfinite(act).all() and np.abs(act).max() <= 1.0

    np.savez(args.out, frames=sel, action=act)
    meta = {"length": n, "action_dim": 4, "task": args.task, "file": Path(args.out).name}
    print(json.dumps(meta), flush=True)
    print(f"saved {n} steps (frames {idx[0]}..{idx[-1]}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
