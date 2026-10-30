"""Record FLUX 3 Action 'rotor'-format drone episodes from Isaac Lab.

Spawns the Crazyflie quadcopter with an onboard 256x256 RGB (tiled) camera and
records scripted-pilot episodes in the episode format consumed by the
flux-action games/drone dataset adapter (docs.bfl.ai/flux_3/flux3_action_drone):

    outputs/data/rotor/
        index.json                    {"length": T, "action_dim": 4, "episodes": [{file, task}, ...]}
        ep_0000.npz                   frames (T,256,256,3) uint8 | action (T,4) float32 in [-1,1]

Action contract of THIS dataset (the adapter is agnostic, so document it):
    [pitch_cmd, roll_cmd, thrust_cmd, yaw_cmd] in [-1, 1] at 15 Hz
    0.526 thrust ~= hover for the Crazyflie (thrust_to_weight 1.9). Instructions are
    saved UNPREFIXED — the adapter prepends "fly the drone: " itself.

Usage:
    python isaac_rotor_logger.py --headless --episodes 2 --steps-per-episode 300 \
        --outdir ~/drone-vla/outputs/data/rotor \
        --task "take off, fly forward, and hover"
"""

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="rotor-format drone episode logger")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--steps-per-episode", type=int, default=300)
parser.add_argument("--outdir", type=str, default=os.path.expanduser("~/drone-vla/outputs/data/rotor"))
parser.add_argument("--task", type=str, default="take off, fly forward, and hover")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import TiledCamera, TiledCameraCfg

from isaaclab_tasks.direct.quadcopter.quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg

HOVER = 0.526  # 1 / thrust_to_weight


class RotorCameraEnvCfg(QuadcopterEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 15 Hz control: sim dt 1/90 x decimation 6 = 1/15
        self.decimation = 6
        self.sim.dt = 1 / 90
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0  # 300 steps @ 15 Hz
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.debug_vis = False
        self.ui_window_class_type = None
        # onboard camera following the drone, looking forward
        self.tiled_camera = TiledCameraCfg(
            prim_path="/World/envs/env_.*/Robot/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.02),
                rot=(0.5, 0.5, -0.5, 0.5),
                convention="ros",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=100.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0)
            ),
            width=256,
            height=256,
        )


class RotorCameraEnv(QuadcopterEnv):
    cfg: RotorCameraEnvCfg

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.sensors["camera"] = self._camera
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


def scripted_pilot(t: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """Varied-but-safe flight segments keeping altitude in [0.35, 1.6] (no early termination)."""
    phase_len = T // 5
    phase = min(t // phase_len, 4)
    sgn = (rng.integers(0, 2, size=4) * 2 - 1).astype(np.float32)
    if phase == 0:  # takeoff
        return np.array([0.0, 0.0, HOVER + 0.06, 0.0], dtype=np.float32)
    if phase == 1:  # tilt forward/back + slight yaw
        return np.array([0.35 * sgn[0], 0.0, HOVER + 0.01, 0.3 * sgn[3]], dtype=np.float32)
    if phase == 2:  # strafe
        return np.array([0.0, 0.30 * sgn[1], HOVER, 0.0], dtype=np.float32)
    if phase == 3:  # combined drift + yaw
        return np.array([0.22 * sgn[0], 0.22 * sgn[1], HOVER, 0.35 * sgn[3]], dtype=np.float32)
    return np.array([0.0, 0.0, HOVER - 0.005, 0.0], dtype=np.float32)  # settle to hover


def main():
    cfg = RotorCameraEnvCfg()
    env = RotorCameraEnv(cfg, render_mode=None)
    camera = env.scene.sensors["camera"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    entries = []
    T = args.steps_per_episode

    for ep in range(args.episodes):
        obs, _ = env.reset()
        env.sim.render()  # warm up the render pipeline so frame 0 shows the scene
        frames, actions = [], []
        for t in range(T):
            a = scripted_pilot(t, T, rng)
            frame = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            frames.append(frame)
            actions.append(a)
            env.step(torch.from_numpy(a)[None].to(env.device))
        ep_path = outdir / f"ep_{ep:04d}.npz"
        np.savez(
            ep_path,
            frames=np.stack(frames).astype(np.uint8),
            action=np.stack(actions).astype(np.float32),
        )
        entries.append({"file": ep_path.name, "task": args.task})
        alt = env._robot.data.root_pos_w[0, 2].item()
        print(f"[ep {ep}] saved {len(frames)} frames | final alt {alt:.2f} m", flush=True)

    index = {"length": T, "action_dim": 4, "episodes": entries}
    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"index.json written: {len(entries)} episodes x {T} steps -> {outdir}", flush=True)

    # format self-check mirroring the flux-action recipe's data contract
    sample = np.load(outdir / entries[0]["file"])
    assert sample["frames"].shape == (T, 256, 256, 3) and sample["frames"].dtype == np.uint8
    assert sample["action"].shape == (T, 4) and sample["action"].dtype == np.float32
    assert np.isfinite(sample["action"]).all() and np.abs(sample["action"]).max() <= 1.0
    print("format check OK: frames (T,256,256,3) uint8 | action (T,4) float32 in [-1,1]", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
