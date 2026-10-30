"""Log quadcopter episodes from Isaac Lab into NPZ files for VLA training.

Runs the registered `Isaac-Quadcopter-Direct-v0` environment with a random or
hover-throttle policy and records (obs, action, reward, termination) per step.

Usage:
    python isaac_quad_logger.py --headless --num_envs 4 --steps 2000 --outdir ~/drone-vla/logs/isaac_quad
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quadcopter episode logger")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--outdir", type=str, default=os.path.expanduser("~/drone-vla/logs/isaac_quad"))
parser.add_argument("--policy", type=str, default="random", choices=["random", "zero"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Isaac-Quadcopter-Direct-v0", device=args.device, num_envs=args.num_envs)
    env = gym.make("Isaac-Quadcopter-Direct-v0", cfg=env_cfg)

    os.makedirs(args.outdir, exist_ok=True)

    obs, _ = env.reset()
    obs_buf, act_buf, rew_buf, term_buf = [], [], [], []

    t0 = time.time()
    for step in range(args.steps):
        if args.policy == "random":
            action = env.action_space.sample()
            action = torch.as_tensor(action, device=env.unwrapped.device)
        else:
            action = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.unwrapped.device)

        obs, reward, terminated, truncated, extra = env.step(action)

        obs_buf.append(obs["policy"].cpu().numpy() if isinstance(obs, dict) else obs.cpu().numpy())
        act_buf.append(action.cpu().numpy())
        rew_buf.append(reward.cpu().numpy())
        term_buf.append((terminated | truncated).cpu().numpy())

        if step % 200 == 0:
            fps = (step + 1) / (time.time() - t0)
            print(f"[step {step}/{args.steps}] fps={fps:.1f}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.outdir, f"quad_{stamp}.npz")
    np.savez_compressed(
        out_path,
        obs=np.concatenate(obs_buf, axis=0),
        actions=np.concatenate(act_buf, axis=0),
        rewards=np.concatenate(rew_buf, axis=0),
        terminations=np.concatenate(term_buf, axis=0),
        dt=env.unwrapped.step_dt,
        meta=np.array([f"policy={args.policy}; num_envs={args.num_envs}; steps={args.steps}"]),
    )
    print(f"saved {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
