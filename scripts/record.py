"""Record a trained agent to an MP4 / GIF (fixed replacement for Capture.py).

The original Capture.py imported the removed ``gym.wrappers.monitoring`` API.
This version uses ``render_mode="rgb_array"`` + imageio, which works on modern
Gymnasium and headless machines (e.g. Colab).

Examples
--------
    python scripts/record.py --algo dqn --model model/checkpoint.pth --out video/run.mp4
    python scripts/record.py --algo ppo --model runs/ppo/model.pt --out doc/ppo.gif
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio
import numpy as np

# Allow running as `python scripts/record.py` without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.envs import env_dims, make_env
from rl.train import build_agent
from rl.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn")
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="video/captured.mp4")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    continuous = args.continuous or args.algo in ("ddpg", "td3", "sac")
    env = make_env(args.env, render_mode="rgb_array",
                   continuous=continuous if args.env.startswith("LunarLander") else None,
                   seed=args.seed)
    _, _, discrete = env_dims(env)
    agent = build_agent(args.algo, env, discrete, get_device("cpu"), args.seed)
    agent.load(args.model)

    frames = []
    for _ in range(args.episodes):
        state, _ = env.reset()
        done = False
        while not done:
            frames.append(env.render())
            if args.algo in ("ddpg", "td3"):
                action = agent.act(state, noise=False)
            elif args.algo == "sac":
                action = agent.act(state, deterministic=True)
            elif args.algo == "dqn":
                action = agent.act(state, eps=0.0)
            else:
                action = agent.act(state, greedy=True)
                if isinstance(action, tuple):
                    action = action[0]
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, frames, fps=args.fps)
    else:
        imageio.mimsave(args.out, frames, fps=args.fps, codec="libx264")
    print(f"Saved {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
