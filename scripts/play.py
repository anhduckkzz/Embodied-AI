"""Watch a trained agent play (the modern, fixed replacement for Runner.py).

Examples
--------
    python scripts/play.py --algo dqn --model model/checkpoint.pth
    python scripts/play.py --algo ppo --model runs/ppo/model.pt --episodes 5
"""
from __future__ import annotations

import argparse

import numpy as np

from rl.envs import env_dims, make_env
from rl.train import build_agent
from rl.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="dqn")
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-render", action="store_true", help="run headless (no window)")
    args = p.parse_args()

    set_seed(args.seed)
    continuous = args.continuous or args.algo in ("ddpg", "td3", "sac")
    render_mode = None if args.no_render else "human"
    env = make_env(args.env, render_mode=render_mode,
                   continuous=continuous if args.env.startswith("LunarLander") else None,
                   seed=args.seed)
    _, _, discrete = env_dims(env)

    agent = build_agent(args.algo, env, discrete, get_device("cpu"), args.seed)
    agent.load(args.model)
    print(f"Loaded {args.algo} from {args.model}")

    returns = []
    for ep in range(args.episodes):
        state, _ = env.reset()
        done, score = False, 0.0
        while not done:
            # Use greedy / no-noise actions for evaluation.
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
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        returns.append(score)
        print(f"Episode {ep + 1}: return = {score:.1f}")
    print(f"\nMean over {args.episodes} episodes: {np.mean(returns):.1f}")
    env.close()


if __name__ == "__main__":
    main()
