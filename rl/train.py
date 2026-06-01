"""Unified training entry point.

Different algorithm families have genuinely different interaction loops, and
pretending otherwise hides what makes them tick. So instead of one mega-loop we
expose three honest ones and a CLI that dispatches to the right one:

* ``train_offpolicy`` - DQN, DDPG, TD3, SAC: act, store, learn every step.
* ``train_onpolicy``  - A2C, PPO: collect a fixed-length rollout, then update.
* ``train_episodic``  - REINFORCE: collect a whole episode, then update.

Run from the command line, e.g.::

    python -m rl.train --algo ppo  --env LunarLander-v3
    python -m rl.train --algo dqn  --env LunarLander-v3
    python -m rl.train --algo sac  --env LunarLander-v3 --continuous
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from rl.envs import env_dims, make_env
from rl.utils import ExponentialSchedule, Logger, get_device, plot_scores, set_seed


# ----------------------------------------------------------------------------
# Off-policy: DQN / DDPG / TD3 / SAC
# ----------------------------------------------------------------------------
def train_offpolicy(agent, env, n_steps: int = 200_000, eps_schedule=None,
                    logger: Optional[Logger] = None, solved_at: Optional[float] = None,
                    eval_every: int = 10_000, discrete: bool = True):
    """Step-based loop for replay-buffer agents. Returns episode scores."""
    scores, recent = [], []
    state, _ = env.reset()
    score, ep = 0.0, 0
    for step in range(1, n_steps + 1):
        if discrete:
            eps = eps_schedule.step() if eps_schedule else 0.0
            action = agent.act(state, eps)
        else:
            action = agent.act(state)  # continuous agents add their own noise
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.step(state, action, reward, next_state, terminated)
        state = next_state
        score += reward

        if terminated or truncated:
            scores.append(score)
            recent.append(score)
            recent = recent[-100:]
            ep += 1
            state, _ = env.reset()
            score = 0.0
            if logger and ep % 10 == 0:
                logger.log(step, {"episode": ep, "score": scores[-1],
                                  "avg100": float(np.mean(recent))})
            if solved_at and len(recent) >= 100 and np.mean(recent) >= solved_at:
                print(f"\nSolved in {ep} episodes  (avg100={np.mean(recent):.1f})")
                break
    return scores


# ----------------------------------------------------------------------------
# On-policy: A2C / PPO
# ----------------------------------------------------------------------------
def train_onpolicy(agent, env, total_steps: int = 1_000_000,
                   logger: Optional[Logger] = None, solved_at: Optional[float] = None):
    """Rollout-based loop. Collects ``agent.buffer.size`` steps, then updates."""
    rollout = agent.buffer.size
    scores, recent = [], []
    state, _ = env.reset()
    score, ep = 0.0, 0
    step = 0
    while step < total_steps:
        for _ in range(rollout):
            action, log_prob, value = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.buffer.add(state, action, reward, value, log_prob, terminated)
            state = next_state
            score += reward
            step += 1
            if terminated or truncated:
                scores.append(score)
                recent.append(score)
                recent = recent[-100:]
                ep += 1
                state, _ = env.reset()
                score = 0.0
        # Bootstrap value of the state we stopped on, then learn.
        _, _, last_value = agent.act(state)
        stats = agent.update(last_value, last_done=False)
        if logger:
            logger.log(step, {"episodes": ep,
                              "avg100": float(np.mean(recent)) if recent else 0.0,
                              **stats})
        if solved_at and len(recent) >= 100 and np.mean(recent) >= solved_at:
            print(f"\nSolved at step {step}  (avg100={np.mean(recent):.1f})")
            break
    return scores


# ----------------------------------------------------------------------------
# Episodic: REINFORCE
# ----------------------------------------------------------------------------
def train_episodic(agent, env, n_episodes: int = 2000,
                   logger: Optional[Logger] = None, solved_at: Optional[float] = None):
    scores, recent = [], []
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        score, done = 0.0, False
        while not done:
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            agent.record_reward(reward)
            score += reward
            done = terminated or truncated
        agent.finish_episode()
        scores.append(score)
        recent.append(score)
        recent = recent[-100:]
        if logger and ep % 20 == 0:
            logger.log(ep, {"episode": ep, "avg100": float(np.mean(recent))})
        if solved_at and len(recent) >= 100 and np.mean(recent) >= solved_at:
            print(f"\nSolved in {ep} episodes  (avg100={np.mean(recent):.1f})")
            break
    return scores


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_agent(algo: str, env, discrete: bool, device, seed: int):
    """Construct the requested agent with curriculum-tuned defaults."""
    state_dim, action_dim, _ = env_dims(env)
    algo = algo.lower()
    if algo == "dqn":
        from rl.agents.dqn import DQN, DQNConfig
        return DQN(state_dim, action_dim, DQNConfig(), device, seed)
    if algo == "reinforce":
        from rl.agents.reinforce import REINFORCE, ReinforceConfig
        return REINFORCE(state_dim, action_dim, discrete, ReinforceConfig(), device, seed)
    if algo == "a2c":
        from rl.agents.a2c import A2C, A2CConfig
        return A2C(state_dim, action_dim, discrete, A2CConfig(), device, seed)
    if algo == "ppo":
        from rl.agents.ppo import PPO, PPOConfig
        return PPO(state_dim, action_dim, discrete, PPOConfig(), device, seed)
    # continuous-only agents
    action_scale = float(env.action_space.high[0]) if not discrete else 1.0
    if algo == "ddpg":
        from rl.agents.ddpg import DDPG, DDPGConfig
        return DDPG(state_dim, action_dim, action_scale, DDPGConfig(), device, seed)
    if algo == "td3":
        from rl.agents.td3 import TD3, TD3Config
        return TD3(state_dim, action_dim, action_scale, TD3Config(), device, seed)
    if algo == "sac":
        from rl.agents.sac import SAC, SACConfig
        return SAC(state_dim, action_dim, action_scale, SACConfig(), device, seed)
    raise SystemExit(f"Unknown algo '{algo}'")


def main():
    p = argparse.ArgumentParser(description="Train an RL agent from the rl.* library.")
    p.add_argument("--algo", required=True,
                   choices=["dqn", "reinforce", "a2c", "ppo", "ddpg", "td3", "sac"])
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--continuous", action="store_true",
                   help="Use the continuous action variant (required for ddpg/td3/sac).")
    p.add_argument("--steps", type=int, default=300_000, help="env steps (off/on-policy)")
    p.add_argument("--episodes", type=int, default=2000, help="episodes (reinforce)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--solved-at", type=float, default=200.0)
    p.add_argument("--save", default=None, help="path to save the trained model")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    continuous = args.continuous or args.algo in ("ddpg", "td3", "sac")
    env = make_env(args.env, continuous=continuous if args.env.startswith("LunarLander") else None,
                   seed=args.seed)
    _, _, is_discrete = env_dims(env)
    discrete = is_discrete

    agent = build_agent(args.algo, env, discrete, device, args.seed)
    run_dir = args.run_dir or os.path.join("runs", f"{args.algo}_{args.env}_{args.seed}")
    logger = Logger(run_dir)
    print(f"Training {args.algo.upper()} on {args.env} "
          f"({'discrete' if discrete else 'continuous'}) -> {run_dir}")

    if args.algo == "reinforce":
        scores = train_episodic(agent, env, args.episodes, logger, args.solved_at)
    elif args.algo in ("a2c", "ppo"):
        scores = train_onpolicy(agent, env, args.steps, logger, args.solved_at)
    else:
        eps = ExponentialSchedule(1.0, 0.01, 0.9995) if discrete else None
        scores = train_offpolicy(agent, env, args.steps, eps, logger,
                                 args.solved_at, discrete=discrete)

    if args.save:
        agent.save(args.save)
        print(f"Saved model -> {args.save}")
    plot_scores(scores, title=f"{args.algo.upper()} on {args.env}",
                save_path=os.path.join(run_dir, "scores.png"))
    env.close()


if __name__ == "__main__":
    main()
