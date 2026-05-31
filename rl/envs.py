"""Environment creation helpers.

A thin wrapper over Gymnasium so every script makes environments the same way
and so we can attach common wrappers (episode statistics, observation/reward
normalisation) in one place. The curriculum standardises on **Gymnasium**
(the maintained successor to OpenAI Gym).
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym


# The classic LunarLander config used throughout the curriculum. Wind + gravity
# make it a genuinely interesting control problem rather than a toy.
LUNAR_KWARGS = dict(
    continuous=False,
    gravity=-10.0,
    enable_wind=True,
    wind_power=15.0,
    turbulence_power=1.5,
)


def make_env(env_id: str = "LunarLander-v3", render_mode: Optional[str] = None,
             continuous: Optional[bool] = None, seed: Optional[int] = None,
             record_stats: bool = True, **kwargs) -> gym.Env:
    """Create a single Gymnasium environment with sensible defaults.

    Parameters
    ----------
    env_id : str
        e.g. "LunarLander-v3", "CartPole-v1", "Pendulum-v1", "FrozenLake-v1".
    continuous : Optional[bool]
        For LunarLander, switches between the discrete and continuous variants
        (the continuous one is the bridge toward robotics control).
    record_stats : bool
        Wrap with ``RecordEpisodeStatistics`` so episode return/length are
        reported in ``info["episode"]`` on termination.
    """
    make_kwargs = dict(kwargs)
    if env_id.startswith("LunarLander"):
        merged = dict(LUNAR_KWARGS)
        merged.update(make_kwargs)
        if continuous is not None:
            merged["continuous"] = continuous
        make_kwargs = merged

    env = gym.make(env_id, render_mode=render_mode, **make_kwargs)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def env_dims(env: gym.Env):
    """Return ``(state_dim, action_dim, is_discrete)`` for an environment.

    For discrete spaces ``action_dim`` is the number of actions; for continuous
    spaces it is the action vector length.
    """
    state_dim = int(gym.spaces.utils.flatdim(env.observation_space))
    if isinstance(env.action_space, gym.spaces.Discrete):
        return state_dim, int(env.action_space.n), True
    return state_dim, int(env.action_space.shape[0]), False
