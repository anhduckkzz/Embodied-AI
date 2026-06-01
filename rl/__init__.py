"""rl: a from-scratch, educational Reinforcement Learning library.

This package accompanies the "RL: Zero to Master" curriculum. Every algorithm
is implemented to be *read*, not just run: the code mirrors the maths in the
lessons under ``curriculum/`` and favours clarity over micro-optimisation.

Layout
------
- ``rl.networks``  : reusable neural network building blocks (MLP, Q-net, etc.)
- ``rl.buffers``   : experience storage (replay buffer, rollout buffer)
- ``rl.utils``     : seeding, logging, schedules, plotting helpers
- ``rl.envs``      : environment creation / wrappers
- ``rl.agents``    : the algorithms (DQN, REINFORCE, PPO, SAC, ...)
- ``rl.train``     : a small command-line trainer tying everything together

The public surface is intentionally tiny; import what you need explicitly,
e.g. ``from rl.agents.ppo import PPO``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
