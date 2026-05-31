"""Algorithm implementations.

Each module is self-contained and importable, e.g.::

    from rl.agents.ppo import PPO
    from rl.agents.sac import SAC

The ``REGISTRY`` maps short names (used by ``rl.train`` and the CLI) to agent
classes so you can launch any algorithm by string.
"""

# Lazy registry: import inside the function so that, e.g., the pure-numpy
# tabular lessons don't force a torch import.

def get_agent(name: str):
    """Return an agent class by short name (e.g. "dqn", "ppo", "sac")."""
    name = name.lower()
    if name in ("dqn", "double_dqn", "dueling_dqn"):
        from rl.agents.dqn import DQN
        return DQN
    if name == "reinforce":
        from rl.agents.reinforce import REINFORCE
        return REINFORCE
    if name == "a2c":
        from rl.agents.a2c import A2C
        return A2C
    if name == "ppo":
        from rl.agents.ppo import PPO
        return PPO
    if name == "ddpg":
        from rl.agents.ddpg import DDPG
        return DDPG
    if name == "td3":
        from rl.agents.td3 import TD3
        return TD3
    if name == "sac":
        from rl.agents.sac import SAC
        return SAC
    raise KeyError(f"Unknown agent '{name}'.")


AVAILABLE = ["dqn", "reinforce", "a2c", "ppo", "ddpg", "td3", "sac"]
