"""Experience storage.

Two fundamentally different needs in RL:

* **Off-policy** methods (DQN, DDPG, TD3, SAC) learn from a large *replay
  buffer* of past transitions, sampled uniformly (or by priority).
* **On-policy** methods (REINFORCE, A2C, PPO) learn from a *rollout* of the
  current policy and then throw it away, but first compute advantages with
  Generalised Advantage Estimation (GAE).

Both live here so the contrast is explicit.
"""
from __future__ import annotations

from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size uniform replay buffer for off-policy learning.

    Stores transitions in pre-allocated NumPy arrays (fast, low GC pressure)
    and returns torch tensors on ``sample``. Works for both discrete and
    continuous actions - pass ``action_dim=None`` for scalar discrete actions.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: Optional[int],
                 device="cpu", seed: int = 0):
        self.capacity = int(capacity)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        if action_dim is None:  # discrete: store action index
            self.actions = np.zeros((self.capacity, 1), dtype=np.int64)
        else:
            self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        to_t = lambda a, dt=torch.float32: torch.as_tensor(a[idx], dtype=dt, device=self.device)
        return (
            to_t(self.states),
            to_t(self.actions, torch.int64 if self.actions.dtype == np.int64 else torch.float32),
            to_t(self.rewards),
            to_t(self.next_states),
            to_t(self.dones),
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay (Schaul et al., 2016).

    Transitions are sampled in proportion to their last TD-error magnitude, so
    'surprising' experiences are revisited more often. Importance-sampling
    weights correct the resulting bias. A simple O(n) implementation - clear to
    read; swap in a sum-tree for large buffers.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: Optional[int],
                 alpha: float = 0.6, device="cpu", seed: int = 0):
        self.base = ReplayBuffer(capacity, state_dim, action_dim, device, seed)
        self.alpha = alpha
        self.priorities = np.zeros((self.base.capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *transition) -> None:
        self.priorities[self.base.ptr] = self.max_priority  # new samples = max priority
        self.base.add(*transition)

    def sample(self, batch_size: int, beta: float = 0.4):
        prios = self.priorities[: self.base.size] ** self.alpha
        probs = prios / prios.sum()
        idx = self.base.rng.choice(self.base.size, batch_size, p=probs)

        weights = (self.base.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        dev = self.base.device
        to_t = lambda a, dt=torch.float32: torch.as_tensor(a[idx], dtype=dt, device=dev)
        batch = (
            to_t(self.base.states),
            to_t(self.base.actions,
                 torch.int64 if self.base.actions.dtype == np.int64 else torch.float32),
            to_t(self.base.rewards),
            to_t(self.base.next_states),
            to_t(self.base.dones),
        )
        return batch, idx, torch.as_tensor(weights, dtype=torch.float32, device=dev).unsqueeze(1)

    def update_priorities(self, idx, td_errors) -> None:
        prios = np.abs(td_errors) + 1e-6
        self.priorities[idx] = prios
        self.max_priority = max(self.max_priority, float(prios.max()))

    def __len__(self) -> int:
        return len(self.base)


class RolloutBuffer:
    """On-policy storage for a fixed-length rollout, with GAE advantage computation.

    Collect ``(s, a, r, value, log_prob, done)`` for ``size`` steps, then call
    ``compute_returns_and_advantages`` once. Used by A2C and PPO.
    """

    def __init__(self, size: int, state_dim: int, action_dim: Optional[int],
                 gamma: float = 0.99, gae_lambda: float = 0.95, device="cpu"):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        act_shape = (size,) if action_dim is None else (size, action_dim)

        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros(act_shape, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, state, action, reward, value, log_prob, done) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: float, last_done: bool) -> None:
        """Generalised Advantage Estimation (Schulman et al., 2016).

        A_t = sum_l (gamma*lambda)^l * delta_{t+l},  delta_t = r_t + gamma V_{t+1} - V_t.
        ``lambda`` trades bias (0 -> TD, high bias) for variance (1 -> Monte
        Carlo, high variance). Returns = advantages + values (the value target).
        """
        adv = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_nonterminal = 1.0 - float(last_done)
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t + 1]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + self.gamma * self.gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[: self.ptr] = self.advantages[: self.ptr] + self.values[: self.ptr]

    def get(self, action_dtype=torch.float32):
        """Return the whole rollout as tensors and reset the write pointer."""
        n = self.ptr
        dev = self.device
        adv = self.advantages[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise advantages
        out = dict(
            states=torch.as_tensor(self.states[:n], device=dev),
            actions=torch.as_tensor(self.actions[:n], dtype=action_dtype, device=dev),
            log_probs=torch.as_tensor(self.log_probs[:n], device=dev),
            advantages=torch.as_tensor(adv, dtype=torch.float32, device=dev),
            returns=torch.as_tensor(self.returns[:n], device=dev),
            values=torch.as_tensor(self.values[:n], device=dev),
        )
        self.ptr = 0
        return out

    def __len__(self) -> int:
        return self.ptr
