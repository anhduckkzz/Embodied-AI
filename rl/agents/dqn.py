"""Deep Q-Network (DQN) and its two most important upgrades.

This is the modernised, refactored descendant of the original repo's
``Agent.py``. One class supports three variants via flags:

* **DQN**         - vanilla (Mnih et al., 2015): target network + replay.
* **Double DQN**  - decouple action *selection* (online net) from action
  *evaluation* (target net) to fight Q-value overestimation.
* **Dueling DQN** - a network architecture (see ``DuelingQNetwork``) that
  splits state-value and advantage; orthogonal to Double, so combine freely.

Optionally uses prioritized experience replay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.buffers import PrioritizedReplayBuffer, ReplayBuffer
from rl.networks import DuelingQNetwork, QNetwork


@dataclass
class DQNConfig:
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 5e-4
    tau: float = 1e-3            # soft target update rate (Polyak averaging)
    update_every: int = 4       # gradient step every N environment steps
    hidden: tuple = (64, 64)
    double: bool = True         # Double DQN target by default (strict upgrade)
    dueling: bool = False
    prioritized: bool = False
    per_alpha: float = 0.6
    per_beta: float = 0.4
    learning_starts: int = 1000  # warm-up before any learning


class DQN:
    """A DQN agent that interacts with and learns from a discrete-action env."""

    def __init__(self, state_size: int, action_size: int, cfg: DQNConfig = None,
                 device="cpu", seed: int = 0):
        self.cfg = cfg or DQNConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        torch.manual_seed(seed)

        Net = DuelingQNetwork if self.cfg.dueling else QNetwork
        self.q_online = Net(state_size, action_size, self.cfg.hidden).to(device)
        self.q_target = Net(state_size, action_size, self.cfg.hidden).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=self.cfg.lr)

        if self.cfg.prioritized:
            self.memory = PrioritizedReplayBuffer(
                self.cfg.buffer_size, state_size, None, self.cfg.per_alpha, device, seed)
        else:
            self.memory = ReplayBuffer(self.cfg.buffer_size, state_size, None, device, seed)
        self.t_step = 0
        self.rng = np.random.default_rng(seed)

    # ---- acting -----------------------------------------------------------
    def act(self, state, eps: float = 0.0) -> int:
        """Epsilon-greedy action from the online Q-network."""
        if self.rng.random() < eps:
            return int(self.rng.integers(self.action_size))
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_online.eval()
        with torch.no_grad():
            q = self.q_online(state_t)
        self.q_online.train()
        return int(q.argmax(dim=1).item())

    # ---- experience + learning -------------------------------------------
    def step(self, state, action, reward, next_state, done) -> None:
        """Store a transition and learn on the configured schedule."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if (self.t_step % self.cfg.update_every == 0
                and len(self.memory) > max(self.cfg.batch_size, self.cfg.learning_starts)):
            self._learn()

    def _learn(self) -> None:
        if self.cfg.prioritized:
            (states, actions, rewards, next_states, dones), idx, weights = \
                self.memory.sample(self.cfg.batch_size, self.cfg.per_beta)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)
            weights = 1.0

        # --- target Q ---
        with torch.no_grad():
            if self.cfg.double:
                # select with online net, evaluate with target net
                next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
                q_next = self.q_target(next_states).gather(1, next_actions)
            else:
                q_next = self.q_target(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + self.cfg.gamma * q_next * (1.0 - dones)

        # --- current Q for the actions actually taken ---
        q_expected = self.q_online(states).gather(1, actions)

        td_error = q_target - q_expected
        loss = (weights * td_error.pow(2)).mean()  # weighted MSE (Huber also fine)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optimizer.step()

        if self.cfg.prioritized:
            self.memory.update_priorities(idx, td_error.detach().abs().cpu().numpy().squeeze())

        self._soft_update()

    def _soft_update(self) -> None:
        """Polyak averaging: theta_target <- tau*theta_online + (1-tau)*theta_target."""
        with torch.no_grad():
            for tp, op in zip(self.q_target.parameters(), self.q_online.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * op.data)

    # ---- persistence ------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"q_online": self.q_online.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["q_online"] if isinstance(ckpt, dict) and "q_online" in ckpt else ckpt
        self.q_online.load_state_dict(state)
        self.q_target.load_state_dict(state)
