"""Deep Deterministic Policy Gradient (DDPG) - continuous control, off-policy.

DQN cannot handle continuous actions (you cannot ``argmax`` over an infinite
set). DDPG fixes this with an **actor-critic** where a deterministic actor
mu(s) proposes the action and a critic Q(s, a) scores it; the actor is trained
to climb the critic's gradient. This is the first algorithm in the curriculum
that controls a *torque/thrust* directly - the regime real robots live in.

DDPG is brittle (TD3 and SAC exist to fix that) but it is the clearest
introduction to continuous off-policy RL.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rl.buffers import ReplayBuffer
from rl.networks import ContinuousQNetwork, DeterministicActor


@dataclass
class DDPGConfig:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden: tuple = (256, 256)
    exploration_noise: float = 0.1   # std of Gaussian action noise
    learning_starts: int = 10_000


class DDPG:
    def __init__(self, state_size: int, action_size: int, action_scale: float = 1.0,
                 cfg: DDPGConfig = None, device="cpu", seed: int = 0):
        self.cfg = cfg or DDPGConfig()
        self.device = device
        self.action_size = action_size
        self.action_scale = action_scale
        torch.manual_seed(seed)

        self.actor = DeterministicActor(state_size, action_size, self.cfg.hidden, action_scale).to(device)
        self.critic = ContinuousQNetwork(state_size, action_size, self.cfg.hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.memory = ReplayBuffer(self.cfg.buffer_size, state_size, action_size, device, seed)
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, state, noise: bool = True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_t).cpu().numpy().squeeze(0)
        if noise:
            action += self.rng.normal(0, self.cfg.exploration_noise * self.action_scale,
                                      size=self.action_size)
        return np.clip(action, -self.action_scale, self.action_scale)

    def step(self, *transition) -> None:
        self.memory.add(*transition)
        if len(self.memory) > max(self.cfg.batch_size, self.cfg.learning_starts):
            self._learn()

    def _learn(self) -> None:
        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)

        # --- critic update: regress Q toward the one-step TD target ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards.squeeze(-1) + self.cfg.gamma * (1 - dones.squeeze(-1)) * q_next
        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor update: maximise Q(s, mu(s)) ---
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, target) -> None:
        with torch.no_grad():
            for tp, p in zip(target.parameters(), net.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
