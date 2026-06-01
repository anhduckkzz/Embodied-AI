"""Soft Actor-Critic (SAC) - the state-of-the-art off-policy continuous method.

SAC is the default choice for sample-efficient continuous control and a direct
ancestor of methods used to train real robots. Its key idea is **maximum
entropy RL**: maximise reward *and* policy entropy, i.e.

    J = E[ sum_t r_t + alpha * H(pi(.|s_t)) ].

Acting as randomly as possible while still solving the task yields robust,
well-explored policies. SAC combines this with TD3-style twin critics and a
stochastic, reparameterised (squashed-Gaussian) actor. The temperature
``alpha`` is auto-tuned to hit a target entropy, so there is essentially one
fewer hyper-parameter to babysit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rl.buffers import ReplayBuffer
from rl.networks import ContinuousQNetwork, SquashedGaussianActor


@dataclass
class SACConfig:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    lr: float = 3e-4
    hidden: tuple = (256, 256)
    autotune_alpha: bool = True      # learn the entropy temperature
    alpha: float = 0.2               # used if autotune is off
    learning_starts: int = 10_000


class SAC:
    def __init__(self, state_size: int, action_size: int, action_scale: float = 1.0,
                 cfg: SACConfig = None, device="cpu", seed: int = 0):
        self.cfg = cfg or SACConfig()
        self.device = device
        self.action_size = action_size
        self.action_scale = action_scale
        torch.manual_seed(seed)

        self.actor = SquashedGaussianActor(state_size, action_size, self.cfg.hidden, action_scale).to(device)
        self.critic1 = ContinuousQNetwork(state_size, action_size, self.cfg.hidden).to(device)
        self.critic2 = ContinuousQNetwork(state_size, action_size, self.cfg.hidden).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.cfg.lr)

        # Automatic entropy tuning: target entropy = -|A| is the common heuristic.
        if self.cfg.autotune_alpha:
            self.target_entropy = -float(action_size)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.cfg.alpha

        self.memory = ReplayBuffer(self.cfg.buffer_size, state_size, action_size, device, seed)

    @torch.no_grad()
    def act(self, state, deterministic: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(state_t, deterministic=deterministic, with_logprob=False)
        return action.cpu().numpy().squeeze(0)

    def step(self, *transition) -> None:
        self.memory.add(*transition)
        if len(self.memory) > max(self.cfg.batch_size, self.cfg.learning_starts):
            self._learn()

    def _learn(self) -> None:
        states, actions, rewards, next_states, dones = self.memory.sample(self.cfg.batch_size)
        rewards, dones = rewards.squeeze(-1), dones.squeeze(-1)

        # --- critic update with entropy-augmented target ---
        with torch.no_grad():
            next_actions, next_logp = self.actor(next_states)
            q1 = self.critic1_target(next_states, next_actions)
            q2 = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1, q2) - self.alpha * next_logp
            q_target = rewards + self.cfg.gamma * (1 - dones) * q_next

        critic_loss = (F.mse_loss(self.critic1(states, actions), q_target)
                       + F.mse_loss(self.critic2(states, actions), q_target))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- actor update: maximise Q - alpha*logp (reparameterised) ---
        new_actions, logp = self.actor(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * logp - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- temperature update ---
        if self.cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for net, target in ((self.critic1, self.critic1_target),
                            (self.critic2, self.critic2_target)):
            with torch.no_grad():
                for tp, p in zip(target.parameters(), net.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic1": self.critic1.state_dict(),
                    "critic2": self.critic2.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
