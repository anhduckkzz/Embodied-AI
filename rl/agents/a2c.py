"""Advantage Actor-Critic (A2C) - the synchronous bridge to PPO.

A2C sits exactly between REINFORCE and PPO. Like REINFORCE it is a policy
gradient; like PPO it uses a learned critic and GAE advantages. The difference
from PPO is that it does a *single* gradient step per rollout with no clipping -
simpler, but less sample-efficient and less stable. Reading A2C right before
PPO makes the clipped objective's purpose obvious.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rl.buffers import RolloutBuffer
from rl.networks import CategoricalActor, GaussianActor, VNetwork


@dataclass
class A2CConfig:
    rollout_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0     # 1.0 -> n-step returns; lower for GAE
    lr: float = 7e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: tuple = (64, 64)


class A2C:
    def __init__(self, state_size: int, action_size: int, discrete: bool = True,
                 cfg: A2CConfig = None, device="cpu", seed: int = 0):
        self.cfg = cfg or A2CConfig()
        self.discrete = discrete
        self.device = device
        torch.manual_seed(seed)

        Actor = CategoricalActor if discrete else GaussianActor
        self.actor = Actor(state_size, action_size, self.cfg.hidden).to(device)
        self.critic = VNetwork(state_size, self.cfg.hidden).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.cfg.lr)
        self.buffer = RolloutBuffer(
            self.cfg.rollout_steps, state_size, None if discrete else action_size,
            self.cfg.gamma, self.cfg.gae_lambda, device)

    @torch.no_grad()
    def act(self, state, greedy: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.distribution(state_t)
        action = (dist.probs.argmax(-1) if self.discrete else dist.mean) if greedy else dist.sample()
        log_prob = dist.log_prob(action)
        if not self.discrete:
            log_prob = log_prob.sum(-1)
        value = self.critic(state_t)
        a = int(action.item()) if self.discrete else action.squeeze(0).cpu().numpy()
        return a, float(log_prob.item()), float(value.item())

    def update(self, last_value: float, last_done: bool):
        self.buffer.compute_returns_and_advantages(last_value, last_done)
        data = self.buffer.get(action_dtype=torch.int64 if self.discrete else torch.float32)

        dist = self.actor.distribution(data["states"])
        log_probs = dist.log_prob(data["actions"])
        if not self.discrete:
            log_probs = log_probs.sum(-1)
            entropy = dist.entropy().sum(-1).mean()
        else:
            entropy = dist.entropy().mean()
        values = self.critic(data["states"])

        policy_loss = -(log_probs * data["advantages"]).mean()
        value_loss = (data["returns"] - values).pow(2).mean()
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.cfg.max_grad_norm)
        self.optimizer.step()
        return {"policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item())}

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
