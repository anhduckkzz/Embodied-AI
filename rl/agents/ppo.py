"""Proximal Policy Optimization (PPO) - the modern default.

PPO is the workhorse of applied RL: it trains robot controllers, game agents,
and is the policy-gradient algorithm behind RLHF for LLMs and much VLA
post-training. It keeps the simplicity of policy gradients but fixes their
instability with a **clipped surrogate objective** that stops each update from
moving the policy too far from the data-collecting policy.

Pipeline per iteration:
  1. Roll out the current policy for ``rollout_steps`` (an actor-critic).
  2. Compute advantages with GAE (see ``RolloutBuffer``).
  3. Do several epochs of minibatch SGD on the clipped objective.

Supports both discrete (Categorical) and continuous (Gaussian) actions, so the
same class trains LunarLander-v3 *and* its continuous, robotics-flavoured twin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl.buffers import RolloutBuffer
from rl.networks import CategoricalActor, GaussianActor, VNetwork


@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2     # the epsilon in the clipped objective
    lr: float = 3e-4
    epochs: int = 10            # optimisation passes over each rollout
    minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.015    # early-stop epochs if policy moves too far
    hidden: tuple = (64, 64)


class PPO:
    def __init__(self, state_size: int, action_size: int, discrete: bool = True,
                 cfg: PPOConfig = None, device="cpu", seed: int = 0):
        self.cfg = cfg or PPOConfig()
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

    # ---- acting -----------------------------------------------------------
    @torch.no_grad()
    def act(self, state, greedy: bool = False):
        """Return (action, log_prob, value) for environment interaction."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.distribution(state_t)
        if greedy:
            action = dist.probs.argmax(-1) if self.discrete else dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if not self.discrete:
            log_prob = log_prob.sum(-1)
        value = self.critic(state_t)
        a = int(action.item()) if self.discrete else action.squeeze(0).cpu().numpy()
        return a, float(log_prob.item()), float(value.item())

    # ---- learning ---------------------------------------------------------
    def _evaluate(self, states, actions):
        """Re-evaluate log-probs, entropy and values under the *current* policy."""
        dist = self.actor.distribution(states)
        log_probs = dist.log_prob(actions)
        if not self.discrete:
            log_probs = log_probs.sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            entropy = dist.entropy()
        values = self.critic(states)
        return log_probs, entropy, values

    def update(self, last_value: float, last_done: bool):
        """Run the PPO update on one full rollout. Returns a metrics dict."""
        self.buffer.compute_returns_and_advantages(last_value, last_done)
        data = self.buffer.get(action_dtype=torch.int64 if self.discrete else torch.float32)
        n = data["states"].shape[0]
        idx = np.arange(n)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        for epoch in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                mb = idx[start:start + self.cfg.minibatch_size]
                log_probs, entropy, values = self._evaluate(
                    data["states"][mb], data["actions"][mb])

                # Probability ratio r = pi_new / pi_old.
                ratio = torch.exp(log_probs - data["log_probs"][mb])
                adv = data["advantages"][mb]

                # Clipped surrogate objective (the heart of PPO).
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio,
                                      1 + self.cfg.clip_ratio) * adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = (data["returns"][mb] - values).pow(2).mean()
                entropy_loss = entropy.mean()

                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (data["log_probs"][mb] - log_probs).mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["approx_kl"] += approx_kl
                n_updates += 1

            # Early stopping protects against destructively large updates.
            if self.cfg.target_kl is not None and abs(approx_kl) > 1.5 * self.cfg.target_kl:
                break

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
