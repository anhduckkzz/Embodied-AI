"""Behavioral Cloning (BC) - supervised imitation, the gateway to VLA.

BC turns control into ordinary supervised learning: collect demonstrations
``(state, expert_action)`` and train a policy to copy them. No reward, no
exploration, no Bellman equation. This matters enormously for your VLA goal,
because today's Vision-Language-Action models (RT-2, OpenVLA, Octo, pi0) are
*overwhelmingly* trained by BC on large robot-demonstration datasets - RL is
then used to *refine* the imitation-pretrained policy.

Limitation to understand deeply: **covariate shift**. A BC policy only sees
expert states in training; a small mistake at test time leads to unfamiliar
states where errors compound. DAgger and RL fine-tuning are the usual fixes,
and this is precisely why RL remains essential even in a BC-dominated world.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.networks import CategoricalActor, GaussianActor


@dataclass
class BCConfig:
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    hidden: tuple = (64, 64)


class BehavioralCloning:
    """Fit a policy to a fixed dataset of (state, action) expert demonstrations."""

    def __init__(self, state_size: int, action_size: int, discrete: bool = True,
                 cfg: BCConfig = None, device="cpu", seed: int = 0):
        self.cfg = cfg or BCConfig()
        self.discrete = discrete
        self.device = device
        torch.manual_seed(seed)
        Actor = CategoricalActor if discrete else GaussianActor
        self.policy = Actor(state_size, action_size, self.cfg.hidden).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    def fit(self, states: np.ndarray, actions: np.ndarray, verbose: bool = True):
        """Train by maximum likelihood on the demonstrations. Returns loss history."""
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if self.discrete:
            actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        else:
            actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        n = states_t.shape[0]
        history = []
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                dist = self.policy.distribution(states_t[idx])
                if self.discrete:
                    # Negative log-likelihood = cross-entropy on action logits.
                    loss = F.cross_entropy(self.policy.logits(states_t[idx]), actions_t[idx])
                else:
                    log_prob = dist.log_prob(actions_t[idx]).sum(-1)
                    loss = -log_prob.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss)
            if verbose and (epoch + 1) % max(1, self.cfg.epochs // 10) == 0:
                print(f"epoch {epoch + 1:>3}  loss={epoch_loss:.4f}")
        return history

    @torch.no_grad()
    def act(self, state, greedy: bool = True):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.policy.distribution(state_t)
        if self.discrete:
            action = dist.probs.argmax(-1) if greedy else dist.sample()
            return int(action.item())
        action = dist.mean if greedy else dist.sample()
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])


def collect_demonstrations(env, expert, n_episodes: int = 50, expert_is_agent: bool = True):
    """Roll out an expert to build a demonstration dataset.

    ``expert`` is anything with an ``act(state)`` method (e.g. a trained DQN or
    SAC agent). Returns ``(states, actions)`` NumPy arrays ready for ``fit``.
    """
    states, actions = [], []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = expert.act(state) if expert_is_agent else expert(state)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    return np.array(states), np.array(actions)
