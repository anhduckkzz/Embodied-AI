"""REINFORCE - the original (Monte-Carlo) policy gradient.

The conceptual root of every policy-gradient method, including PPO and the
RLHF/VLA fine-tuning you are heading toward. The idea in one line:

    increase the log-probability of actions that led to high return,
    decrease it for actions that led to low return.

Gradient:  grad J = E[ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t)) ]

where ``G_t`` is the return-to-go and ``b`` is an optional baseline (a learned
value function) that reduces variance *without* adding bias. We support both
discrete (Categorical) and continuous (Gaussian) policies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl.networks import CategoricalActor, GaussianActor, VNetwork


@dataclass
class ReinforceConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    hidden: tuple = (64, 64)
    use_baseline: bool = True   # REINFORCE-with-baseline (a.k.a. vanilla PG)
    entropy_coef: float = 0.0   # small positive value encourages exploration


class REINFORCE:
    def __init__(self, state_size: int, action_size: int, discrete: bool = True,
                 cfg: ReinforceConfig = None, device="cpu", seed: int = 0):
        self.cfg = cfg or ReinforceConfig()
        self.discrete = discrete
        self.device = device
        torch.manual_seed(seed)

        Actor = CategoricalActor if discrete else GaussianActor
        self.actor = Actor(state_size, action_size, self.cfg.hidden).to(device)
        params = list(self.actor.parameters())
        if self.cfg.use_baseline:
            self.value = VNetwork(state_size, self.cfg.hidden).to(device)
            params += list(self.value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr)
        self._reset_episode()

    def _reset_episode(self):
        self._log_probs, self._rewards, self._states, self._entropies = [], [], [], []

    def act(self, state, greedy: bool = False):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.actor.distribution(state_t)
        if greedy:
            action = dist.probs.argmax(-1) if self.discrete else dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if not self.discrete:
            log_prob = log_prob.sum(-1)
        # cache for the end-of-episode update
        self._log_probs.append(log_prob.squeeze(0))
        self._entropies.append(dist.entropy().sum())
        self._states.append(state_t.squeeze(0))
        return action.squeeze(0).cpu().numpy() if not self.discrete else int(action.item())

    def record_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    def finish_episode(self) -> float:
        """Compute returns-to-go and take one policy-gradient step. Returns loss."""
        # Discounted return-to-go G_t.
        returns, G = [], 0.0
        for r in reversed(self._rewards):
            G = r + self.cfg.gamma * G
            returns.insert(0, G)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        log_probs = torch.stack(self._log_probs)
        entropy = torch.stack(self._entropies).mean()

        if self.cfg.use_baseline:
            states = torch.stack(self._states)
            values = self.value(states)
            advantages = returns - values
            value_loss = advantages.pow(2).mean()
            advantages = advantages.detach()
        else:
            # Standardise returns as a cheap, biased-but-effective baseline.
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            value_loss = torch.tensor(0.0, device=self.device)

        policy_loss = -(log_probs * advantages).mean()
        loss = policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._reset_episode()
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
