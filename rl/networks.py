"""Reusable neural-network building blocks.

Every deep-RL agent in this package is "just" one or two of these networks
plus a learning rule. Pulling them out keeps the agents short and lets you
compare, say, the DQN Q-network against the SAC critic side by side.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes: Sequence[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    """Build a plain multi-layer perceptron from a list of layer sizes.

    ``sizes=[8, 64, 64, 4]`` builds 8->64->64->4 with ReLU between hidden
    layers and no activation on the output. This single helper backs almost
    every network below.
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """State-action value network for discrete action spaces (DQN).

    Maps an observation to one Q-value *per action*. The greedy policy is then
    ``argmax_a Q(s, a)``.
    """

    def __init__(self, state_size: int, action_size: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([state_size, *hidden, action_size])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DuelingQNetwork(nn.Module):
    """Dueling DQN: separate value V(s) and advantage A(s, a) streams.

    Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)). Decoupling "how good is this
    state" from "how much better is each action" speeds up learning when many
    actions have similar value.
    """

    def __init__(self, state_size: int, action_size: int, hidden=(64, 64)):
        super().__init__()
        self.feature = mlp([state_size, hidden[0]], output_activation=nn.ReLU)
        self.value = mlp([hidden[0], hidden[-1], 1])
        self.advantage = mlp([hidden[0], hidden[-1], action_size])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        z = self.feature(state)
        v = self.value(z)
        a = self.advantage(z)
        return v + (a - a.mean(dim=1, keepdim=True))


class CategoricalActor(nn.Module):
    """Stochastic policy for discrete actions: outputs a categorical distribution."""

    def __init__(self, state_size: int, action_size: int, hidden=(64, 64)):
        super().__init__()
        self.logits = mlp([state_size, *hidden, action_size])

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits(state))

    def forward(self, state: torch.Tensor):
        dist = self.distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class GaussianActor(nn.Module):
    """Stochastic policy for continuous actions with a state-independent std.

    The network outputs the mean; ``log_std`` is a free parameter vector. This
    is the standard PPO/A2C continuous-control head. Actions are *not* squashed
    here - the environment is expected to clip. (SAC uses a squashed variant,
    see ``SquashedGaussianActor``.)
    """

    def __init__(self, state_size: int, action_size: int, hidden=(64, 64)):
        super().__init__()
        self.mu = mlp([state_size, *hidden, action_size])
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_size))

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mu(state)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)

    def forward(self, state: torch.Tensor):
        dist = self.distribution(state)
        action = dist.sample()
        # Sum log-probs across action dimensions (independent Gaussians).
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, dist.entropy().sum(axis=-1)


class SquashedGaussianActor(nn.Module):
    """SAC actor: Gaussian + tanh squashing into [-1, 1], with corrected log-prob.

    The tanh keeps actions bounded; the log-prob correction term accounts for
    the change of variables so the entropy bonus stays exact.
    """

    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_size: int, action_size: int, hidden=(256, 256),
                 action_scale=1.0):
        super().__init__()
        self.body = mlp([state_size, *hidden], output_activation=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], action_size)
        self.log_std = nn.Linear(hidden[-1], action_size)
        self.action_scale = action_scale

    def forward(self, state: torch.Tensor, deterministic: bool = False,
                with_logprob: bool = True):
        z = self.body(state)
        mu = self.mu(z)
        log_std = torch.clamp(self.log_std(z), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        raw = mu if deterministic else dist.rsample()  # reparameterised sample
        action = torch.tanh(raw) * self.action_scale

        if with_logprob:
            logp = dist.log_prob(raw).sum(axis=-1)
            # tanh correction: log(1 - tanh(x)^2), numerically stable form
            logp -= (2 * (np.log(2) - raw - nn.functional.softplus(-2 * raw))).sum(axis=-1)
        else:
            logp = None
        return action, logp


class VNetwork(nn.Module):
    """State-value baseline V(s) used by REINFORCE-with-baseline, A2C and PPO."""

    def __init__(self, state_size: int, hidden=(64, 64)):
        super().__init__()
        self.net = mlp([state_size, *hidden, 1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ContinuousQNetwork(nn.Module):
    """Q(s, a) for continuous actions (DDPG / TD3 / SAC critics).

    Unlike the discrete ``QNetwork`` this *takes the action as input* and
    returns a single scalar value.
    """

    def __init__(self, state_size: int, action_size: int, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([state_size + action_size, *hidden, 1])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class DeterministicActor(nn.Module):
    """Deterministic policy mu(s) for DDPG / TD3, tanh-bounded to the action range."""

    def __init__(self, state_size: int, action_size: int, hidden=(256, 256),
                 action_scale=1.0):
        super().__init__()
        self.net = mlp([state_size, *hidden, action_size], output_activation=nn.Tanh)
        self.action_scale = action_scale

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_scale
