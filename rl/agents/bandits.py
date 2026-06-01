"""Multi-armed bandits - the simplest RL problem, and where exploration is born.

A bandit has ``k`` arms; each pull of arm ``a`` returns a reward from an unknown
distribution with mean ``q*(a)``. No states, no transitions - just the pure
**exploration vs exploitation** dilemma: pull the arm you *think* is best, or try
others to learn more? Every exploration strategy in deep RL (epsilon-greedy in
DQN, entropy in SAC, curiosity bonuses) is a descendant of these ideas.

We implement the three canonical strategies and a runner that reports the
*regret* (reward lost vs always pulling the true-best arm).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BernoulliBandit:
    """A k-armed bandit with Bernoulli rewards (each arm pays 1 w.p. p_a, else 0)."""

    def __init__(self, probs, seed: int = 0):
        self.probs = np.asarray(probs, float)
        self.k = len(self.probs)
        self.best = self.probs.max()
        self.rng = np.random.default_rng(seed)

    def pull(self, a: int) -> float:
        return float(self.rng.random() < self.probs[a])

    def regret(self, a: int) -> float:
        return self.best - self.probs[a]   # expected reward lost this step


@dataclass
class EpsilonGreedy:
    """Explore randomly with prob epsilon, otherwise exploit the best estimate."""

    k: int
    eps: float = 0.1
    seed: int = 0

    def __post_init__(self):
        self.q = np.zeros(self.k)     # value estimates
        self.n = np.zeros(self.k)     # pull counts
        self.rng = np.random.default_rng(self.seed)

    def select(self) -> int:
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.q))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]   # incremental sample average


@dataclass
class UCB1:
    """Upper Confidence Bound: optimism in the face of uncertainty.

    Pick argmax_a [ q(a) + c*sqrt(ln t / n(a)) ]. The bonus shrinks as an arm is
    pulled, so under-explored arms get tried - principled exploration, no epsilon.
    """

    k: int
    c: float = 2.0

    def __post_init__(self):
        self.q = np.zeros(self.k)
        self.n = np.zeros(self.k)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        if (self.n == 0).any():
            return int(np.argmin(self.n))   # try each arm once first
        bonus = self.c * np.sqrt(np.log(self.t) / self.n)
        return int(np.argmax(self.q + bonus))

    def update(self, a: int, r: float) -> None:
        self.n[a] += 1
        self.q[a] += (r - self.q[a]) / self.n[a]


@dataclass
class ThompsonSampling:
    """Bayesian: keep a Beta posterior per arm, sample from it, pull the argmax.

    Naturally balances exploration/exploitation and is often the strongest
    simple method. Beta(alpha, beta) is the conjugate prior for Bernoulli.
    """

    k: int
    seed: int = 0

    def __post_init__(self):
        self.alpha = np.ones(self.k)   # successes + 1
        self.beta = np.ones(self.k)    # failures + 1
        self.rng = np.random.default_rng(self.seed)

    def select(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, a: int, r: float) -> None:
        if r > 0.5:
            self.alpha[a] += 1
        else:
            self.beta[a] += 1


def run_bandit(bandit: BernoulliBandit, agent, steps: int = 2000):
    """Run an agent on a bandit; return (rewards, cumulative_regret) arrays."""
    rewards = np.zeros(steps)
    cum_regret = np.zeros(steps)
    total = 0.0
    for t in range(steps):
        a = agent.select()
        r = bandit.pull(a)
        agent.update(a, r)
        rewards[t] = r
        total += bandit.regret(a)
        cum_regret[t] = total
    return rewards, cum_regret
