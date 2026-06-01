"""Dyna-Q: the bridge between model-free and model-based RL.

Q-learning (model-free) throws each experience away after one update. Dyna-Q
(Sutton, 1990) instead **learns a model** of the environment from experience and
then **plans** with it - replaying simulated transitions to update Q many extra
times per real step. The result: far better sample efficiency, because each real
interaction is amplified by ``n`` planning updates.

This single idea scales all the way up to modern model-based RL (MBPO, Dreamer,
MuZero): learn a model, generate imagined data, train the policy/value on it.
Here the "model" is a tabular memory of observed (s,a) -> (r, s'); in deep
model-based RL it's a neural network "world model."
"""
from __future__ import annotations

import numpy as np


class DynaQ:
    """Tabular Dyna-Q for discrete state/action environments.

    Each real step does: (1) a normal Q-learning update, (2) record the
    transition in the model, (3) ``planning_steps`` simulated Q-updates from
    random previously-seen (s, a) pairs.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1,
                 gamma: float = 0.99, eps: float = 0.1, planning_steps: int = 20,
                 seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.planning_steps = planning_steps
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}                      # (s, a) -> (r, s_next)
        self.seen = []                       # list of observed (s, a)

    def act(self, s: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def _q_update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def step(self, s, a, r, s_next, done):
        # 1) direct (model-free) RL update from the real transition
        self._q_update(s, a, r, s_next, done)
        # 2) model learning: remember what happened
        if (s, a) not in self.model:
            self.seen.append((s, a))
        self.model[(s, a)] = (r, s_next, done)
        # 3) planning: replay imagined transitions from the learned model
        for _ in range(self.planning_steps):
            si, ai = self.seen[self.rng.integers(len(self.seen))]
            ri, sni, di = self.model[(si, ai)]
            self._q_update(si, ai, ri, sni, di)


def train_dyna(env, agent: DynaQ, n_episodes: int = 50, max_t: int = 200):
    """Train Dyna-Q on a discrete Gymnasium env; return per-episode returns."""
    returns = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total = 0.0
        for _ in range(max_t):
            a = agent.act(s)
            s_next, r, term, trunc, _ = env.step(a)
            agent.step(s, a, r, s_next, term)
            s = s_next
            total += r
            if term or trunc:
                break
        returns.append(total)
    return returns
