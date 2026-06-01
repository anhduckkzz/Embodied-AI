"""Tabular value-based control: Q-learning and SARSA.

This is where RL *clicks*. No neural networks, no PyTorch - just a table
``Q[state, action]`` and the update rule written almost exactly as it appears
in Sutton & Barto. Run it on FrozenLake / CliffWalking / Taxi to watch a value
function form. Everything later (DQN, PPO) is a way to scale these same ideas
to states too numerous to tabulate.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class TabularAgent:
    """Q-learning (off-policy) and SARSA (on-policy) in one small class.

    The *only* difference between the two is the bootstrap target:

    * Q-learning uses ``max_a' Q[s', a']``  (value of the greedy next action)
    * SARSA uses     ``Q[s', a']``          (value of the action actually taken)

    Off-policy vs on-policy in two lines - that is the whole distinction.
    """

    def __init__(self, n_states: int, n_actions: int, algo: str = "q_learning",
                 alpha: float = 0.1, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 0.999,
                 seed: int = 0):
        assert algo in ("q_learning", "sarsa")
        self.algo = algo
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def act(self, state: int, greedy: bool = False) -> int:
        """Epsilon-greedy action selection (greedy at evaluation time)."""
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        # Random tie-break keeps early exploration unbiased.
        q = self.Q[state]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool,
               a_next: Optional[int] = None) -> float:
        """Apply one TD update and return the TD error (for inspection)."""
        if done:
            target = r
        elif self.algo == "q_learning":
            target = r + self.gamma * self.Q[s_next].max()
        else:  # SARSA needs the next action that the policy actually took
            if a_next is None:
                a_next = self.act(s_next)
            target = r + self.gamma * self.Q[s_next, a_next]
        td_error = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train_tabular(env, agent: TabularAgent, n_episodes: int = 5000,
                  max_t: int = 200, log_every: int = 500, verbose: bool = True):
    """Generic training loop for a discrete-observation Gymnasium env.

    Returns the list of episode returns. Works with FrozenLake-v1,
    CliffWalking-v0, Taxi-v3, etc.
    """
    scores = []
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        action = agent.act(state)
        score = 0.0
        for _ in range(max_t):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if agent.algo == "sarsa":
                next_action = agent.act(next_state)
                agent.update(state, action, reward, next_state, done, next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)
                action = agent.act(next_state)
            state = next_state
            score += reward
            if done:
                break
        agent.decay_epsilon()
        scores.append(score)
        if verbose and ep % log_every == 0:
            avg = np.mean(scores[-log_every:])
            print(f"episode {ep:>6}  avg_return={avg:6.3f}  eps={agent.eps:.3f}")
    return scores
