"""Dynamic Programming: solving an MDP when you KNOW the model.

Before learning-from-experience (Q-learning, DQN), understand the case where the
transition probabilities ``P(s'|s,a)`` and rewards ``R(s,a,s')`` are **given**.
Then the Bellman optimality equation can be solved directly by iteration. This is
the *planning* baseline and the conceptual parent of model-based RL (Part 1,
module on model-based methods): a learned model + DP-style planning.

Two classic algorithms, both guaranteed to converge to the optimal policy:

* **Value Iteration** - repeatedly apply the Bellman optimality backup.
* **Policy Iteration** - alternate policy *evaluation* and greedy *improvement*.

The MDP is specified by arrays: P[s, a, s'] (probabilities) and R[s, a, s'].
"""
from __future__ import annotations

import numpy as np


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.99,
                    theta: float = 1e-8, max_iters: int = 10_000):
    """Solve for V* and the greedy optimal policy by Bellman optimality backups.

    V_{k+1}(s) = max_a  sum_s' P[s,a,s'] ( R[s,a,s'] + gamma V_k(s') )
    """
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    for _ in range(max_iters):
        # Q[s,a] = sum_s' P (R + gamma V)
        Q = np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new
    Q = np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])
    policy = Q.argmax(axis=1)
    return V, policy


def policy_evaluation(policy: np.ndarray, P: np.ndarray, R: np.ndarray,
                      gamma: float = 0.99, theta: float = 1e-8) -> np.ndarray:
    """Compute V^pi for a fixed (deterministic) policy by iterative backups."""
    n_states = P.shape[0]
    V = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            a = policy[s]
            v = np.sum(P[s, a] * (R[s, a] + gamma * V))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            return V


def policy_iteration(P: np.ndarray, R: np.ndarray, gamma: float = 0.99,
                     max_iters: int = 1000):
    """Alternate evaluation + greedy improvement until the policy is stable."""
    n_states, n_actions, _ = P.shape
    policy = np.zeros(n_states, dtype=int)
    for _ in range(max_iters):
        V = policy_evaluation(policy, P, R, gamma)
        Q = np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])
        new_policy = Q.argmax(axis=1)
        if np.array_equal(new_policy, policy):
            return V, policy          # stable => optimal
        policy = new_policy
    return V, policy
