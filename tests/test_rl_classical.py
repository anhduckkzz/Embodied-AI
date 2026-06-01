"""Tests for the classical/foundational RL additions: bandits, DP, Dyna-Q, LQR."""
import numpy as np
import pytest


# ------------------------------------------------------------- bandits ----
def test_bandit_strategies_beat_random():
    from rl.agents.bandits import (BernoulliBandit, EpsilonGreedy, ThompsonSampling,
                                   UCB1, run_bandit)
    probs = [0.2, 0.5, 0.75, 0.3]      # arm 2 is best
    steps = 3000
    results = {}
    for name, agent in [("eps", EpsilonGreedy(4, eps=0.1)),
                        ("ucb", UCB1(4, c=2.0)),
                        ("ts", ThompsonSampling(4))]:
        bandit = BernoulliBandit(probs, seed=0)
        _, regret = run_bandit(bandit, agent, steps)
        results[name] = regret[-1] / steps    # average regret
    # All should drive average regret well below the random-policy baseline.
    random_regret = np.mean([0.75 - p for p in probs])   # ~0.3
    for name, avg in results.items():
        assert avg < random_regret * 0.6, f"{name} regret too high: {avg}"


# ---------------------------------------------------- dynamic programming ----
def _chain_mdp():
    """3-state chain: action 1 moves right (+ reward at the end), action 0 stays."""
    n_s, n_a = 3, 2
    P = np.zeros((n_s, n_a, n_s))
    R = np.zeros((n_s, n_a, n_s))
    for s in range(n_s):
        P[s, 0, s] = 1.0                      # action 0: stay
        nxt = min(s + 1, n_s - 1)
        P[s, 1, nxt] = 1.0                    # action 1: move right
        if nxt == n_s - 1 and s != n_s - 1:
            R[s, 1, nxt] = 1.0                # reward for reaching the goal
    return P, R


def test_value_and_policy_iteration_agree():
    from rl.agents.dynamic_programming import policy_iteration, value_iteration
    P, R = _chain_mdp()
    V_vi, pi_vi = value_iteration(P, R, gamma=0.9)
    V_pi, pi_pi = policy_iteration(P, R, gamma=0.9)
    assert np.allclose(V_vi, V_pi, atol=1e-4)
    assert np.array_equal(pi_vi, pi_pi)
    # Optimal policy should prefer moving right from the start state.
    assert pi_vi[0] == 1


# -------------------------------------------------------------- Dyna-Q ----
def test_dyna_q_learns_chain_fast():
    import gymnasium as gym
    from rl.agents.dyna import DynaQ, train_dyna
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = DynaQ(env.observation_space.n, env.action_space.n,
                  planning_steps=30, eps=0.2)
    train_dyna(env, agent, n_episodes=80)
    # Evaluate the GREEDY policy (no exploration) - the true test of learning.
    successes = 0
    for _ in range(20):
        s, _ = env.reset()
        for _ in range(100):
            s, r, term, trunc, _ = env.step(agent.act(s, greedy=True))
            if term or trunc:
                successes += r
                break
    assert successes / 20 > 0.9      # planning solves this small MDP reliably


# ---------------------------------------------------------------- LQR ----
def test_lqr_stabilizes_double_integrator():
    from robotics.control import lqr_gain
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])      # double integrator (position, velocity)
    B = np.array([[0], [dt]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[0.1]])
    K = lqr_gain(A, B, Q, R)
    # Closed-loop A - B K must be stable (all eigenvalues inside the unit circle).
    eig = np.linalg.eigvals(A - B @ K)
    assert np.all(np.abs(eig) < 1.0)
    # Simulate from a disturbance; state should decay to ~0.
    x = np.array([1.0, 0.0])
    for _ in range(200):
        x = (A - B @ K) @ x
    assert np.linalg.norm(x) < 1e-2
