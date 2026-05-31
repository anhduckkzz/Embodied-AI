"""Fast sanity tests for the rl.* library.

These are deliberately lightweight - they check shapes, invariants and that a
few hundred updates *learn something*, not full convergence. Run with:

    pytest -q

Tests that need torch / gymnasium are skipped automatically if those packages
are absent, so the pure-numpy parts still run anywhere.
"""
import numpy as np
import pytest

from rl.utils import ExponentialSchedule, LinearSchedule, RunningMeanStd, moving_average

torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------- utils ----
def test_linear_schedule_endpoints():
    s = LinearSchedule(1.0, 0.0, 100)
    assert s.value(0) == 1.0
    assert s.value(100) == pytest.approx(0.0)
    assert s.value(50) == pytest.approx(0.5)


def test_exponential_schedule_floor():
    s = ExponentialSchedule(1.0, 0.1, 0.5)
    for _ in range(50):
        s.step()
    assert s.value == pytest.approx(0.1)


def test_running_mean_std_matches_numpy():
    rms = RunningMeanStd(shape=(3,))
    data = np.random.randn(1000, 3) * 5 + 2
    rms.update(data)
    assert np.allclose(rms.mean, data.mean(0), atol=0.1)
    assert np.allclose(np.sqrt(rms.var), data.std(0), atol=0.1)


def test_moving_average_length():
    assert len(moving_average(list(range(200)), 100)) == 101


# -------------------------------------------------------------- buffers ----
def test_replay_buffer_roundtrip():
    from rl.buffers import ReplayBuffer
    buf = ReplayBuffer(100, state_dim=4, action_dim=None)
    for i in range(150):  # overflow capacity to test the ring buffer
        buf.add(np.ones(4) * i, i % 2, float(i), np.ones(4) * (i + 1), i % 10 == 0)
    assert len(buf) == 100
    s, a, r, ns, d = buf.sample(16)
    assert s.shape == (16, 4) and a.shape == (16, 1) and r.shape == (16, 1)
    assert a.dtype == torch.int64


def test_gae_advantages_constant_reward():
    """With constant reward and zero values, GAE returns should be discounted sums."""
    from rl.buffers import RolloutBuffer
    buf = RolloutBuffer(5, state_dim=2, action_dim=None, gamma=0.9, gae_lambda=1.0)
    for _ in range(5):
        buf.add(np.zeros(2), 0, 1.0, 0.0, 0.0, False)
    buf.compute_returns_and_advantages(last_value=0.0, last_done=True)
    # Last step return = 1, previous = 1 + 0.9*next.
    expected_last = 1.0
    assert buf.returns[4] == pytest.approx(expected_last)
    assert buf.returns[3] == pytest.approx(1 + 0.9 * 1.0)


# ------------------------------------------------------------- networks ----
def test_network_output_shapes():
    from rl.networks import (CategoricalActor, ContinuousQNetwork, GaussianActor,
                             QNetwork, SquashedGaussianActor, VNetwork)
    x = torch.randn(8, 6)
    assert QNetwork(6, 4)(x).shape == (8, 4)
    assert VNetwork(6)(x).shape == (8,)
    a, logp, ent = CategoricalActor(6, 4)(x)
    assert a.shape == (8,) and logp.shape == (8,)
    a, logp, ent = GaussianActor(6, 3)(x)
    assert a.shape == (8, 3) and logp.shape == (8,)
    act, logp = SquashedGaussianActor(6, 3)(x)
    assert act.shape == (8, 3) and logp.shape == (8,)
    assert (act.abs() <= 1.0 + 1e-5).all()  # tanh-squashed into [-1, 1]
    assert ContinuousQNetwork(6, 3)(x, a).shape == (8,)


# ----------------------------------------------------- tabular learning ----
def test_tabular_q_learning_learns_trivial_mdp():
    """A 2-state chain where action 1 always gives +1: Q must prefer action 1."""
    from rl.agents.tabular import TabularAgent
    agent = TabularAgent(n_states=2, n_actions=2, algo="q_learning",
                         alpha=0.5, eps_start=0.3, eps_decay=1.0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        s = int(rng.integers(2))
        a = agent.act(s)
        r = 1.0 if a == 1 else 0.0
        agent.update(s, a, r, s, done=False)
    assert agent.Q[0, 1] > agent.Q[0, 0]
    assert agent.Q[1, 1] > agent.Q[1, 0]


# -------------------------------------------------------- smoke: agents ----
def test_dqn_agent_acts_and_learns():
    from rl.agents.dqn import DQN, DQNConfig
    agent = DQN(8, 4, DQNConfig(batch_size=8, learning_starts=10, buffer_size=1000))
    s = np.random.randn(8).astype(np.float32)
    assert isinstance(agent.act(s, eps=0.0), int)
    for _ in range(50):
        agent.step(np.random.randn(8), np.random.randint(4), 1.0,
                   np.random.randn(8), False)
    # No exception == buffers + learning wired correctly.


def test_ppo_update_runs():
    from rl.agents.ppo import PPO, PPOConfig
    agent = PPO(8, 4, discrete=True, cfg=PPOConfig(rollout_steps=32, minibatch_size=8, epochs=2))
    s = np.random.randn(8).astype(np.float32)
    for _ in range(32):
        a, lp, v = agent.act(s)
        agent.buffer.add(s, a, 1.0, v, lp, False)
    stats = agent.update(last_value=0.0, last_done=True)
    assert "policy_loss" in stats and np.isfinite(stats["policy_loss"])
