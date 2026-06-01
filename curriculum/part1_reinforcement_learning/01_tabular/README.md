# Module 01 — Tabular methods: Q-learning & SARSA

> Code: [`rl/agents/tabular.py`](../../../rl/agents/tabular.py) ·
> Notebook: `notebook.ipynb` · Envs: FrozenLake-v1, CliffWalking-v0, Taxi-v3

This is the module where RL stops being abstract. We store a table
`Q[state, action]` and update it from experience until it converges to `Q*`.
No neural networks. Once you've *watched* a value function form here, DQN is
just "the same idea, but the table is a neural network."

## 1. Temporal-Difference (TD) learning

We can't compute the Bellman equation directly — we don't know `P(s'|s,a)`. But
we can **sample** transitions by acting in the environment and nudge our estimate
toward what we observe. After experiencing `(s, a, r, s')`:

```
Q(s,a) ← Q(s,a) + α · [ target − Q(s,a) ]
                      └────── TD error ──────┘
```

`α` is the learning rate. The only question is: **what's the target?**

## 2. Q-learning (off-policy control)

Q-learning uses the **greedy** next action in its target:

```
target = r + γ · max_{a'} Q(s', a')
```

It learns the value of the *optimal* policy regardless of how it actually
behaved while collecting data (it can explore randomly and still converge to
`Q*`). That's what **off-policy** means — learn about one policy from another's
data.

```python
target = r + gamma * Q[s_next].max()      # the whole algorithm, one line
Q[s, a] += alpha * (target - Q[s, a])
```

## 3. SARSA (on-policy control)

SARSA uses the action the policy **actually took** next, `a'`:

```
target = r + γ · Q(s', a')        # a' ~ π (the same ε-greedy policy)
```

The name is the tuple it uses: **S**tate, **A**ction, **R**eward, next **S**tate,
next **A**ction. Because it accounts for its own exploration, SARSA learns a
*safer* policy — the classic demonstration is CliffWalking, where SARSA walks a
cautious path away from the cliff while Q-learning hugs the optimal-but-risky
edge.

**Q-learning vs SARSA in one sentence:** they differ *only* in whether the
bootstrap uses `max_a' Q` (off-policy) or `Q(s', a')` for the sampled `a'`
(on-policy). See it in [`tabular.py`](../../../rl/agents/tabular.py) — it's a
single `if`.

## 4. Exploration: ε-greedy

If we always act greedily we may never discover a better action. **ε-greedy**:
with probability ε act randomly, otherwise act greedily. We **decay ε** over
time — explore early, exploit once we've learned.

```python
eps = max(eps_end, eps * eps_decay)   # e.g. 1.0 → 0.01 over training
```

## 5. Run it

```bash
python -c "
import gymnasium as gym
from rl.agents.tabular import TabularAgent, train_tabular
env = gym.make('FrozenLake-v1', is_slippery=False)
agent = TabularAgent(env.observation_space.n, env.action_space.n, algo='q_learning')
train_tabular(env, agent, n_episodes=5000)
"
```

Watch the average return climb from ~0 toward 1.0 as the table fills in. Then
try `is_slippery=True` (stochastic transitions) and `algo='sarsa'` and compare.

## 6. Why we need to go deeper

Tabular methods are **provably convergent** and beautifully simple — but they
need one table entry per state. LunarLander's state is 8 *continuous* numbers:
infinitely many states, no table possible. We need to **generalize** across
similar states. That's a job for a function approximator — a neural network.

➡️ **Next:** [Module 02 — Value-based deep RL](../02_value_based/): replace the
table `Q[s,a]` with a neural network `Q(s,a; θ)` and land the lunar module.

## ✅ Check your understanding
1. What exactly is the TD error, in words and in symbols?
2. Change one line of Q-learning to get SARSA. Which line?
3. Why does Q-learning converge to `Q*` even while exploring randomly?
4. On CliffWalking, why does SARSA's learned path differ from Q-learning's?
5. Why does the tabular approach fail on LunarLander?

## 📖 Go deeper
- Sutton & Barto, Ch. 6 (TD learning) — the canonical treatment.
- Watkins & Dayan (1992), "Q-learning" — the original convergence proof.
