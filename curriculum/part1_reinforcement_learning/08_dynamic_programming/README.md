# Module 08 — Dynamic Programming: planning when you know the model

> Code: [`rl/agents/dynamic_programming.py`](../../../rl/agents/dynamic_programming.py).
> The *planning* baseline that all of RL approximates. If you **know** the MDP
> (transitions `P` and rewards `R`), you can compute the optimal policy exactly —
> no learning, no samples. Understanding this makes Q-learning, DQN, and
> model-based RL "obvious."

## 1. The setting

Given the full MDP `(S, A, P, R, γ)` with `P(s'|s,a)` and `R` **known**, solve the
Bellman optimality equation directly. DP exploits two properties: the problem has
**optimal substructure** (Bellman recursion) and **overlapping subproblems**
(reuse `V(s')`).

## 2. Value Iteration

Repeatedly apply the Bellman optimality **backup** until `V` stops changing:

```
V_{k+1}(s) = max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V_k(s') ]
```

Each sweep makes `V` more accurate; it provably converges to `V*` (the backup is
a γ-contraction). Then read off the greedy policy `π*(s) = argmax_a Q*(s,a)`.

```python
from rl.agents.dynamic_programming import value_iteration
V, policy = value_iteration(P, R, gamma=0.9)   # P[s,a,s'], R[s,a,s']
```

## 3. Policy Iteration

Alternate two steps until the policy stops changing:
1. **Policy evaluation** — solve `V^π` for the current policy (iterate the
   Bellman *expectation* backup, or solve the linear system).
2. **Policy improvement** — act greedily w.r.t. `V^π` to get a better policy.

```
π → evaluate V^π → improve (greedy) → π' → evaluate → ... → π*
```

Often converges in *very few* iterations (policies are discrete; there are
finitely many). Value iteration is policy iteration with a single evaluation
sweep — two ends of the same spectrum (generalized policy iteration).

## 4. Why this matters even though we rarely know `P`

- **Conceptual root:** TD methods (Module 01) are "DP with sampled backups" —
  replace the known `Σ_{s'} P(...)` expectation with a sampled `s'`. DQN is "DP
  with a neural net for `V`/`Q` and sampled backups." Seeing this demystifies them.
- **Model-based RL (Module 10):** *learn* `P`/`R` from data, then run DP-style
  planning on the learned model — this is exactly Dyna / MuZero / Dreamer.
- **Practical planners:** when you *do* have a model (board games, known robot
  dynamics, grid maps), DP/exact planning is the right tool — and it connects to
  A*/Dijkstra (Part 8), which are DP on shortest-path problems.

## 5. Generalized Policy Iteration (the unifying view)
Almost every RL algorithm is a dance between **evaluation** (make the value
estimate consistent with the policy) and **improvement** (make the policy greedy
w.r.t. the value). Keep this lens and the zoo of algorithms collapses into one idea.

## 🛠️ Project
1. Build the 3-state chain MDP from
   [`tests/test_rl_classical.py`](../../../tests/test_rl_classical.py); run
   `value_iteration` and `policy_iteration` and confirm they agree.
2. Model **FrozenLake** as `(P, R)` (Gymnasium exposes `env.P`) and solve it
   exactly with DP. Compare the optimal `V` to what tabular Q-learning (Module 01)
   *learned* from samples.
3. Vary γ and watch the optimal policy change from myopic to far-sighted.

## ✅ Check your understanding
1. What does each Bellman *optimality* backup do to the value estimate?
2. Value iteration vs policy iteration — how are they two ends of one idea?
3. Why does DP need the model, and what do TD methods replace it with?
4. How is model-based RL "DP on a learned model"?
5. Why does policy iteration often converge in few steps?

## 📖 Go deeper
- Sutton & Barto, Ch. 4 (Dynamic Programming) — the canonical treatment.
- Bertsekas, *Dynamic Programming and Optimal Control* (the deep reference).

➡️ **Next:** [Module 09 — Exploration & bandits](../09_exploration_bandits/).
