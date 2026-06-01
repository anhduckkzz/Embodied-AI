# Module 10 — Model-Based RL: learn a model, then plan/imagine

> Code: [`rl/agents/dyna.py`](../../../rl/agents/dyna.py).
> Model-free RL (DQN, PPO, SAC) is powerful but **sample-hungry** — millions of
> interactions. Model-based RL learns a **model of the environment** and uses it
> to *plan* or to *generate imagined experience*, often reaching the same
> performance with **10–100× fewer real samples**. For real robots, where samples
> are expensive and dangerous, this is huge.

## 1. The core idea

A **model** predicts the consequences of actions: `(s, a) → (s', r)` (and maybe
done). Given a model you can:
- **Plan** — search over imagined action sequences (e.g. MCTS, trajectory
  optimization) without touching the real world.
- **Generate data** — roll out the model to train a value/policy on *imagined*
  transitions (Dyna-style).

```
real experience ──► learn model P̂(s'|s,a), R̂ ──► plan / imagine ──► improve policy
       ▲                                                                   │
       └───────────────────── act in the real world ──────────────────────┘
```

## 2. Dyna-Q — the simplest, clearest version (runnable)

Sutton's **Dyna-Q** does, each real step: (1) a normal Q-learning update, (2)
record the transition in a tabular model, (3) `n` **planning** updates from
random remembered transitions. Each real sample is amplified by `n` imagined
ones → dramatically faster learning.

```python
from rl.agents.dyna import DynaQ, train_dyna
agent = DynaQ(n_states, n_actions, planning_steps=30)
train_dyna(env, agent, n_episodes=80)   # solves small MDPs in far fewer episodes
```

Try `planning_steps=0` (pure Q-learning) vs `30` and compare episodes-to-solve —
the gap *is* the value of a model.

## 3. Deep model-based RL — the modern lineage

When the state is high-dimensional (images, robot states), the model is a **neural
network "world model."**

- **PETS / MPC-style** — learn a probabilistic dynamics net, plan each step with
  sampling-based MPC (CEM). Strong on continuous control from few samples.
- **MBPO** — learn an ensemble dynamics model, generate *short* imagined rollouts
  to augment a model-free learner (SAC). Best-of-both sample efficiency.
- **World Models (Ha & Schmidhuber)** — compress observations (VAE) + learn a
  recurrent dynamics model; train a tiny controller *inside the dream*.
- **Dreamer (v1–v3)** — learn a **latent** world model from pixels and train the
  actor-critic purely on **imagined latent rollouts**. DreamerV3 solves diverse
  tasks (including Minecraft diamonds) with one config — a landmark.
- **MuZero** — learn a model in a *value-equivalent* latent space and plan with
  **MCTS**; masters Go/chess/Atari **without being told the rules**. Its ancestor
  **AlphaZero** plans with MCTS over a *known* model (Module 08's planning, scaled).

## 4. Planning algorithms you'll meet
- **MCTS (Monte Carlo Tree Search)** — selectively grow a search tree using UCB
  (Module 09!) to balance exploring promising vs uncertain moves. The engine of
  AlphaGo/AlphaZero/MuZero.
- **Trajectory optimization / MPC** — optimize a finite-horizon action sequence
  against the model, execute the first action, replan (Part 14). Dominant in
  model-based robot control.
- **CEM (Cross-Entropy Method)** — sample action sequences, keep the elite, refit
  — simple, robust planning used in PETS.

## 5. The trade-off (why model-free still rules in many places)
- **Pros:** sample efficiency, transfer, planning/lookahead, safety (test in
  imagination first).
- **Cons:** **model bias** — a wrong model leads the policy astray ("hallucinated"
  rewards); learning a good model of complex dynamics/contacts is itself hard.
- Rule of thumb: model-based shines when **samples are expensive** (real robots)
  or **a model is available/learnable**; model-free when samples are cheap
  (fast sims, Part 4) and dynamics are nasty.

## 🛠️ Project
1. On FrozenLake/Taxi, plot episodes-to-solve for Dyna-Q with `planning_steps ∈
   {0, 5, 30}`. Quantify the sample-efficiency win.
2. Read the DreamerV3 and MuZero papers; map each to "learn model → imagine/plan
   → improve policy" and to Modules 08 (planning) and 09 (MCTS uses UCB).
3. (Stretch) Implement CEM-MPC on `Pendulum` with a learned 2-layer dynamics net.

## ✅ Check your understanding
1. What is a "model" in RL, and the two ways you can use one?
2. In Dyna-Q, what exactly do the planning steps do, and why does it speed learning?
3. What does Dreamer train its policy on, and why is that sample-efficient?
4. How does MuZero differ from AlphaZero?
5. What is model bias, and when would you prefer model-free RL anyway?

## 📖 Go deeper
- Sutton & Barto, Ch. 8 (planning & learning / Dyna).
- Papers: PETS (Chua 2018), MBPO (Janner 2019), World Models (Ha 2018),
  DreamerV3 (Hafner 2023), MuZero (Schrittwieser 2020).

➡️ **Next:** [Module 11 — Distributional & advanced value methods](../11_advanced_value_distributional/).
