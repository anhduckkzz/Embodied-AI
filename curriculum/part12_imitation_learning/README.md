# Part 12 — Imitation Learning (the dominant paradigm for robots)

> Code: [`rl/agents/bc.py`](../../rl/agents/bc.py). RL learns from *reward*;
> imitation learns from *demonstrations*. For real robots — where reward is hard
> to design and exploration is dangerous — imitation is the **primary** way
> policies (and today's VLAs) are trained. This part goes far beyond the Part-1
> intro into the full landscape.

## 1. Why imitation dominates robotics

- Reward functions for real tasks ("set the table") are **brutally hard** to
  specify; demonstrations sidestep that.
- Exploration on hardware is **slow and unsafe**; demos give competent behavior
  immediately.
- Demos are **collectable at scale** via teleoperation (Part 9): ALOHA, Open
  X-Embodiment, LeRobot datasets.
- **Every major VLA (RT-2, OpenVLA, Octo, π0) is trained mostly by imitation.**

## 2. Behavioral Cloning (BC) — supervised control

Treat control as supervised learning on `(observation → expert action)`:
classification for discrete actions, regression (or a Gaussian/mixture/diffusion
head) for continuous. No environment, no reward, no Bellman.

```python
from rl.agents.bc import BehavioralCloning, collect_demonstrations
states, actions = collect_demonstrations(env, expert)   # from a teleop/RL expert
bc = BehavioralCloning(state_dim, action_dim, discrete=True); bc.fit(states, actions)
```

**The fatal flaw — covariate shift.** BC only sees expert states. At test time a
small error → unfamiliar state → bigger error → compounding failure. Errors grow
**quadratically** in horizon. This single fact motivates everything below.

## 3. Fixing covariate shift: interactive imitation

- **DAgger (Dataset Aggregation)** — run the *learner*, have the **expert label
  the states the learner actually visits**, append to the dataset, retrain.
  Directly attacks the distribution mismatch; needs an interactive/queryable
  expert. The canonical fix for BC.
- **DART** — inject noise into the expert demos so the dataset covers recovery
  states (a cheaper, non-interactive cousin).
- **Variants:** HG-DAgger / EnsembleDAgger only query the expert when the policy
  is uncertain or unsafe (human-gated).

## 4. Inverse Reinforcement Learning (IRL) — recover the *reward*

Instead of copying actions, **infer the reward function** the expert was
optimizing, then run RL on it. More robust to covariate shift (it learns *intent*,
generalizing beyond demonstrated states).
- **Maximum-Entropy IRL** — the principled formulation: find the reward under
  which the expert's behavior is maximally likely while staying max-entropy
  (handles suboptimal/noisy demos). 
- **Challenge:** IRL is ill-posed (many rewards explain the same behavior) and
  classically requires RL in an inner loop (expensive).

## 5. Adversarial imitation — GAIL & friends

**GAIL (Generative Adversarial Imitation Learning)** skips recovering an explicit
reward: a **discriminator** learns to tell expert state-actions from the policy's;
its output is used as the **reward** to train the policy (PPO, Module 04). It's a
GAN where the generator is a policy. Matches the expert's *state-action
distribution* directly → far more robust than BC to compounding error. **AIRL**
recovers a transferable reward too.

```
expert demos ─┐
              ├─► discriminator D (expert vs policy?) ──reward──► PPO ──► policy
policy rollouts┘                ▲───────────────────── policy fools D ─────┘
```

## 6. Modern imitation: diffusion & sequence models (what VLAs use)

The action *representation* has become the frontier:
- **Diffusion Policy** — model the distribution of (chunked) expert actions with a
  conditional **diffusion model**. Handles multimodal behavior ("go left *or*
  right") that a single-mode Gaussian BC can't, and is remarkably stable. A
  backbone of Octo/π0-style policies.
- **Action chunking (ACT)** — predict a *sequence* of future actions at once
  (transformer + CVAE), reducing compounding error and producing smooth motion
  (the ALOHA recipe).
- **Implicit BC** — model behavior with an energy function `argmin_a E(s,a)`
  rather than an explicit mapping; better at discontinuous/multimodal actions.

## 7. Offline RL — learn the *best* policy from a fixed dataset

The bridge between imitation and full RL: given a **fixed** dataset (no new
interaction), learn a policy that can be **better than the demonstrator**, unlike
pure BC. The challenge is **distributional shift in the value function** —
overestimating out-of-dataset actions.
- **CQL (Conservative Q-Learning)** — penalize Q-values of unseen actions.
- **IQL (Implicit Q-Learning)** — never query out-of-distribution actions.
- **Decision Transformer** — frame offline RL as **sequence modeling**: condition
  on desired return + history, predict the next action (a GPT for trajectories).
Hugely relevant: robot datasets are large and *fixed*; offline RL squeezes the
most out of them.

## 8. The unified mental model (carry this to VLA, Parts 10/16)
```
   demonstrations (teleop, Part 9)              reward / preferences (Part 14)
              │                                            │
         BC / DAgger / diffusion ──pretrain──► policy ──RL/offline fine-tune──► better policy
         (imitation, this part)                          (Modules 04, 14, 15)
```
**Imitate at scale, then refine with RL** — the recipe behind both LLMs (RLHF) and
robot foundation models (VLA).

## 🛠️ Project
1. Train an expert (PPO/SAC), clone with **BC**, measure the gap = covariate
   shift; then implement **DAgger** (re-label learner states with the expert) and
   show the gap shrink.
2. Implement **GAIL** on LunarLander: a discriminator + your `rl/agents/ppo.py` as
   the generator. Compare sample efficiency and robustness vs BC.
3. (Stretch) Train a **Diffusion Policy** on a manipulation dataset via LeRobot
   (Part 9) on your A100.

## ✅ Check your understanding
1. Why is imitation often preferable to RL for real robots?
2. Explain covariate shift and why BC errors compound quadratically.
3. How does DAgger fix it, and what does it require?
4. What does GAIL use as the reward, and why is that more robust than BC?
5. How does offline RL differ from BC, and what makes it hard?

## 📖 Go deeper
- Ross et al. (2011) **DAgger**; Ziebart et al. (2008) MaxEnt IRL; Ho & Ermon
  (2016) **GAIL**.
- Chi et al. (2023) **Diffusion Policy**; Zhao et al. (2023) **ACT/ALOHA**.
- Kumar et al. (2020) CQL; Kostrikov et al. (2021) IQL; Chen et al. (2021)
  Decision Transformer. Survey: Osa et al. (2018) *An Algorithmic Perspective on IL*.

➡️ **Next:** [Part 13 — ROS 2](../part13_ros2/), the software framework that runs
real robots.
