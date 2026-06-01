# Module 11 — Distributional & Advanced Value Methods

> Builds on DQN (Module 02). A collection of upgrades that, combined, make value-
> based RL far stronger. The headline idea: **don't just predict the mean return —
> predict its whole distribution.**

## 1. Distributional RL — learn the return *distribution*

Standard Q-learning estimates `Q(s,a) = E[return]` (a single number).
**Distributional RL** instead learns the full distribution `Z(s,a)` of returns
and applies a *distributional* Bellman backup. Why it helps:
- The mean is recovered for acting, but learning the distribution provides a
  **richer training signal** (the network must explain *variance*, not just
  average), which empirically improves representations and stability.
- Captures **risk**: a policy can prefer low-variance returns (risk-sensitive RL),
  important for safety.

Key algorithms:
- **C51** — represent `Z` over 51 fixed value "atoms"; learn a categorical
  distribution via a projected Bellman update (cross-entropy loss).
- **QR-DQN** — learn the **quantiles** of `Z` instead of fixed atoms (more
  flexible support); **IQR/IQN** generalize to implicit quantiles.

## 2. The other DQN upgrades (→ Rainbow)

**Rainbow** (Hessel et al., 2018) combines six improvements; each is worth knowing:
- **Double DQN** — fix overestimation (Module 02). ✅ in our `dqn.py`.
- **Dueling networks** — split V and advantage (Module 02). ✅ in our `networks.py`.
- **Prioritized replay** — sample high-TD-error transitions (Module 02). ✅.
- **Multi-step (n-step) returns** — bootstrap after `n` rewards: lower bias than
  1-step, lower variance than Monte Carlo (the GAE idea, for value methods).
- **Distributional (C51)** — section 1 above.
- **Noisy Nets** — learnable parametric noise in the weights for *state-dependent*
  exploration (replaces ε-greedy).

Rainbow ≈ the strongest pre-2019 discrete-action agent; on Atari it crushes vanilla
DQN. The lesson: these are **composable, mostly-orthogonal** tricks.

## 3. n-step returns & eligibility traces (the bias–variance dial)
- **n-step TD:** `G = r_t + γr_{t+1} + … + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})`.
  `n=1` is TD (biased, low variance); `n=∞` is Monte Carlo (unbiased, high
  variance). Tune `n` for the sweet spot.
- **TD(λ) / eligibility traces:** elegantly average *all* n-step returns with
  weight `λ`. The value-function cousin of GAE (Part 1, Module 04). A unifying
  view of MC↔TD.

## 4. When to reach for these
- Discrete action spaces where you want maximum performance → **Rainbow**.
- Risk-sensitivity / safety → **distributional** (act on a quantile, not the mean).
- Most continuous-control robotics → you'll usually still pick **SAC/PPO** (Module
  04–05), but n-step returns and distributional critics appear there too (e.g.
  distributional critics in D4PG, TQC).

## 🛠️ Project
1. Add **n-step returns** to our `rl/agents/dqn.py` (accumulate `n` rewards before
   bootstrapping) and measure the effect on LunarLander.
2. Implement **C51** on CartPole/LunarLander; visualize the learned return
   distribution for a few states — see it sharpen as the agent improves.

## ✅ Check your understanding
1. What does distributional RL predict that vanilla DQN doesn't, and why help?
2. Name Rainbow's six components and which our repo already implements.
3. What does `n` trade off in n-step returns?
4. How is TD(λ) related to GAE?
5. When would risk-sensitivity (a quantile policy) matter on a real robot?

## 📖 Go deeper
- Bellemare et al. (2017) C51; Dabney et al. (2018) QR-DQN/IQN.
- Hessel et al. (2018) **Rainbow**. Sutton & Barto Ch. 7 & 12 (n-step, traces).

➡️ **Next:** [Module 12 — Hierarchical & Multi-agent RL](../12_hierarchical_multiagent/).
