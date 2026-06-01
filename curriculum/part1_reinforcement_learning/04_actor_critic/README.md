# Module 04 — Actor-Critic & PPO: the modern default

> Code: [`rl/agents/a2c.py`](../../../rl/agents/a2c.py),
> [`rl/agents/ppo.py`](../../../rl/agents/ppo.py) ·
> Notebook: `notebook.ipynb` · Env: LunarLander-v3

This is the most important module for your goal. **PPO** is the workhorse of
applied RL — it trains robot controllers, game agents, and is the algorithm
behind **RLHF for LLMs** and much **VLA post-training**. We get there in two
steps: A2C, then PPO.

## 1. Actor-Critic: learn the baseline online

REINFORCE waits until the episode ends to compute returns. **Actor-Critic** runs
two networks together:

- **Actor** `π(a|s; θ)` — the policy (what to do).
- **Critic** `V(s; φ)` — estimates state value (how good things are).

The critic supplies the baseline/advantage *as you go*, so you can update every
few steps instead of every episode. The actor's gradient uses the advantage; the
critic is trained by regression toward the TD/return target.

## 2. GAE: the bias–variance dial for advantages

How to estimate the advantage `A_t`? Two extremes:
- **Monte-Carlo** (`G_t − V`): unbiased, high variance.
- **One-step TD** (`r + γV(s') − V(s)`): low variance, biased.

**Generalized Advantage Estimation (GAE)** smoothly interpolates with a
parameter λ ∈ [0,1]:

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)             # TD residual
A_t = Σ_{l≥0} (γλ)^l · δ_{t+l}                # exponentially-weighted sum
```

λ=0 → one-step TD; λ=1 → Monte-Carlo. λ≈0.95 is the sweet spot used everywhere.
Implemented in [`rl/buffers.py: RolloutBuffer`](../../../rl/buffers.py).

## 3. A2C — Advantage Actor-Critic

Collect a short rollout (e.g. 5 steps), compute GAE advantages, take **one**
gradient step on:

```
L = −E[ A_t · log π(a_t|s_t) ]  +  c_v·(V − V_target)²  −  c_e·H(π)
     └ policy (actor) ┘            └ value (critic) ┘     └ entropy bonus ┘
```

The **entropy bonus** `H(π)` keeps the policy from collapsing too early
(exploration). A2C works, but a single big step on freshly-collected data can
overshoot and destroy the policy. That's the problem PPO solves.

## 4. PPO — Proximal Policy Optimization

PPO's insight: after collecting a rollout, we'd like to **reuse it for several
gradient steps** (sample efficiency) — but the policy drifts away from the one
that generated the data, and large updates are catastrophic. PPO constrains each
update with a **clipped surrogate objective**.

Let `r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)` be the probability ratio. PPO
maximizes:

```
L_CLIP = E[ min( r_t·A_t ,  clip(r_t, 1−ε, 1+ε)·A_t ) ]
```

- If an action was good (`A_t>0`), increase its probability — but **clip** the
  incentive once `r_t > 1+ε`, so one update can't move the policy too far.
- If bad (`A_t<0`), decrease it — clipped at `1−ε`.

The `min` makes the bound **pessimistic**: it only removes incentive to change,
never adds it. The result is a method that's nearly as simple as A2C but stable
enough to run 10 epochs of minibatch SGD per rollout. See
[`rl/agents/ppo.py`](../../../rl/agents/ppo.py) — the clip is four lines.

Extras this implementation includes (all standard "PPO tricks"):
- **Advantage normalization** per batch.
- **Value & gradient clipping.**
- **Early stopping** when approximate KL exceeds a target — a guardrail.

## 5. Train it

```bash
python -m rl.train --algo a2c --env LunarLander-v3 --steps 500000
python -m rl.train --algo ppo --env LunarLander-v3 --steps 500000 --save runs/ppo.pt
python scripts/play.py --algo ppo --model runs/ppo.pt
```

PPO should comfortably solve LunarLander (avg ≥ 200) and is far more robust to
hyper-parameters than REINFORCE.

## 6. Why PPO matters for VLA
The exact same clipped objective, with a reward model in place of the
environment reward, is **RLHF** — how instruction-following is fine-tuned into
LLMs and into Vision-Language-Action models. Understanding PPO here means you
already understand the RL half of modern robot foundation models. (Module 07.)

## ✅ Check your understanding
1. What does the critic give the actor that REINFORCE computed only at episode end?
2. What does λ in GAE trade off? What happens at λ=0 and λ=1?
3. Why does PPO clip the probability ratio? What goes wrong without it?
4. Why is the `min` in `L_CLIP` important (vs just clipping)?
5. What is the entropy bonus for?

## 📖 Go deeper
- Schulman et al. (2017), **PPO** · Schulman et al. (2016), **GAE** · Mnih et al. (2016), A3C/A2C.
- "The 37 Implementation Details of PPO" (ICLR blog) — read before you trust your own PPO.
- OpenAI Spinning Up: VPG → TRPO → PPO.

➡️ **Next:** [Module 05 — Continuous control](../05_continuous_control/): off-policy
methods built for the continuous-action, sample-expensive world of robots.
