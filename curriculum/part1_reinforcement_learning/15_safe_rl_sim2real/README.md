# Module 15 — Safe RL & Sim-to-Real Transfer

> The last mile from "works in simulation" to "works on a real robot without
> breaking it (or anyone)." Two intertwined concerns: **safety** (don't take
> catastrophic actions) and **transfer** (a sim policy must survive reality).

---

## Part A — Safe & Constrained RL

Plain RL maximizes reward and will happily learn dangerous behavior if it pays.
Real robots need **constraints**: joint/torque limits, no collisions, stay
upright, don't exceed speed.

- **Constrained MDPs (CMDP)** — maximize reward **subject to** expected-cost
  constraints `E[Σ cost] ≤ d`. Solved with **Lagrangian** methods (adapt a penalty
  multiplier) — e.g. **CPO (Constrained Policy Optimization)**, Lagrangian-PPO/SAC.
- **Safety layers / shielding** — a hard filter that overrides unsafe actions
  (control-barrier functions, reachability) — provable safety on top of a learned
  policy.
- **Risk-sensitive RL** — optimize a risk measure (CVaR) instead of the mean
  (connects to distributional RL, Module 11): avoid rare catastrophes, not just
  low average return.
- **Safe exploration** — explore without entering unsafe states (crucial when
  exploring on hardware): conservative initial policies, recovery policies,
  human oversight.

## Part B — Sim-to-Real Transfer

You train in sim (Part 4) because it's fast/safe, but **sim ≠ reality** (the
sim-to-real gap, Part 4.6). Bridging techniques:

- **Domain randomization (DR)** — randomize sim parameters (masses, friction,
  latency, sensor noise, **textures/lighting** for vision) during training so the
  real value is "just another sample." The policy learns to be **robust** rather
  than overfit. (OpenAI's dexterous-hand cube, ANYmal locomotion.)
- **System identification** — measure the real robot and tune the sim to match
  (closing the gap from the sim side).
- **Domain adaptation** — adapt perception features from sim to real (e.g.
  adversarial/feature alignment, real-image fine-tuning).
- **Real-world fine-tuning** — a little on-robot RL or imitation (Parts 9, 12) to
  polish a sim-pretrained policy.
- **Robust & adaptive control** — policies that *infer* the dynamics online
  (e.g. **RMA: Rapid Motor Adaptation** for legged robots) and adjust within
  milliseconds on hardware.

> The modern legged-robot recipe (Part 17): **massively parallel PPO in Isaac Lab
> + heavy domain randomization + (optional) online adaptation** → deploy. That is
> this module in one sentence.

## Part C — Generalization & curriculum

- **Curriculum learning** — train on easy tasks first, gradually harder
  (automatic curricula via self-play, Module 12, or procedural difficulty).
- **Procedural environment generation / domain randomization of *tasks*** — train
  on a distribution of environments to generalize to unseen ones.
- **Evaluation** — test on held-out conditions; beware "it worked on the training
  seed." Report robustness, not just peak reward.

## Where this lands in your path
Everything that makes a policy from Parts 1/12 actually **deployable** on the
robots of Part 11 (drones, cars, humanoids) goes through this module. Safety +
transfer is what separates a demo from a product.

## 🛠️ Project
1. **Safe RL:** add a cost (e.g. penalize tilt > threshold) on a MuJoCo task and
   implement Lagrangian-PPO (adapt a penalty multiplier to satisfy the constraint).
   Compare reward/violations vs unconstrained PPO.
2. **Sim-to-real (in sim):** train SAC on `Pendulum`/`Ant` with **domain
   randomization** (randomize mass/friction each episode) and show it's robust to
   a *held-out* dynamics setting that a non-randomized policy fails on.

## ✅ Check your understanding
1. Why will unconstrained RL learn unsafe behavior, and what is a CMDP?
2. How does a Lagrangian method enforce a constraint?
3. What does domain randomization do, and why does it improve real-world transfer?
4. System identification vs domain randomization — opposite ends of bridging the gap?
5. What does online adaptation (e.g. RMA) add beyond a fixed robust policy?

## 📖 Go deeper
- Achiam et al. (2017) **CPO**; García & Fernández (2015) safe-RL survey; CVaR/risk RL.
- Tobin et al. (2017) Domain Randomization; OpenAI (2019) dexterous hand;
  Kumar et al. (2021) **RMA**; Lee et al. (2020) ANYmal locomotion.

🎉 You now have the **full RL taxonomy** — model-free and model-based, value and
policy, exploration, hierarchy, multi-agent, partial observability, preference
learning, and safe transfer. ➡️ Continue to
[Part 12 — Imitation Learning](../../part12_imitation_learning/) for the other
great way to learn behavior.
