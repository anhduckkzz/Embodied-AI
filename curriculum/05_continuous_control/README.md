# Module 05 — Continuous control: DDPG → TD3 → SAC

> Code: [`ddpg.py`](../../rl/agents/ddpg.py), [`td3.py`](../../rl/agents/td3.py),
> [`sac.py`](../../rl/agents/sac.py) · Notebook: `notebook.ipynb` ·
> Envs: LunarLanderContinuous-v3, Pendulum-v1

**This is the robotics core of the curriculum.** Real robots take continuous
actions — joint torques, end-effector velocities, gripper forces. The methods
here output those directly and are *sample-efficient* (off-policy), which
matters when each sample is an expensive real-world interaction.

LunarLander has a **continuous variant**: instead of 4 buttons, two real-valued
throttles (main engine, lateral engines). Same physics, robot-style action space.

```bash
python -m rl.train --algo sac --env LunarLander-v3 --continuous --steps 300000
```

## 1. DDPG — Deterministic policy gradient

DQN can't `argmax` over a continuous action. DDPG's trick: learn a
**deterministic actor** `μ(s)` that *outputs* the maximizing action, and a
**critic** `Q(s,a)` that scores it. Train the actor to climb the critic's
gradient:

```
actor loss   = − E[ Q(s, μ(s)) ]            # push μ toward high-Q actions
critic loss  = ( Q(s,a) − [r + γ Q'(s', μ'(s'))] )²   # off-policy TD, like DQN
```

It's "DQN for continuous actions": replay buffer, target networks, off-policy.
Exploration comes from adding noise to `μ(s)` while acting. DDPG **works but is
fragile** — sensitive to hyper-parameters and prone to Q-overestimation.

## 2. TD3 — three fixes that make it reliable

TD3 keeps DDPG's structure and patches its failure modes (see
[`td3.py`](../../rl/agents/td3.py)):

1. **Twin critics + clipped double-Q.** Learn two critics, use the **minimum**
   in the target. Counters overestimation (the continuous analogue of Double DQN).
2. **Target policy smoothing.** Add small clipped noise to the target action so
   the critic can't exploit sharp, spurious peaks.
3. **Delayed policy updates.** Update the actor (and targets) every 2 critic
   updates — the actor chases a more stable value estimate.

These three lines of defense turn DDPG from "sometimes works" into a solid
baseline.

## 3. SAC — Soft Actor-Critic (the one to reach for)

SAC is the modern default for continuous control and a direct ancestor of
methods used on real robots. Its idea is **maximum-entropy RL**: maximize reward
**and** policy randomness:

```
J = E[ Σ_t  r_t  +  α · H(π(·|s_t)) ]
```

Acting as randomly as possible *while still solving the task* yields policies
that are robust and explore well. SAC combines this with:
- a **stochastic, squashed-Gaussian actor** (reparameterized for low-variance
  gradients; tanh keeps actions bounded — [`SquashedGaussianActor`](../../rl/networks.py)),
- **twin critics** (from TD3),
- **automatic temperature tuning** — α is learned to hit a target entropy, so
  there's one fewer knob to tune.

SAC is typically the **most sample-efficient** and **most stable** of the three.
For your robotics ambitions, SAC (and PPO from Module 04) are the two you'll use
most.

## 4. PPO vs SAC — which, when?

| | **PPO** (on-policy) | **SAC** (off-policy) |
|---|---|---|
| Sample efficiency | lower | **higher** |
| Stability / tuning | very robust | robust, a bit more delicate |
| Parallelism | scales to thousands of sims | harder to parallelize |
| Used for | sim-heavy RL, RLHF, VLA fine-tune | real-robot, sample-limited |

In robotics you'll meet **both**: PPO with massively parallel simulators (Isaac
Gym) for locomotion, SAC/offline methods when samples are precious.

## 5. Experiments to run
- SAC vs TD3 vs DDPG on `LunarLanderContinuous` — plot sample efficiency.
- Turn off SAC's entropy auto-tuning (`autotune_alpha=False`, try α=0.05 vs 0.5).
- Disable TD3's twin critics (use one) — watch Q-values explode.

## ✅ Check your understanding
1. Why can't DQN be applied to a continuous action space?
2. How does DDPG get a "max over actions" without an argmax?
3. What are TD3's three fixes, and which DQN idea does each echo?
4. What does the entropy term in SAC's objective encourage, and why is that good?
5. When would you pick PPO over SAC, and vice versa?

## 📖 Go deeper
- Lillicrap et al. (2016), DDPG · Fujimoto et al. (2018), TD3 ·
  Haarnoja et al. (2018), SAC.
- OpenAI Spinning Up implementations of DDPG/TD3/SAC.

➡️ **Next:** [Module 06 — Imitation & offline RL](../06_imitation_offline/):
when reward is hard but demonstrations are easy — the dominant paradigm in
modern robotics.
