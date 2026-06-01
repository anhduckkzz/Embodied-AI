# Module 13 — POMDPs & Memory: acting under partial observability

> Real robots **never** see the full state. A camera doesn't show what's behind
> the robot; a single frame doesn't show velocity; sensors are noisy and occluded.
> This is a **Partially Observable MDP (POMDP)**, and handling it well is
> essential for embodiment.

## 1. MDP vs POMDP

- **MDP:** the agent observes the true state `s` (Markov). Everything in Modules
  00–05 assumed this.
- **POMDP:** the agent gets an **observation** `o` that's an incomplete/noisy
  function of the hidden state `s`. A single observation is **not** Markov — the
  optimal action depends on **history**.

Formally a POMDP adds an observation model `P(o|s)` to the MDP. The agent must act
on a **belief** — a distribution over the true state given all past observations.

## 2. How to act without the true state

- **Belief state** — maintain `b(s) = P(s | history)` and treat it as the state
  (a "belief MDP"). This is exactly the filtering you do in Part 7 (Bayes/Kalman/
  particle filters): the *filter's posterior is the belief*. Classical POMDP
  planning (point-based value iteration) operates on beliefs.
- **History / memory in the policy** — instead of explicit beliefs, give the
  policy **memory** so it can summarize history itself:
  - **Frame stacking** — feed the last `k` observations (how DQN sees velocity in
    Atari/LunarLander). Cheap, works for short memory.
  - **Recurrent policies (RNN/LSTM/GRU)** — the policy/value net carries a hidden
    state across time (**R2D2**, recurrent PPO). The standard deep approach.
  - **Transformers over history** — attend over a window of past observations
    (Decision Transformer, and the backbone of VLAs that condition on observation
    history). Increasingly dominant.

## 3. Why this is central to robotics & VLA
- **Sensing limits:** occlusion, limited FoV, noise → partial observability is the
  *default*, not an edge case.
- **Velocity/dynamics:** position-only sensors hide velocity → need memory or
  stacking (your LunarLander state *includes* velocities precisely to stay
  Markov; pixel-based versions don't).
- **VLA models** are POMDP solvers: they condition on a **history of images** (and
  the instruction) and often carry recurrent/attention memory — exactly to handle
  partial observability over a manipulation episode.

## 4. Practical recipe
1. First try to **make the state more Markov**: add velocities, stack frames,
   fuse sensors (Part 7) so a good belief is available.
2. If history still matters, use a **recurrent or transformer policy** (train PPO/
   SAC with an LSTM/attention encoder over observations).
3. For long, sparse tasks, combine with hierarchy (Module 12) and good
   exploration (Module 09).

## 🛠️ Project
1. Take CartPole and **hide the velocities** (observation = positions only). Show
   that a feedforward policy struggles, then fix it with (a) frame-stacking and
   (b) an LSTM policy. Compare.
2. Read the Decision Transformer paper and relate "RL as sequence modeling" to how
   a VLA conditions on observation/instruction history (Part 16).

## ✅ Check your understanding
1. Why is a single observation in a POMDP generally not Markov?
2. What is a belief state, and how does Part 7's filter relate to it?
3. Three ways to give a policy memory — trade-offs of each.
4. Why does LunarLander's *vector* state avoid the POMDP problem that a *pixel*
   version would have?
5. In what sense is a VLA a POMDP policy?

## 📖 Go deeper
- Kaelbling, Littman & Cassandra (1998), POMDPs — the classic.
- Hausknecht & Stone (2015) DRQN; Kapturowski et al. (2019) R2D2 (recurrent deep RL).
- Chen et al. (2021) Decision Transformer (RL as sequence modeling).

➡️ **Next:** [Module 14 — RLHF & preference-based RL](../14_rlhf_preference/).
