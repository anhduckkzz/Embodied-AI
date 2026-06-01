# Module 00 — Foundations: the RL problem

> No code yet. Twenty minutes here saves you twenty hours of confusion later.
> Everything in this repo is an answer to a question posed in this module.

## 1. What is Reinforcement Learning?

An **agent** interacts with an **environment** over discrete time steps. At each
step *t* the agent sees a **state** `s_t`, picks an **action** `a_t`, and the
environment returns a **reward** `r_t` and a next state `s_{t+1}`. The agent's
goal is to choose actions that maximize **total reward over time** — not just
the immediate reward.

```
        action a_t
   ┌──────────────────►┐
 AGENT              ENVIRONMENT
   └◄──────────────────┘
     state s_{t+1}, reward r_t
```

That's it. The hard part is "over time": a good move now (firing the lander's
main engine) costs fuel/reward now but prevents a crash later. RL is the science
of making decisions whose payoff is **delayed** and **uncertain**.

Contrast with the ML you may know:
- **Supervised learning:** given `x`, predict the correct label `y` (provided).
- **RL:** given a state, choose an action; nobody tells you the *correct*
  action — you only get a scalar reward, possibly much later.

## 2. The Markov Decision Process (MDP)

RL problems are formalized as an MDP, the tuple **(S, A, P, R, γ)**:

- **S** — set of states.
- **A** — set of actions.
- **P(s' | s, a)** — transition dynamics (probability of next state).
- **R(s, a)** — reward function.
- **γ ∈ [0, 1)** — discount factor.

**Markov property:** the future depends only on the present state, not the full
history. `s_t` summarizes everything you need to decide. (For LunarLander, the
8-number state — position, velocity, angle, leg contacts — is Markov enough.)

## 3. Return: what we actually maximize

The **return** `G_t` is the discounted sum of future rewards:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + … = Σ_{k≥0} γ^k · r_{t+k}
```

Why discount with γ?
- **Mathematical:** keeps the infinite sum finite.
- **Practical:** a reward now is worth more than the same reward later
  (uncertainty about the future).
- γ near 0 → myopic; γ near 1 (e.g. 0.99) → far-sighted. Most of this repo uses
  **γ = 0.99**.

## 4. Policy: the thing we are learning

A **policy** π is the agent's behavior — a mapping from states to actions:

- **Deterministic:** `a = π(s)`.
- **Stochastic:** `π(a | s)` = probability of action `a` in state `s`.

**The entire goal of RL is to find a policy that maximizes expected return.**
Every algorithm here is a different strategy for searching the space of policies.

## 5. Value functions: how good is a state?

Two central quantities let us reason about long-term value:

- **State-value** `V^π(s) = E_π[ G_t | s_t = s ]`
  — expected return starting from `s`, following π.
- **Action-value** `Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]`
  — expected return after taking `a` in `s`, then following π.

If you know `Q*(s, a)` (the value under the *optimal* policy), acting optimally
is trivial: **pick `argmax_a Q*(s, a)`**. Much of RL is "just" estimating Q or V.

## 6. The Bellman equations (the heart of RL)

Value functions obey a recursive self-consistency, the **Bellman equation**:

```
V^π(s)   = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s,a) + γ V^π(s') ]
Q^π(s,a) =          Σ_{s'} P(s'|s,a) [ R(s,a) + γ Σ_{a'} π(a'|s') Q^π(s',a') ]
```

and the **Bellman optimality** equation (greedy over actions):

```
Q*(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a) + γ · max_{a'} Q*(s',a') ]
```

> **This recursion is the engine of nearly every algorithm in this repo.**
> Q-learning, DQN, DDPG, TD3, SAC all turn this equation into an update rule:
> nudge your estimate of `Q(s,a)` toward `r + γ·max Q(s',a')`. The difference
> between the two sides is the **TD error**, the learning signal.

## 7. The two great families

| | **Value-based** | **Policy-based** |
|---|---|---|
| Learns | Q(s,a), act greedily | π(a\|s) directly |
| Examples | Q-learning, DQN (02) | REINFORCE, PPO (03–04) |
| Action spaces | discrete | discrete **and continuous** |
| Strength | sample-efficient | handles continuous, stochastic policies |

**Actor-Critic** (Module 04) combines both: a *policy* (actor) and a *value
function* (critic). This hybrid dominates modern RL and robotics.

## 8. Core tensions you'll meet everywhere

- **Exploration vs exploitation** — try new actions to learn, or cash in what
  you know? (ε-greedy, entropy bonuses, stochastic policies.)
- **On-policy vs off-policy** — learn from data the *current* policy generated
  (PPO), or from *any* past data in a replay buffer (DQN, SAC)?
- **Bias vs variance** — Monte-Carlo returns (unbiased, noisy) vs bootstrapped
  TD targets (biased, stable). GAE (Module 04) interpolates between them.

## ✅ Check your understanding
1. Why can't we just maximize the immediate reward `r_t`?
2. What does the Markov property let us *not* store?
3. If you had `Q*`, how would you act? Why is finding `Q*` hard?
4. Write the TD error for Q-learning in one line.
5. Give one reason to prefer a stochastic policy over a deterministic one.

## 📖 Go deeper
- Sutton & Barto, *RL: An Introduction*, Ch. 3–4 (MDPs, Bellman). [Free PDF](http://incompleteideas.net/book/the-book.html)
- David Silver's RL Course, Lectures 1–2 (YouTube).
- OpenAI Spinning Up: "Key Concepts in RL".

➡️ **Next:** [Module 01 — Tabular methods](../01_tabular/), where we turn the
Bellman equation into a working learner with nothing but a NumPy array.
