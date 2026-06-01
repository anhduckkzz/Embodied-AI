# Module 03 — Policy gradients: REINFORCE

> Code: [`rl/agents/reinforce.py`](../../../rl/agents/reinforce.py) ·
> Notebook: `notebook.ipynb` · Envs: CartPole-v1, LunarLander-v3

So far we learned values and acted greedily. Now we **parameterize the policy
directly** as `π(a | s; θ)` and optimize θ by gradient *ascent* on expected
return. This unlocks continuous actions and naturally stochastic policies — and
it is the conceptual root of PPO and of the RLHF/VLA fine-tuning you're aiming
at.

## 1. The objective and its gradient

Maximize expected return `J(θ) = E_{τ~π_θ}[ R(τ) ]`. The miracle is the
**Policy Gradient Theorem**, which lets us differentiate an expectation over
trajectories we can only *sample*:

```
∇_θ J(θ) = E[ Σ_t ∇_θ log π(a_t | s_t; θ) · G_t ]
```

In words: **push up the log-probability of actions, weighted by the return that
followed.** Good outcome → make those actions more likely; bad outcome → less
likely. No model of the environment needed — just the ability to sample and to
differentiate `log π`.

## 2. REINFORCE (Monte-Carlo policy gradient)

1. Run a full episode with the current policy, recording `log π(a_t|s_t)`.
2. Compute the **return-to-go** `G_t = Σ_{k≥t} γ^{k-t} r_k` for each step.
3. Take one gradient step: `loss = − Σ_t log π(a_t|s_t) · G_t`.

```python
G = 0
for r in reversed(rewards):       # returns-to-go, computed backwards
    G = r + gamma * G
    returns.insert(0, G)
loss = -(log_probs * returns).mean()    # gradient ascent via negation
```

## 3. The variance problem — and the baseline

REINFORCE works but is **noisy**: returns vary wildly between episodes, so the
gradient is high-variance and learning is slow. The fix is a **baseline** `b(s)`
subtracted from the return:

```
∇_θ J = E[ Σ_t ∇_θ log π(a_t|s_t) · (G_t − b(s_t)) ]
```

Subtracting a baseline that doesn't depend on the action **does not change the
expected gradient** (it stays unbiased) but **dramatically reduces variance**.
The best practical baseline is a learned value function `V(s) ≈ E[G_t]`. Then
`G_t − V(s_t)` is an estimate of the **advantage** `A(s,a)` — "how much better
was this action than average?" This repo's `use_baseline=True` does exactly this.

> **Advantage is the bridge to everything next.** A2C and PPO are, at heart,
> "policy gradient with a really good advantage estimate."

## 4. Discrete *and* continuous policies

The same code handles both — only the policy distribution changes:
- **Discrete:** a `Categorical` over action logits (`CategoricalActor`).
- **Continuous:** a `Gaussian` whose mean is a network output, std a learned
  parameter (`GaussianActor`). Sample a real-valued action; `log π` is the
  Gaussian log-density.

This is the first time we can, in principle, output a continuous thrust value —
the door to robotics control opens here.

## 5. Train it

```bash
python -m rl.train --algo reinforce --env CartPole-v1 --episodes 1000
python -m rl.train --algo reinforce --env LunarLander-v3 --episodes 3000
```

CartPole (solved at 195) is the gentle start; LunarLander shows how high
variance makes pure policy gradient slower than DQN here — motivating actor-critic.

## ✅ Check your understanding
1. State the policy gradient theorem in words.
2. Why doesn't subtracting a state-dependent baseline bias the gradient?
3. What is the advantage `A(s,a)`, and why is it a great weighting?
4. Why is REINFORCE higher-variance than DQN is for the same task?
5. What changes in the code to go from discrete to continuous actions?

## 📖 Go deeper
- Williams (1992), "REINFORCE" · Sutton et al. (2000), Policy Gradient Theorem.
- Karpathy, "Deep Reinforcement Learning: Pong from Pixels".

➡️ **Next:** [Module 04 — Actor-Critic & PPO](../04_actor_critic/): learn the
baseline *online* and make policy gradients stable enough to be the modern default.
