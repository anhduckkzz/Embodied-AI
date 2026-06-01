# Module 02 — Value-based deep RL: DQN and friends

> Code: [`rl/agents/dqn.py`](../../../rl/agents/dqn.py) ·
> Notebook: `notebook.ipynb` · Env: **LunarLander-v3**

This is the upgrade of the original repo. We replace the table `Q[s,a]` with a
neural network `Q(s, a; θ)` that maps an 8-D state to 4 action-values. This is
**DQN** (Mnih et al., 2015) — the algorithm that learned to play Atari from
pixels and kicked off the deep-RL era.

## 1. The problem with naive "deep Q-learning"

Just bolting a network onto Q-learning **diverges**. Two reasons:

1. **Correlated samples.** Consecutive transitions are highly correlated;
   SGD assumes i.i.d. data. Training on a live stream is unstable.
2. **Moving target.** The TD target `r + γ·max Q(s',a'; θ)` uses the *same*
   network we're updating. We chase a target that moves every step — like a dog
   chasing its tail.

DQN introduces two fixes, both implemented in `dqn.py`:

### Fix 1 — Experience Replay
Store transitions `(s, a, r, s', done)` in a large **replay buffer** and train
on **random minibatches**. This breaks correlations and reuses each experience
many times (sample efficiency). → [`rl/buffers.py: ReplayBuffer`](../../../rl/buffers.py)

### Fix 2 — Target Network
Keep a **separate, slowly-updated** copy of the network, `Q(·; θ⁻)`, to compute
targets. It changes slowly, so the target is stable:

```
target = r + γ · max_{a'} Q(s', a'; θ⁻)     # θ⁻ = target network
loss   = ( target − Q(s, a; θ) )²           # update θ only
```

This repo uses a **soft update** (Polyak averaging) every step instead of a hard
periodic copy — smoother:

```
θ⁻ ← τ·θ + (1−τ)·θ⁻        # τ = 1e-3
```

## 2. The DQN learning step (annotated)

```python
# target Q (no grad through the target network)
q_next   = q_target(next_states).max(dim=1, keepdim=True)[0]
q_target = rewards + gamma * q_next * (1 - dones)
# current Q for the action we actually took
q_expected = q_online(states).gather(1, actions)
loss = F.mse_loss(q_expected, q_target)   # then backprop + soft-update target
```

`(1 - dones)` zeroes the bootstrap on terminal states: a crashed lander has no
future value.

## 3. Three upgrades you should always consider

The `DQN` class supports all three via config flags — flip them on and compare.

### Double DQN — fix overestimation
Vanilla DQN's `max` systematically **overestimates** Q (it maximizes over noisy
estimates). Double DQN decouples *selection* from *evaluation*:

```
a* = argmax_{a'} Q(s', a'; θ)        # select with ONLINE net
target = r + γ · Q(s', a*; θ⁻)       # evaluate with TARGET net
```

A near-free, strict improvement — it's **on by default** in this repo.

### Dueling DQN — a better architecture
Split the network into a state-value stream `V(s)` and an advantage stream
`A(s,a)`, recombined as `Q = V + (A − mean A)`. Helps when many actions have
similar value (the agent learns "this state is good" without evaluating every
action). → [`rl/networks.py: DuelingQNetwork`](../../../rl/networks.py)

### Prioritized Experience Replay (PER) — sample what matters
Replay surprising transitions (high TD-error) more often, with importance
weights to correct the bias. → [`rl/buffers.py: PrioritizedReplayBuffer`](../../../rl/buffers.py)

## 4. Train it

```bash
# Library version (recommended):
python -m rl.train --algo dqn --env LunarLander-v3 --steps 300000 --save runs/dqn.pt
python scripts/play.py --algo dqn --model runs/dqn.pt

# Or the original, preserved for reference:
python legacy/Learn.py
```

LunarLander is "solved" at **average return ≥ 200** over 100 episodes. With the
default config you should get there in a few hundred episodes.

## 5. Ablation ideas (do these!)
- Turn **off** the target network (set `tau=1.0`). Watch it destabilize.
- Turn **off** Double DQN (`double=False`). Compare learning curves.
- Shrink the buffer to 1,000. Watch sample efficiency collapse.
- Enable `dueling=True` and `prioritized=True`. Measure the difference.

## 6. The ceiling of value-based methods
DQN needs `argmax_a Q(s,a)` — fine for 4 discrete actions, **impossible** for a
continuous action like "thrust = 0.73." Real robots have continuous actions.
To control those, we must learn the **policy directly**.

➡️ **Next:** [Module 03 — Policy gradients](../03_policy_gradient/).

## ✅ Check your understanding
1. Why does deep Q-learning diverge without a replay buffer *and* a target net?
2. What overestimation does Double DQN fix, and how?
3. Why does `(1 - dones)` appear in the target?
4. Soft vs hard target updates — what does τ control?
5. Why can't DQN drive a continuous actuator?

## 📖 Go deeper
- Mnih et al. (2015), "Human-level control through deep RL" (Nature DQN).
- van Hasselt et al. (2016), Double DQN · Wang et al. (2016), Dueling ·
  Schaul et al. (2016), Prioritized Replay.
