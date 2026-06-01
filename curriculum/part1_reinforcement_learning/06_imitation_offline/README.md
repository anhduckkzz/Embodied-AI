# Module 06 — Imitation & offline RL: learning from demonstrations

> Code: [`rl/agents/bc.py`](../../../rl/agents/bc.py) ·
> Notebook: `notebook.ipynb` · Env: LunarLander-v3

The single most important module for understanding today's robot foundation
models. **VLAs like RT-2, OpenVLA, Octo, and π0 are trained primarily by
imitation** — supervised learning on huge datasets of human teleoperation — not
by trial-and-error RL. RL then *refines* the imitation-pretrained policy. This
module teaches the imitation half and why RL is still indispensable.

## 1. Why imitation?

RL needs a reward function and millions of trial-and-error interactions. For
real robots:
- Reward functions are **hard to design** ("fold the towel" has no obvious scalar).
- Exploration is **dangerous and slow** (a flailing robot arm breaks things).
- But **demonstrations are cheap-ish**: a human teleoperates the robot.

So: turn control into **supervised learning** on `(observation, expert_action)`
pairs. That's **Behavioral Cloning**.

## 2. Behavioral Cloning (BC)

Collect a dataset of expert transitions, then fit a policy by maximum likelihood
— literally classification (discrete actions) or regression (continuous):

```python
# discrete: cross-entropy of predicted action logits vs expert action
loss = F.cross_entropy(policy.logits(states), expert_actions)
# continuous: negative Gaussian log-likelihood of the expert action
loss = -policy.distribution(states).log_prob(expert_actions).sum(-1).mean()
```

No environment, no reward, no Bellman equation — just `(x → y)`. In this repo,
[`collect_demonstrations`](../../../rl/agents/bc.py) rolls out a *trained* DQN/SAC
agent as the "expert" so you can clone it and compare.

```bash
# 1) train an expert (e.g. PPO), 2) clone it with BC, 3) compare returns
```

## 3. The fatal flaw: covariate shift

BC's hidden assumption: **test states look like training states.** They don't.

A cloned policy makes a small error → drifts to a state the expert never visited
→ has no idea what to do there → makes a bigger error → **compounding failure**.
Errors grow quadratically in the episode length, not linearly. This is
*distribution shift* (a.k.a. covariate shift), and it's why a BC policy that
looks great on held-out data can still crash the lander.

### Fixes
- **DAgger** (Dataset Aggregation): run the learner, have the **expert label the
  states the learner actually visits**, add to the dataset, retrain. Directly
  attacks the shift. (Needs an interactive expert.)
- **More + diverse data**, including recovery behaviors.
- **RL fine-tuning**: let the policy interact and improve with a reward signal —
  exactly what closes the loop in modern VLA training.

## 4. Beyond BC (the landscape)

- **Inverse RL / GAIL** — *infer* the reward the expert was optimizing, then do
  RL on it. GAIL uses a GAN-style discriminator ("is this state-action from the
  expert or the policy?") as the reward. More robust to shift than BC.
- **Offline RL** (CQL, IQL, Decision Transformer) — learn the *best possible*
  policy from a **fixed** dataset, allowed to outperform the demonstrator,
  without any new interaction. The bridge between imitation and full RL, and
  very relevant to learning robot policies from logged data.

## 5. The mental model to carry into Module 07

```
        BIG demonstration data                small reward signal
                │                                     │
                ▼                                     ▼
   Behavioral Cloning  ──pretrain──►  Policy  ──RL fine-tune──►  Better policy
   (imitation, this module)        (a VLA model)   (PPO/RLHF, Module 04)
```

This is the recipe behind modern robot (and LLM) foundation models: **imitate at
scale, then refine with RL.**

## ✅ Check your understanding
1. Why is imitation often more practical than RL for real robots?
2. Write BC's loss for discrete and for continuous actions.
3. Explain covariate shift and why errors compound.
4. How does DAgger attack the problem BC suffers from?
5. How do BC and RL combine in modern robot foundation models?

## 📖 Go deeper
- Pomerleau (1989), ALVINN (BC for driving) · Ross et al. (2011), **DAgger**.
- Ho & Ermon (2016), **GAIL** · Kumar et al. (2020), CQL · Chen et al. (2021),
  Decision Transformer.

➡️ **Next:** [Module 07 — VLA & robotics](../07_vla_robotics/): put it all
together into Vision-Language-Action models.
