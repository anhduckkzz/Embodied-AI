# Module 14 — RLHF & Preference-Based RL

> The bridge from "games" RL to **foundation models** — both LLMs and VLAs. When
> you *can't write a reward function* (how do you score "a helpful answer" or "a
> graceful grasp"?), learn the reward from **human preferences**, then optimize it
> with the PPO you already know (Module 04).

## 1. The problem: reward design is hard

For LunarLander the reward is given. For "fold the laundry," "drive courteously,"
or "write a good summary," there's **no obvious scalar**. Hand-crafted rewards get
**reward-hacked** (the agent exploits loopholes). Solution: **let humans judge,
and learn a reward model from their judgments.**

## 2. RLHF — Reinforcement Learning from Human Feedback

The recipe that aligned ChatGPT/Claude, in three stages:

1. **Supervised fine-tuning (SFT)** — start from a pretrained model; fine-tune on
   demonstrations of good behavior. *(This is imitation learning — Part 12.)*
2. **Reward model (RM)** — collect **comparisons**: humans rank outputs A vs B.
   Train a model `r_φ(x, y)` to predict the preferred one (Bradley-Terry loss on
   pairwise preferences). The RM is a *learned reward function*.
3. **RL optimization** — fine-tune the policy with **PPO** (Module 04!) to
   maximize the reward model, with a **KL penalty** to stay close to the SFT model
   (so it doesn't drift into degenerate, reward-hacking text).

```
demos ─SFT(BC)─► base policy ─┐
human comparisons ─► reward model r_φ ─► PPO (max r_φ − β·KL) ─► aligned policy
```

The objective is literally PPO's clipped surrogate with reward `r_φ(x,y) −
β·KL(π‖π_SFT)`. **You already implemented the hard part** in `rl/agents/ppo.py`.

## 3. The newer, simpler alternatives

- **DPO (Direct Preference Optimization)** — skip the explicit reward model *and*
  the RL loop: a clever loss turns preferences **directly** into a policy update
  (a classification-style objective). Simpler, stable, now very popular.
- **RLAIF** — replace human labels with **AI feedback** (a strong model judges),
  scaling preference data cheaply. (Anthropic's *Constitutional AI* is a variant.)
- **GRPO / RLOO / others** — efficient policy-gradient variants for preference/
  verifiable-reward fine-tuning (used in recent reasoning models).

## 4. Why this is the heart of VLA fine-tuning (Part 16)

Robot foundation models follow the *same* pattern:
- **SFT = behavioral cloning** on teleop demos (Part 9/12) — the imitation bulk.
- **Reward/preference fine-tuning** — improve beyond demos using success signals,
  human preferences over robot trajectories, or learned reward models; optimize
  with PPO-style RL.
- The **KL-to-reference** trick keeps the fine-tuned robot policy from
  catastrophically forgetting its imitation-pretrained competence.

So the "imitate at scale, then refine with RL" slogan (Parts 1, 9, 16) **is**
RLHF applied to robots.

## 5. Pitfalls (real and important)
- **Reward hacking / over-optimization** — push the RM too hard and the policy
  exploits its errors; the KL penalty and RM ensembles mitigate it.
- **Preference noise & bias** — humans disagree and are inconsistent; data quality
  dominates.
- **Distribution shift** — the RM is only valid near the data it was trained on.

## 🛠️ Project
1. **Toy RLHF:** on a simple env, hand-define a hidden "true" reward, generate
   preference pairs from it, train a small reward model, then PPO-optimize against
   the *learned* RM (with a KL penalty to a reference policy). Show it recovers
   good behavior — and that removing the KL term causes reward hacking.
2. Read the InstructGPT and DPO papers; map every stage to Modules 04 (PPO) and 12
   (imitation).

## ✅ Check your understanding
1. Why learn a reward model instead of writing the reward?
2. Name RLHF's three stages and which earlier module each reuses.
3. What does the KL penalty prevent, and against what reference?
4. How does DPO simplify the pipeline?
5. How does RLHF map onto training a VLA from teleop demos?

## 📖 Go deeper
- Christiano et al. (2017) *Deep RL from Human Preferences*; Ouyang et al. (2022)
  InstructGPT. Rafailov et al. (2023) **DPO**. Bai et al. (2022) Constitutional AI.

➡️ **Next:** [Module 15 — Safe RL & sim-to-real](../15_safe_rl_sim2real/).
