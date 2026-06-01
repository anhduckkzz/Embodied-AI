# Part 1 — Reinforcement Learning (the full taxonomy)

> Decision-making under uncertainty: learning *what to do* from reward. This is
> the largest part of the curriculum because RL is the brain of an embodied agent
> and the refinement engine of VLA models. Below is the **map of the whole field**
> so you always know where an algorithm sits and why it exists.

## The big picture — how to classify any RL algorithm

```
                         REINFORCEMENT LEARNING
                                  │
        ┌─────────────────────────┼─────────────────────────┐
   MODEL-FREE                                          MODEL-BASED
 (learn from samples;                          (learn/known dynamics model,
  no dynamics model)                            then plan/imagine with it)
        │                                                   │
   ┌────┴────┐                                   Dyna-Q, MBPO, PETS,
 VALUE-BASED  POLICY-BASED                        World Models, Dreamer,
 (learn Q,     (learn π directly)                 MuZero, AlphaZero (MCTS)
  act greedy)      │
   │          ┌────┴────┐
 Q-learning  Policy-grad  ACTOR-CRITIC (both)
 SARSA,DQN   REINFORCE    A2C, PPO, DDPG, TD3, SAC
```

Three more axes you'll use to place any method:
- **On-policy** (learn from the *current* policy's data: REINFORCE, A2C, PPO) vs
  **off-policy** (reuse *any* past data via a replay buffer: Q-learning, DQN,
  DDPG, TD3, SAC). Off-policy = more sample-efficient; on-policy = more stable.
- **Value-based** (DQN) vs **policy-based** (REINFORCE) vs **actor-critic** (PPO/SAC).
- **Model-free** (most of the above) vs **model-based** (plan with a model — the
  most sample-efficient, and how MuZero masters games / Dreamer learns from pixels).

## Module map

| # | Module | Where it sits | Code |
|---|--------|---------------|------|
| 00 | [Foundations](00_foundations/) | MDPs, returns, Bellman | — |
| 01 | [Tabular](01_tabular/) | model-free, value, on/off-policy | `tabular.py` |
| 02 | [Value-based deep RL](02_value_based/) | model-free, value | `dqn.py` |
| 03 | [Policy gradients](03_policy_gradient/) | model-free, policy | `reinforce.py` |
| 04 | [Actor-Critic & PPO](04_actor_critic/) | model-free, hybrid | `a2c.py`, `ppo.py` |
| 05 | [Continuous control](05_continuous_control/) | off-policy, continuous | `ddpg/td3/sac.py` |
| 06 | [Imitation & offline (intro)](06_imitation_offline/) | learn from data | `bc.py` |
| 07 | [VLA & robotics (bridge)](07_vla_robotics/) | the synthesis preview | `vla_minidemo.py` |
| 08 | [Dynamic programming](08_dynamic_programming/) | **planning with a known model** | `dynamic_programming.py` |
| 09 | [Exploration & bandits](09_exploration_bandits/) | the explore/exploit core | `bandits.py` |
| 10 | [Model-based RL](10_model_based_rl/) | **Dyna, world models, Dreamer, MuZero** | `dyna.py` |
| 11 | [Distributional & advanced value](11_advanced_value_distributional/) | C51, QR-DQN, Rainbow | — |
| 12 | [Hierarchical & multi-agent](12_hierarchical_multiagent/) | options, HRL, MARL | — |
| 13 | [POMDPs & memory](13_pomdp_memory/) | partial observability | — |
| 14 | [RLHF & preference RL](14_rlhf_preference/) | the LLM/VLA fine-tuning link | — |
| 15 | [Safe RL & sim-to-real](15_safe_rl_sim2real/) | constraints, robustness, transfer | — |

## Suggested order
Do **00 → 07** first (the core path you may already know). Then fill in the
classical/advanced theory **08 → 15** in roughly that order — module 08 (DP) and
09 (exploration) actually underpin 00–05, so they're great to revisit early.

## The deadly triad (know this trap)
Combining **function approximation + bootstrapping + off-policy** learning can
diverge (the "deadly triad," Sutton & Barto Ch. 11). Target networks (DQN),
trust regions (PPO), and careful tuning are all, in part, responses to it.

➡️ Start at [00 — Foundations](00_foundations/), or jump to any module above.
