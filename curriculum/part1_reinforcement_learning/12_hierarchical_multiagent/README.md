# Module 12 — Hierarchical & Multi-Agent RL

Two extensions of RL that matter for real robots and the real world: **acting at
multiple time scales** (hierarchy) and **many agents interacting** (multi-agent).

---

## Part A — Hierarchical RL (HRL): temporal abstraction

Flat RL chooses a low-level action every step. For long-horizon tasks ("make
coffee") that's hopeless — the reward is thousands of steps away (credit
assignment + exploration both explode). **HRL** introduces structure: high-level
policies pick **sub-goals or skills**; low-level policies execute them.

- **Options framework** — an *option* is a temporally-extended action: a policy +
  an initiation set + a termination condition ("walk to the door"). The high
  level chooses options; the low level runs them. Semi-MDP theory makes this
  rigorous.
- **Feudal / goal-conditioned HRL** — a *manager* sets goals (in state or latent
  space); a *worker* is rewarded for reaching them (FeUdal Networks, HIRO).
- **Skill discovery** — learn a repertoire of reusable skills *without* task
  reward (e.g. **DIAYN**: "diversity is all you need" — learn skills that are
  distinguishable), then compose them.

**Why it matters for embodiment:** a humanoid's "walk / grasp / open" skills are
options; a VLA's language sub-goals ("pick up the cup, then place it") are a
natural hierarchy. Hierarchy = better exploration, transfer, and interpretability.

## Part B — Multi-Agent RL (MARL): many learners at once

When multiple agents act in a shared environment, each agent's environment is
*non-stationary* (others are learning too), breaking single-agent guarantees.
Settings:
- **Cooperative** — shared goal (a fleet of warehouse robots, swarm drones).
- **Competitive** — zero-sum (game-playing, adversaries).
- **Mixed** — autonomous driving (self-interested but must coordinate/avoid).

Core ideas:
- **Centralized training, decentralized execution (CTDE)** — train with access to
  all agents' info (a central critic), but each agent acts on its *own*
  observation at test time. The dominant paradigm: **MADDPG, QMIX, MAPPO**.
- **Self-play** — agents improve by playing copies of themselves (AlphaGo,
  OpenAI Five, AlphaStar). Generates an automatic curriculum of opponents.
- **Game theory** — equilibria (Nash), credit assignment among cooperators
  (who contributed to the team reward?), communication learning.

**Why it matters for embodiment:** multi-robot coordination, swarms, and
human-robot interaction are inherently multi-agent; driving is a negotiation with
other drivers.

## Where these sit in your path
- HRL → long-horizon **manipulation** (Part 9) and **VLA** (Part 10/16) where
  tasks decompose into skills/sub-goals.
- MARL → **drones/swarms** and **autonomous driving** (Part 11), and any
  multi-robot deployment.

## 🛠️ Project
1. **HRL:** on a long-corridor gridworld with sparse reward, compare flat
   Q-learning vs a 2-level options agent (hand-coded options) — see the
   exploration difference.
2. **MARL:** use **PettingZoo** (the multi-agent Gymnasium) to run a cooperative
   task; train independent PPO vs MAPPO and compare coordination.

## ✅ Check your understanding
1. Why does flat RL struggle with long-horizon, sparse-reward tasks?
2. What are the three pieces that define an "option"?
3. What does skill discovery (e.g. DIAYN) optimize, with no task reward?
4. Why is a multi-agent environment non-stationary from one agent's view?
5. Explain centralized-training / decentralized-execution and why it helps.

## 📖 Go deeper
- Sutton, Precup & Singh (1999) Options; Vezhnevets et al. (2017) FeUdal; Eysenbach
  et al. (2018) DIAYN.
- Lowe et al. (2017) MADDPG; Rashid et al. (2018) QMIX; Yu et al. (2021) MAPPO.
- PettingZoo & the *Multi-Agent RL* book (Albrecht, Christianos, Schäfer, free).

➡️ **Next:** [Module 13 — POMDPs & memory](../13_pomdp_memory/).
