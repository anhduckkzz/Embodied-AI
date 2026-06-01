# RL: Zero → Master → VLA 🚀🌙🤖

A hands-on Reinforcement Learning curriculum that grows from a single
LunarLander DQN into a complete journey: **tabular methods → deep RL → modern
policy optimization → continuous control → imitation learning → Vision-Language-Action (VLA) models for robotics.**

Every concept comes in two forms, on purpose:

- 📓 **Notebooks** (`curriculum/**/notebook.ipynb`) — read the theory, run the
  cells, train on a GPU (Colab Pro). Best for *learning* and *experimenting*.
- 🧱 **Library** (`rl/`) — the same algorithms as clean, importable,
  professionally structured Python. Best for *learning real codebase
  architecture* and reusing code across experiments.

> The original repo was a single DQN that lands the lunar module. It now lives
> in [`legacy/`](legacy/) as a historical reference — everything below is the
> upgrade.

![LunarLander](doc/captured.gif)

---

## 🗺️ The roadmap

The curriculum is 8 modules. Each builds directly on the previous one; the
"why" of each algorithm is the "weakness" of the one before it.

| # | Module | You learn | Key algorithms | Env |
|---|--------|-----------|----------------|-----|
| 00 | [Foundations](curriculum/00_foundations/) | MDPs, returns, value functions, Bellman equations, the RL problem | *(math, no code)* | — |
| 01 | [Tabular methods](curriculum/01_tabular/) | Learning from interaction with a table | Q-learning, SARSA | FrozenLake, CliffWalking |
| 02 | [Value-based deep RL](curriculum/02_value_based/) | Scaling Q-learning with neural nets | DQN, Double, Dueling, PER | **LunarLander** |
| 03 | [Policy gradients](curriculum/03_policy_gradient/) | Optimizing the policy directly | REINFORCE, baselines | CartPole, LunarLander |
| 04 | [Actor-Critic & PPO](curriculum/04_actor_critic/) | Combining value + policy; the modern default | A2C, **PPO**, GAE | LunarLander |
| 05 | [Continuous control](curriculum/05_continuous_control/) | Controlling torques/thrust — the robotics regime | DDPG, TD3, **SAC** | LunarLander-Continuous, Pendulum |
| 06 | [Imitation & offline RL](curriculum/06_imitation_offline/) | Learning from demonstrations | Behavioral Cloning, DAgger, GAIL | LunarLander |
| 07 | [VLA & robotics](curriculum/07_vla_robotics/) | How RL + IL power robot foundation models | RT-2, OpenVLA, π0, RLHF | mini grid-world demo |

```
Module:   00 ──► 01 ──► 02 ──► 03 ──► 04 ──► 05 ──► 06 ──► 07
          math   tab    DQN    PG    PPO    SAC    BC    VLA
                  │      │      │      │      │      │      │
                  └─ value-based ┘    └ policy-based ┘     └ foundation models
```

---

## ⚡ Quick start

### On your machine
```bash
git clone <this-repo> && cd LunarLander
pip install -r requirements.txt          # or: pip install -e ".[box2d,dev]"

# Train PPO on LunarLander and watch it solve:
python -m rl.train --algo ppo --env LunarLander-v3 --save runs/ppo.pt

# Watch the trained agent:
python scripts/play.py --algo ppo --model runs/ppo.pt

# Record a GIF:
python scripts/record.py --algo ppo --model runs/ppo.pt --out doc/ppo.gif
```

### On Google Colab (recommended for training — free GPU)
Open any `curriculum/**/notebook.ipynb` in Colab, run the setup cell, and
train with hardware acceleration. Each notebook is self-contained.

### Run the tests
```bash
pytest -q          # shape + learning sanity checks for the whole library
```

---

## 🧱 Library architecture (`rl/`)

A deliberately small, readable package — study it as an example of how to
structure an ML codebase:

```
rl/
├── networks.py     # MLP, Q-net, dueling, Gaussian/categorical/squashed actors, critics
├── buffers.py      # ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer (+ GAE)
├── envs.py         # Gymnasium env factory + wrappers
├── utils.py        # seeding, schedules, normalization, logging, plotting
├── train.py        # unified CLI: off-policy / on-policy / episodic loops
└── agents/
    ├── tabular.py   reinforce.py  ppo.py   td3.py
    ├── dqn.py       a2c.py        ddpg.py  sac.py   bc.py
```

Design principles:
- **Every agent is `act()` + a learning rule.** The networks and buffers do the
  heavy lifting, so each algorithm file stays short and comparable.
- **Configs are dataclasses.** Hyper-parameters are explicit and documented.
- **The maths in the lessons matches the code.** Comments cite the update rules.

---

## 🤖 Why this path leads to VLA

Your destination — **Vision-Language-Action models** that let robots follow
instructions — is built from exactly these pieces:

1. **Continuous control (05)** is how policies output real robot actions
   (joint torques, gripper commands), not discrete button presses.
2. **Imitation learning (06)** is how today's robot foundation models
   (RT-2, OpenVLA, Octo, π0) are *primarily* trained — supervised learning on
   massive teleoperation datasets.
3. **Policy optimization (04)** — specifically PPO — is how those models are
   *refined* after pretraining (the same RLHF machinery used for LLMs).
4. **The RL formalism (00)** — states, actions, rewards, policies — is the
   shared language that makes "a robot that acts in the world" a learnable
   problem at all.

Module 07 connects all of this and includes a small language-conditioned
policy demo so the abstraction becomes concrete.

---

## 📚 Suggested study plan

- **Week 1:** Modules 00–01. Get the formalism and watch a value function form.
- **Week 2–3:** Module 02. Re-derive DQN, train LunarLander, read `rl/agents/dqn.py`.
- **Week 4:** Modules 03–04. REINFORCE → A2C → PPO. Solve LunarLander with PPO.
- **Week 5:** Module 05. SAC on continuous control. This is the robotics core.
- **Week 6:** Module 06. Clone an expert; understand covariate shift.
- **Week 7+:** Module 07. Read the VLA papers, run the mini-demo, plan a project.

Each module README ends with **"Check your understanding"** questions and
**"Go deeper"** references.

---

## 🙏 Credits & license

- Environment: [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (Box2D).
- Original DQN baseline by Pajuhaan (preserved in `legacy/`).
- Curriculum & library: this repo. **License: BSD-3-Clause.**

The classics worth owning: Sutton & Barto, *Reinforcement Learning: An
Introduction* (free online); Spinning Up in Deep RL (OpenAI).
