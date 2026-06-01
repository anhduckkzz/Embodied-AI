# Embodied AI: Zero → Master

> From the Bellman equation to robots, self-driving cars, drones, and humanoid
> **Vision-Language-Action** models — a complete, hands-on curriculum that assumes
> **zero prior knowledge** and builds, with runnable code at every step, toward
> modern embodied intelligence.

This repo began as a single LunarLander DQN. It is now an **18-part** curriculum
spanning the whole field: math/ML foundations, the full **reinforcement-learning
taxonomy**, **imitation learning**, robotics **physics** and **3D math**,
**control theory**, **simulation** (MuJoCo, Isaac Sim/Lab), **ROS 2**,
**sensors & perception** (RGB, depth, LiDAR, GPS/IMU), **3D vision** (SfM, NeRF,
Gaussian Splatting), **state estimation & SLAM**, **navigation & planning**,
**manipulation & teleoperation**, **foundation models** (transformers, diffusion),
**VLA models**, **legged locomotion**, and **applications** (driving, drones,
humanoids).

---

## Learning philosophy: mechanisms → frameworks → real systems

This is a personal learning repo for building embodied AI intuition **and**
practical fluency. Each topic should be studied in three passes:

1. **Mechanisms first** — read the math and the from-scratch implementation so
   the core idea is visible instead of hidden behind an API.
2. **Frameworks second** — run the same idea with the standard library,
   framework, or service used in practice (PyTorch, Gymnasium/SB3, MuJoCo,
   ROS 2, Open3D, CARLA, LeRobot, Hugging Face, etc.).
3. **Real systems third** — connect the piece into an embodied stack: sensors →
   perception → state estimation → planning/control → learning policy → robot,
   car, drone, or VLA application.

The goal is not just to collect algorithms. The goal is to build a clear path
from fundamentals to your niche applications: robotics, autonomous driving,
drones, manipulation, and Vision-Language-Action systems. See
[`learning_path.md`](learning_path.md) for the full map.

---

## How to use this repo

Four layers — learn by **understanding**, **running frameworks**, **integrating systems**, and **building**:

1. **Theory lessons** — `curriculum/part*/README.md`, written from zero
   (intuition → physics/math → how real systems use it), each with a **Project**,
   **"Check your understanding"**, and **"Go deeper"** references.
2. **Three runnable libraries** you can read end-to-end:
   - [`rl/`](rl/) — reinforcement learning (tabular → DQN → PPO/SAC → bandits,
     dynamic programming, Dyna model-based, behavioral cloning).
   - [`robotics/`](robotics/) — transforms, kinematics, **PID & LQR control**,
     Kalman/EKF, SLAM blocks, planning (A*/RRT), point clouds, sensor models.
   - [`dl/`](dl/) — the model architecture stack from scratch: attention,
     transformers, a small GPT, CNN/ViT, CLIP (zero-shot), and MiniSAM
     (promptable segmentation). The foundation for understanding VLMs and VLAs
     (Part 16) and segmentation models (Part 6).
3. **Framework/integration path** — optional-dependency demos and guides show how
   the same ideas appear in production libraries and APIs. The from-scratch
   libraries above are for learning the mechanisms; application code is built on
   the real frameworks. See [`applications/`](applications/) for systems built on
   PyTorch and real libraries (they do not import the teaching code):
   [`vla/`](applications/vla/) (end-to-end multi-task Vision-Language-Action
   policy, `python -m applications.vla.train`) and
   [`open_vocab_segmentation/`](applications/open_vocab_segmentation/) (SAM + CLIP
   text-prompted segmentation).
4. **Demos, curated notebooks & setup guides** — runnable `*_demo.py` in many
   parts, beginner-friendly notebooks for curriculum lessons, and setup guides
   for the heavy tools (MuJoCo, Isaac, ROS 2, CARLA, LeRobot).

```bash
pip install -e ".[box2d,dev]"     # core install
pytest -q                          # sanity tests across rl/, robotics/, and dl/
```

### Your hardware (RTX 4060 8 GB VRAM, 32 GB RAM) + A100 (Colab)
Lessons tag compute with **laptop** (runs on the laptop: MuJoCo, classical
perception/SLAM/control/planning, small-model RL/VLA, all demos) or **A100** (use
the A100: big VLA/LLM fine-tuning, 3D detection, NeRF/3DGS training, large-scale
Isaac Lab RL). With 8 GB VRAM, use 4-bit/LoRA/gradient-checkpointing for big models.

---

## The catalog (18 parts)

> **Folder numbers are IDs, not a strict order** — follow the **Study Plan** below.
> Parts 0–11 keep their original numbers; 12–17 were added later but slot into the
> plan where they belong.

**Foundations & decision-making**
| Part | Topic | Code |
|------|-------|------|
| [0](curriculum/part0_foundations/) | Math & ML foundations (linalg, prob, optimization, neural nets) | — |
| [1](curriculum/part1_reinforcement_learning/) | **Reinforcement Learning** — full taxonomy (16 modules) | [`rl/`](rl/) |
| [12](curriculum/part12_imitation_learning/) | **Imitation learning** (BC, DAgger, IRL, GAIL, diffusion policy, offline RL) | [`rl/agents/bc.py`](rl/agents/bc.py) |

**Robot fundamentals**
| Part | Topic | Code |
|------|-------|------|
| [2](curriculum/part2_robotics_physics/) | Robotics physics (dynamics, torque, contact, actuators) | — |
| [3](curriculum/part3_3d_math_transforms/) | 3D math & transforms (SE(3), quaternions, FK/IK, Jacobian) | [`robotics/transforms.py`](robotics/transforms.py) |
| [14](curriculum/part14_control_theory/) | **Control theory** (PID → LQR → MPC, stability) | [`robotics/control.py`](robotics/control.py) |

**Tools & infrastructure**
| Part | Topic | Code |
|------|-------|------|
| [4](curriculum/part4_simulation/) | Simulation (MuJoCo, Isaac Sim/Lab, Gazebo) | [`mujoco_demo.py`](curriculum/part4_simulation/mujoco_demo.py) |
| [13](curriculum/part13_ros2/) | **ROS 2** (nodes/topics/tf2, Nav2, MoveIt, RViz) | [`minimal_nodes.py`](curriculum/part13_ros2/minimal_nodes.py) |

**Perceive**
| Part | Topic | Code |
|------|-------|------|
| [5](curriculum/part5_sensors/) | Sensors (RGB, depth, LiDAR, radar, GPS, IMU) | [`robotics/sensors.py`](robotics/sensors.py) |
| [6](curriculum/part6_perception/) | Perception (detection, segmentation, point clouds, fusion) | [`robotics/pointcloud.py`](robotics/pointcloud.py) |
| [15](curriculum/part15_3d_vision/) | **3D vision** (multi-view geometry, SfM, NeRF, Gaussian Splatting) | — |
| [7](curriculum/part7_state_estimation_slam/) | State estimation & SLAM (KF/EKF, particle, graph SLAM) | [`robotics/filters.py`](robotics/filters.py) |

**Act**
| Part | Topic | Code |
|------|-------|------|
| [8](curriculum/part8_navigation_planning/) | Navigation & planning (A*, RRT, MPC, decision) | [`robotics/planning.py`](robotics/planning.py) |
| [9](curriculum/part9_manipulation_teleop/) | Manipulation & teleoperation (grasping, the VLA data engine) | [`robotics/kinematics.py`](robotics/kinematics.py) |
| [17](curriculum/part17_legged_locomotion/) | **Legged locomotion & humanoids** (parallel RL, sim-to-real) | — |

**Synthesis**
| Part | Topic | Code |
|------|-------|------|
| [16](curriculum/part16_foundation_models/) | **Foundation models** (attention→transformer→vision/language→VLM, diffusion, LoRA) | [`dl/`](dl/) |
| [10](curriculum/part10_vla_models/) | **Vision-Language-Action models** | [`vla_minidemo.py`](curriculum/part1_reinforcement_learning/07_vla_robotics/vla_minidemo.py) |
| [11](curriculum/part11_applications/) | Applications: autonomous driving, drones, humanoids | — |

---

## The Study Plan (recommended order)

```
 0 Foundations
   │
 1 Reinforcement Learning ───────────────► 12 Imitation Learning
   │  (00→07 core, then 08→15 advanced)         │
 2 Robotics Physics ─► 3 3D Math ─► 14 Control Theory
   │
 4 Simulation ─► 13 ROS 2
   │
 5 Sensors ─► 6 Perception ─► 15 3D Vision ─► 7 State Estimation & SLAM
   │
 8 Navigation & Planning ─► 9 Manipulation & Teleop ─► 17 Legged Locomotion
   │
 16 Foundation Models ─► 10 VLA Models ─► 11 Applications
```

A realistic pace from zero is **one part per 1–2 weeks**.

### Goal-based tracks (don't have to do everything in order)
- **VLA / robot foundation models (your goal):** 0 → 1 → 12 → 3 → 4 → 5 → 6 → 9 → 16 → 10.
- **Self-driving cars:** 0 → 3 → 5 → 6 → 15 → 7 → 8 → 14 → 11(driving).
- **Drones/UAVs:** 0 → 2 → 3 → 14 → 5 → 7 → 8 → 11(drones).
- **Humanoids/legged:** 0 → 1 → 2 → 3 → 14 → 4 → 17.
- **RL mastery (theory-deep):** 0 → all of Part 1 (00→15) → 12 → 14.

---

## The code, at a glance

```python
# rl/ — reinforcement learning
from rl.agents.ppo import PPO; from rl.agents.sac import SAC
from rl.agents.bandits import UCB1, ThompsonSampling          # exploration
from rl.agents.dynamic_programming import value_iteration      # planning
from rl.agents.dyna import DynaQ                                # model-based
python -m rl.train --algo ppo --env LunarLander-v3

# robotics/ — classical robotics (NumPy)
from robotics.transforms import quat_slerp, invert_transform
from robotics.kinematics import DHChain                        # FK/IK/Jacobian
from robotics.control import PID, lqr_gain                     # control
from robotics.filters import ExtendedKalmanFilter              # estimation/SLAM
from robotics.planning import astar, RRT                       # planning
from robotics.pointcloud import fit_plane_ransac, icp          # LiDAR

# dl/ — the model architecture stack from scratch (PyTorch)
from dl.attention import MultiHeadAttention                    # the core mechanism
from dl.transformer import GPT                                 # a small language model
from dl.vision import ViT                                      # a vision transformer
from dl.multimodal import clip_contrastive_loss, CrossAttentionFusion  # VLM fusion
```

Application-layer guides: [CARLA for driving](curriculum/part11_applications/carla_guide.md)
and [LeRobot for robot policies/VLAs](curriculum/part9_manipulation_teleop/lerobot_guide.md).

Runnable demos (CPU, no heavy deps): `mujoco_demo.py`, `perception_demo.py`,
`ekf_localization_demo.py`, `planning_demo.py`, `manipulation_demo.py`,
`control_demo.py`, `vla_minidemo.py`.

Curriculum folders include curated `notebook.ipynb` study workspaces. These are
intended to be beginner-friendly, detailed, and manually shaped around each
lesson rather than mechanically generated templates.

---

## Foundational references (all free)
- **RL:** Sutton & Barto, *Reinforcement Learning*; OpenAI Spinning Up.
- **Robotics:** Lynch & Park, *Modern Robotics*; Thrun et al., *Probabilistic Robotics*; Tedrake, *Underactuated Robotics*.
- **Vision/3D:** Szeliski, *Computer Vision*; Hartley & Zisserman, *Multiple View Geometry*.
- **Deep learning:** Karpathy *Zero to Hero*; *Dive into Deep Learning* (d2l.ai).
- **VLA/robot learning:** Open X-Embodiment, RT-2, OpenVLA, Octo, π0; Hugging Face **LeRobot**.

## License
BSD-3-Clause. Built on [Gymnasium](https://gymnasium.farama.org/) and the open robotics ecosystem.

---
*Start at [Part 0 — Foundations](curriculum/part0_foundations/), or jump to
[Part 1 — RL](curriculum/part1_reinforcement_learning/) if your math is solid.*
