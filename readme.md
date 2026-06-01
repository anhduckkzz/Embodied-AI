# Embodied AI: Zero → Master 🤖🧠🚗🚁

> From the Bellman equation to robots, self-driving cars, and drones that see,
> think, and act. A complete, hands-on curriculum that assumes **zero prior
> knowledge** and builds — with runnable code at every step — toward
> **Vision-Language-Action (VLA)** models and modern autonomy.

This repo started as a single LunarLander DQN. It is now a full **embodied
intelligence** curriculum: reinforcement learning, the **physics and math** of
robots, **simulation** (MuJoCo, Isaac Sim/Lab, ROS/Gazebo), **sensors &
perception** (RGB, depth, LiDAR, GPS/IMU, 3D detection, point clouds),
**state estimation & SLAM**, **navigation & planning**, **manipulation &
teleoperation**, **VLA foundation models**, and **applications** in autonomous
driving and drones.

---

## 🧭 How to use this repo

Every topic comes in three layers so you can learn by **reading**, **running**,
and **building**:

1. 📖 **Theory lessons** — `curriculum/partN_*/README.md`. Written from zero:
   intuition first, then the physics/math, then how it's used in real systems.
2. 🧪 **Runnable code** — two NumPy/PyTorch libraries you can read end-to-end:
   - [`rl/`](rl/) — reinforcement learning algorithms (DQN → PPO → SAC → BC).
   - [`robotics/`](robotics/) — transforms, kinematics, control, Kalman/EKF,
     SLAM building blocks, planning, point clouds, sensor models.
3. 📓 **Notebooks & tools** — Colab-ready notebooks and setup guides for the
   heavy software (MuJoCo, Isaac, ROS) you'll run on your own GPU.

```bash
pip install -e ".[box2d,dev]"     # core install
pytest -q                          # 26 sanity tests across rl/ and robotics/
```

### 💻 Your hardware (and how we'll use it)
- **RTX 4060 laptop (8 GB VRAM, 32 GB RAM):** MuJoCo, classical perception &
  SLAM, point clouds, planning, small-model RL/VLA fine-tuning, all notebooks.
- **A100 Colab Pro (40/80 GB):** big VLA models (OpenVLA/Octo), 3D object
  detection nets, Isaac Lab large-scale RL, anything memory-hungry.
- Each lesson tags compute with 💻 (laptop-fine) or 🅰️ (use the A100).

---

## 🗺️ The full roadmap

The curriculum is organized into **11 parts**. The dependency arrows tell you
what builds on what — but you can also dip into any part that excites you.

```
                ┌─────────────────────────── PERCEIVE ───────────────────────────┐
 Part 1 RL ──┐  Part 5 Sensors → Part 6 Perception → Part 7 SLAM ──┐
 Part 2 Physics ─┤                                                  ├─► Part 10 VLA ─► Part 11 Apps
 Part 3 3D Math ─┤  Part 8 Navigation & Planning ───────────────────┤   (the synthesis) (driving,
 Part 4 Sim   ───┘  Part 9 Manipulation & Teleop ───────────────────┘                     drones,
                └──────────────────────────── ACT ───────────────────────────────┘        humanoids)
```

| Part | Topic | What you'll master | Key code |
|------|-------|--------------------|----------|
| **1** | [Reinforcement Learning](curriculum/part1_reinforcement_learning/) | MDPs, DQN, PPO, SAC, imitation — decision-making | [`rl/`](rl/) |
| **2** | [Robotics Physics](curriculum/part2_robotics_physics/) | Rigid bodies, forces, torque, inertia, friction, contact, actuators | — |
| **3** | [3D Math & Transforms](curriculum/part3_3d_math_transforms/) | Frames, rotations, quaternions, SE(3), kinematics, Jacobians | [`robotics/transforms.py`](robotics/transforms.py), [`kinematics.py`](robotics/kinematics.py) |
| **4** | [Simulation](curriculum/part4_simulation/) | MuJoCo, Isaac Sim, Isaac Lab, PyBullet, ROS 2 + Gazebo + RViz | — |
| **5** | [Sensors](curriculum/part5_sensors/) | RGB, depth/stereo, LiDAR, radar, GPS, IMU, calibration | [`robotics/sensors.py`](robotics/sensors.py) |
| **6** | [Perception](curriculum/part6_perception/) | 2D/3D detection, segmentation, point clouds, depth, fusion | [`robotics/pointcloud.py`](robotics/pointcloud.py) |
| **7** | [State Estimation & SLAM](curriculum/part7_state_estimation_slam/) | Bayes/Kalman/EKF/particle filters, localization, SLAM | [`robotics/filters.py`](robotics/filters.py) |
| **8** | [Navigation & Planning](curriculum/part8_navigation_planning/) | A*, RRT, motion planning, costmaps, decision-making | [`robotics/planning.py`](robotics/planning.py) |
| **9** | [Manipulation & Teleop](curriculum/part9_manipulation_teleop/) | Control, grasping, trajectories, teleoperation, data collection | [`robotics/control.py`](robotics/control.py) |
| **10** | [VLA Models](curriculum/part10_vla_models/) | Vision-language-action foundation models for robots | — |
| **11** | [Applications](curriculum/part11_applications/) | Autonomous driving, drones/UAVs, humanoids | — |

---

## 🎯 Suggested learning tracks

You don't have to go strictly in order. Pick a track that matches your goal:

- **"I want to understand VLA / robot foundation models"** (your stated goal):
  **1 → 3 → 4 → 5 → 6 → 9 → 10**. (RL, the 3D math, a simulator, how robots
  perceive and act, then the VLA synthesis.)
- **"I want self-driving cars":** **3 → 5 → 6 → 7 → 8 → 11(driving)**.
- **"I want drones/UAVs":** **2 → 3 → 5 → 7 → 8 → 11(drones)**.
- **"I want the complete foundation":** go **1 → 11** in order.

A realistic pace from zero is **one part every 1–2 weeks**; each README ends with
a **Project**, **"Check your understanding"**, and **"Go deeper"** references.

---

## 🧪 The two code libraries

### `rl/` — reinforcement learning
Tabular Q-learning, DQN (+Double/Dueling/PER), REINFORCE, A2C, **PPO**, DDPG,
TD3, **SAC**, behavioral cloning — each a short, readable file whose comments
cite the update rules. Unified CLI:
```bash
python -m rl.train --algo ppo --env LunarLander-v3 --save runs/ppo.pt
```

### `robotics/` — classical robotics, from scratch (NumPy only)
Everything a perception/control stack needs, written to be read:
```python
from robotics.transforms import euler_to_matrix, quat_slerp, invert_transform
from robotics.kinematics import DHChain, planar_2link_ik
from robotics.filters import ExtendedKalmanFilter      # GPS+IMU fusion, EKF-SLAM
from robotics.planning import astar, RRT                # path planning
from robotics.pointcloud import fit_plane_ransac, icp   # LiDAR processing
from robotics.sensors import simulate_lidar_2d          # a virtual LiDAR
```
For production, graduate to **Open3D / PCL** (point clouds), **Pinocchio /
MuJoCo** (dynamics), **ROS 2 tf2 / Nav2 / MoveIt** (transforms, nav, manip),
and **PyTorch3D** (3D deep learning) — but learn the ideas here first.

---

## 📚 Foundational references (free)
- **RL:** Sutton & Barto, *Reinforcement Learning: An Introduction*; OpenAI Spinning Up.
- **Robotics:** Lynch & Park, *Modern Robotics* (book + free Coursera); Thrun et al., *Probabilistic Robotics* (SLAM/filters).
- **Perception/3D:** Szeliski, *Computer Vision: Algorithms and Applications*; Hartley & Zisserman, *Multiple View Geometry*.
- **Driving:** Pomerleau→nuScenes/Waymo open datasets; **Drones:** PX4/ArduPilot docs.
- **VLA:** Open X-Embodiment, RT-2, OpenVLA, Octo, π0 papers; Hugging Face **LeRobot**.

## License
BSD-3-Clause. Built on [Gymnasium](https://gymnasium.farama.org/) and the wider
open robotics ecosystem.

---
*Start with [Part 1 — Reinforcement Learning](curriculum/part1_reinforcement_learning/00_foundations/)
or jump to the [Part 4 — Simulation setup guides](curriculum/part4_simulation/) to get MuJoCo running today.*
