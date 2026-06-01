# Learning Path — Mechanisms First, Frameworks Second, Real Systems Third

This repo is a personal curriculum for learning embodied AI deeply enough to
build niche applications in robotics, autonomous driving, drones, manipulation,
and Vision-Language-Action (VLA) systems.

The guiding rule is:

```text
mechanism → framework → real system
```

That means every important topic should eventually answer three questions:

1. **Mechanism:** what is the core idea, and can I implement a small readable
   version myself?
2. **Framework:** what standard library, API, simulator, or robotics framework do
   practitioners use for the same idea?
3. **Real system:** where does this component sit inside a robot, car, drone, or
   VLA application?

---

## 1. The three-pass method

### Pass 1 — Mechanisms first

Read the lesson and the local from-scratch implementation. The point is to see
what the framework will later hide.

Examples:

- RL: Bellman backups, replay buffers, PPO clipping, SAC entropy objectives.
- Robotics: SO(3)/SE(3), quaternions, FK/IK, PID/LQR, Kalman filtering.
- Perception: point-cloud downsampling, RANSAC, ICP, sensor noise.
- DL/foundation models: attention, transformers, ViT, CLIP-style contrastive
  alignment, cross-attention fusion.

### Pass 2 — Frameworks second

Run the equivalent idea through the standard toolchain. The goal is to become
comfortable with real APIs without treating them as magic.

Examples:

- RL: Gymnasium, Stable-Baselines3, TensorBoard, vectorized environments.
- DL: PyTorch modules, torchvision/timm, Hugging Face Transformers, PEFT/LoRA.
- Robotics math/control: SciPy Rotation, python-control, CasADi, Pinocchio,
  MoveIt 2, ros2_control.
- Perception/3D: OpenCV, Open3D, COLMAP, PyTorch3D, Nerfstudio.
- Simulation: MuJoCo, Isaac Sim/Lab, Gazebo, CARLA, AirSim/PX4 SITL.
- Robot data/VLA: LeRobot datasets and policies, ACT, diffusion policy,
  OpenVLA/SmolVLA-style inference and fine-tuning workflows.

### Pass 3 — Real systems third

Combine pieces into application stacks. This is where isolated algorithms become
embodied AI systems.

Examples:

- Autonomous driving: camera/LiDAR → perception → localization → prediction →
  planning → control → CARLA vehicle.
- Drone: IMU/GPS/barometer/camera → state estimation → waypoint planning →
  cascaded control → PX4/AirSim mission.
- Manipulation: camera/depth → object pose → grasp planning → IK → controller →
  LeRobot/MoveIt-style execution.
- VLA: image tokens + language instruction → multimodal backbone → action head →
  policy execution → data collection and fine-tuning.

---

## 2. Topic map: mechanism, framework, system

| Area | Mechanism first | Frameworks second | Real systems third |
|---|---|---|---|
| Reinforcement learning | `rl/agents/*`, `rl/buffers.py`, `rl/train.py` | Gymnasium, Stable-Baselines3, CleanRL, RLlib | robot/drone/car policy training and evaluation |
| Imitation learning | `rl/agents/bc.py`, toy demonstrations | LeRobot, ACT, diffusion policy, dataset tools | teleoperation data → robot policy → evaluation |
| Deep learning | `dl/attention.py`, `dl/transformer.py`, `dl/vision.py` | PyTorch, torchvision/timm, Hugging Face, PEFT | perception backbones, VLMs, VLA policy models |
| 3D math/robotics | `robotics/transforms.py`, `robotics/kinematics.py` | SciPy Rotation, ROS tf2, Pinocchio, MoveIt 2 | frame trees, arm control, sensor fusion |
| Control | `robotics/control.py` | python-control, CasADi/acados, ros2_control | tracking loops for arms, vehicles, drones |
| State estimation | `robotics/filters.py` | FilterPy, robot_localization, PX4 EKF2 concepts | localization and SLAM loops |
| Planning | `robotics/planning.py` | Nav2, OMPL, MoveIt 2 planners | navigation and manipulation planning |
| Perception/point clouds | `robotics/pointcloud.py`, demos | OpenCV, Open3D, Detectron/YOLO, COLMAP | camera/LiDAR perception pipelines |
| Simulation | simple env loops and demos | MuJoCo, Isaac Lab, Gazebo, CARLA, AirSim | sim-to-real experimentation and data generation |
| Applications | application guides | CARLA, PX4/MAVSDK, ROS 2 stacks, LeRobot | autonomous driving, drones, humanoids, VLA systems |

---

## 3. How to use each lesson notebook

Each curated `notebook.ipynb` should be treated as a guided study workspace, not
as a disposable generated template. A good notebook includes:

1. **Setup** — install/import the repo in a local notebook or Colab.
2. **Mechanism pass** — read the local lesson and identify the from-scratch code.
3. **Framework pass** — map the concept to real libraries and APIs.
4. **Real-system pass** — write down where this concept belongs in an embodied
   stack.
5. **Run cells** — execute local demos when the lesson has a runnable `*.py` file.

The notebooks should be edited intentionally as your understanding grows. Heavy
dependencies remain optional so the core repo stays readable and easy to install.

---

## 4. End-to-end study routes

### VLA / robot foundation models

```text
0 Foundations
→ 1 RL core + 12 Imitation
→ 3 Transforms + 9 Manipulation
→ 5 Sensors + 6 Perception
→ 16 Foundation Models
→ 10 VLA Models
→ LeRobot / robot-policy practice
```

### Autonomous driving

```text
0 Foundations
→ 3 3D Math
→ 5 Sensors
→ 6 Perception
→ 15 3D Vision
→ 7 State Estimation & SLAM
→ 8 Planning
→ 14 Control
→ 11 Autonomous Driving + CARLA
```

### Drones

```text
0 Foundations
→ 2 Robotics Physics
→ 3 3D Math
→ 14 Control
→ 5 Sensors
→ 7 State Estimation
→ 8 Planning
→ 11 Drones/UAV + PX4/AirSim/MAVSDK practice
```

### Manipulation / teleoperation

```text
0 Foundations
→ 3 3D Math
→ 14 Control
→ 4 Simulation
→ 9 Manipulation & Teleoperation
→ 12 Imitation Learning
→ 10 VLA Models
→ LeRobot / MoveIt 2 practice
```

---

## 5. What to add whenever a new topic appears

For any new lesson, add or plan these three artifacts:

1. A **mechanism** explanation and/or small implementation.
2. A **framework** demo or guide using standard tools.
3. A **real-system** note explaining how the concept plugs into robotics,
   autonomous driving, drones, or VLA systems.

This keeps the repo focused on the final direction instead of becoming a random
collection of disconnected algorithms.
