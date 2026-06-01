# Deep-dive: Autonomous Driving 🚗

How the curriculum's pieces become a self-driving car. The classic **modular
stack** (with notes on the **end-to-end** shift).

## The stack, layer by layer

### 1. Sensing (Part 5)
- **Cameras** (6–12): semantics, traffic lights, signs, lanes. Cheap, high-res.
- **LiDAR** (1–5): accurate 3D geometry, day/night. (Tesla famously omits it;
  Waymo/most others rely on it.)
- **Radar:** velocity via Doppler, all-weather, long range.
- **GPS/RTK + IMU + wheel odometry:** ego-motion and global position.
- **Redundancy is the point:** any one sensor fails in some condition; overlap
  keeps the system safe. All must be **calibrated** (extrinsics) and **time-synced**.

### 2. Perception (Part 6)
- **3D object detection & tracking:** vehicles, pedestrians, cyclists as oriented
  3D boxes with velocities (PointPillars, CenterPoint, BEVFormer; LiDAR/camera
  fusion). Increasingly unified in a **Bird's-Eye-View (BEV)** representation.
- **Semantic segmentation:** drivable area, lane markings, curbs.
- **Traffic-light/sign recognition.**

### 3. Localization & mapping (Part 7)
- **HD maps:** centimeter-accurate prior maps of lanes, signs, geometry.
- **Map matching + LiDAR/visual SLAM + GPS/IMU fusion (EKF):** localize within
  the HD map to ~10 cm. (Map-light/mapless approaches are an active push.)

### 4. Prediction (the hardest part)
- Forecast the future trajectories of every road agent, multimodally ("the car
  *might* turn or go straight"). Learned models over agent histories + map
  context. Errors here cause most planning failures.

### 5. Planning & decision (Part 8)
- **Route planning:** road-level path (a navigation graph).
- **Behavior planning:** discrete decisions — lane change, yield, merge,
  stop-for-pedestrian. FSMs/behavior trees or learned policies.
- **Motion planning:** a smooth, **kinematically feasible** trajectory
  (Dubins/Reeds-Shepp, hybrid-A*, optimization) that's comfortable and safe.

### 6. Control (Parts 2, 9)
- Track the trajectory with **steering + throttle + brake**. **PID** for
  lateral/longitudinal, or **MPC** for constraint-aware, anticipatory control.

## Modular vs end-to-end (the central debate)
- **Modular** (Waymo-style): interpretable, debuggable, certifiable; each module
  testable. Heavy engineering; errors compound across modules.
- **End-to-end / driving-VLA** (Tesla FSD direction, research like UniAD): a
  network maps sensors → trajectory/controls, trained on huge fleets. Generalizes,
  less hand-engineering; harder to verify and explain. **Hybrids** (learned
  components in a safety scaffold) are the practical middle.

## Levels of autonomy (SAE)
L0 (none) → L1 (assist) → L2 (hands-on supervision, most "self-driving" today) →
L3 (eyes-off, conditional) → L4 (no driver in a domain, e.g. Waymo robotaxis) →
L5 (anywhere). The jump from L2 to L4 is mostly about the **long tail** of edge
cases, not the average case.

## Tools, sims, datasets
- **Simulators:** **CARLA** (open, the standard for research), NVIDIA DRIVE Sim,
  LGSVL. Great for your laptop (CARLA) and A100 (training).
- **Datasets:** **nuScenes**, **Waymo Open**, **KITTI**, Argoverse (prediction).
- **Frameworks:** **Autoware** (open full-stack on ROS 2), Apollo.

## Project ideas (matched to your hardware)
- 💻 Run **CARLA**; build a perception→planning loop using a pretrained detector +
  A* + PID. Visualize in its dashboard.
- 🅰️ Train **BEV perception** or a **lane-keeping** policy (BC/PPO from camera) on
  the A100; evaluate in CARLA.
- Study **UniAD** / end-to-end driving papers and map each block to Parts 5–9.

## Go deeper
- Surveys: "End-to-End Autonomous Driving: Challenges and Frontiers" (2024).
- CARLA & Autoware docs; nuScenes/Waymo tutorials; UniAD, BEVFormer papers.
