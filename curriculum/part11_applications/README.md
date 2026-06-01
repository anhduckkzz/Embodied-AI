# Part 11 — Applications: driving, drones, humanoids

> Everything you've learned — RL, 3D math, simulation, sensors, perception,
> SLAM, planning, manipulation, VLA — assembles into real autonomous systems.
> This part shows how the same building blocks recombine per domain. Deep-dives:
> [`autonomous_driving.md`](autonomous_driving.md) · [`drones_uav.md`](drones_uav.md).

The unifying architecture of *any* autonomous agent — the loop you've now built
piece by piece:

```
  SENSE ──► PERCEIVE ──► LOCALIZE/MAP ──► PREDICT ──► PLAN/DECIDE ──► CONTROL ──► (act)
 (Part 5)   (Part 6)      (Part 7)       (Part 6/8)    (Part 8/1)    (Part 9/2)
                                   ▲                                      │
                                   └──────────────── feedback ───────────┘
```

Two ways to build it: the **modular pipeline** above (interpretable, certifiable —
today's products) or **end-to-end learning / VLA** (Part 10 — the frontier).
Real systems are increasingly **hybrids**.

---

## 1. Autonomous driving 🚗 (full guide: [`autonomous_driving.md`](autonomous_driving.md))

The most mature, safety-critical autonomy domain.
- **Sensors:** cameras + LiDAR + radar + GPS/IMU + wheel odometry — heavy
  redundancy for all-weather safety (Part 5).
- **Perception:** 3D object detection (PointPillars/CenterPoint/BEVFormer), lane &
  drivable-area segmentation, traffic-light/sign recognition, multi-object
  tracking (Part 6).
- **Localization:** HD-map matching + LiDAR/visual SLAM + GPS/IMU fusion to
  centimeter accuracy (Part 7).
- **Prediction:** forecast every agent's future trajectory (the hardest part).
- **Planning:** route → behavior (lane change, yield) → trajectory, respecting
  car kinematics (Part 8).
- **Control:** steering/throttle/brake via PID/MPC (Part 9/2).
- **The debate:** Waymo-style **modular + LiDAR + HD-maps** vs Tesla-style
  **vision-first, increasingly end-to-end**. End-to-end and driving-VLAs are
  rising. Datasets: **nuScenes, Waymo Open, KITTI**; sims: **CARLA**, NVIDIA
  DRIVE.

## 2. Drones / UAVs 🚁 (full guide: [`drones_uav.md`](drones_uav.md))

Fast, 3D, weight- and compute-constrained — where **state estimation and control**
dominate.
- **Sensors:** IMU (critical), GPS/RTK, downward camera/optical-flow, barometer,
  sometimes LiDAR/depth (Part 5).
- **State estimation:** tight **VIO / EKF** fusion of IMU + vision/GPS at high rate
  — a drone *falls* without good estimation (Part 7).
- **Control:** a cascaded loop — attitude (inner, fast) → velocity → position
  (outer) — usually PID/MPC; underactuated dynamics (Part 2). **PX4** / **ArduPilot**
  are the open autopilot stacks.
- **Planning:** smooth **minimum-snap** 3D trajectories; obstacle avoidance with
  onboard depth (Part 8).
- **Autonomy & learning:** RL for agile flight and **drone racing** (champion-level
  results from RL + VIO); sim: **Flightmare, AirSim, gym-pybullet-drones**.

## 3. Humanoids & legged robots 🦿

The hardest dynamics, the biggest current hype (Figure, Tesla Optimus, Unitree,
Booster; NVIDIA **GR00T**).
- **Locomotion:** high-frequency balance and gait — trained with **massively
  parallel RL** (PPO in Isaac Lab, Part 4) + domain randomization for sim-to-real.
  ZMP/MPC classical methods coexist with learned policies (Part 2/8).
- **Whole-body manipulation:** combine legged locomotion with arm manipulation
  (Part 9).
- **Foundation models:** humanoid **VLAs** (GR00T) for general instruction-
  following — the convergence point of this whole curriculum.

## 4. Other domains (same toolbox)
- **Mobile/warehouse robots** (AMRs): 2D LiDAR SLAM + Nav2 (Parts 7–8) — the most
  *deployed* autonomy today.
- **Agriculture, inspection, space, underwater:** swap the sensor suite and
  dynamics; the SENSE→...→CONTROL loop is unchanged.

## 5. Cross-cutting realities (read before you build)
- **Safety & validation** — redundancy, fail-safes, formal methods, massive
  scenario testing. The bar rises with risk to humans.
- **Real-time & compute** — perception/control run on embedded GPUs (Jetson) at
  strict latencies. Efficiency (quantization, distillation) matters.
- **Edge cases / the long tail** — the last 1% (weird weather, rare scenarios) is
  most of the engineering. Data engines + sim (Part 4) target it.
- **Ethics, regulation, HRI** — autonomy lives in society; design for it.

## 🛠️ Capstone projects (pick your destination)
- **Driving:** build a mini perception→planning loop in **CARLA**; or train a
  lane-keeping policy from camera with **PPO/BC** (Part 1).
- **Drones:** fly **gym-pybullet-drones**; implement the cascaded PID
  ([`robotics/control.py`](../../robotics/control.py)); train an RL hover/waypoint
  policy.
- **Humanoid/legged:** train a quadruped to walk in **Isaac Lab** (PPO, Part 4),
  add domain randomization, study sim-to-real.
- **VLA:** fine-tune a VLA (Part 10) for a manipulation task and deploy in sim.

## ✅ Check your understanding
1. Draw the SENSE→CONTROL loop and label which part of this curriculum owns each box.
2. Why does driving use such a redundant sensor suite while drones minimize sensors?
3. What makes state estimation *the* critical system on a drone?
4. Why are humanoid locomotion policies trained with parallel RL in sim?
5. Modular vs end-to-end/VLA — give a concrete reason each is preferred somewhere.

## 📖 Go deeper
- Driving: nuScenes/Waymo dev-kits, CARLA docs, "End-to-End Autonomous Driving" survey.
- Drones: PX4 & ArduPilot docs; Kaufmann et al. (2023), *Champion-level drone racing with RL* (Nature).
- Humanoids: Isaac Lab locomotion examples; legged-robot RL papers (ANYmal, Unitree); NVIDIA GR00T.

🎉 **You've gone from the Bellman equation to embodied foundation models and real
autonomous systems.** The two libraries ([`rl/`](../../rl/), [`robotics/`](../../robotics/))
and these 11 parts are your launchpad — now go build a robot, a car, or a drone
that sees, thinks, and acts.
