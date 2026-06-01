# Deep-dive: Drones / UAVs 🚁

Where **state estimation and control** dominate: a drone is fast, 3D, **under-
actuated**, and *unstable* — it falls without good estimation and control running
hundreds of times per second.

## Dynamics (Part 2)
A quadrotor has **4 motors** but **6 DOF** — it's **underactuated**: it can only
accelerate along its own "up" axis, so to move horizontally it must **tilt
first**. Thrust and torque come from differential rotor speeds:
- Total thrust (up/down), roll/pitch torque (tilt), yaw torque (spin).
- Highly nonlinear, fast, and unstable — the textbook hard control problem.

## Sensing & state estimation (Parts 5, 7) — the critical system
- **IMU** (gyro+accel) at ~1 kHz: the heartbeat. Integrated for attitude/velocity
  but **drifts** — must be fused.
- **GPS/RTK:** absolute position outdoors (RTK → cm for mapping/agriculture).
- **Downward camera / optical flow + rangefinder:** velocity hold indoors
  (GPS-denied).
- **Barometer / magnetometer:** altitude / heading.
- **VIO / EKF fusion:** tightly couple IMU + camera (+ GPS) for robust high-rate
  state. This *is* Part 7, and on a drone it's life-or-death — bad estimation =
  crash.

## Control architecture (Parts 2, 9) — cascaded loops
```
position setpoint ─► [POSITION ctrl] ─► velocity setpoint ─► [VELOCITY ctrl]
   ─► attitude setpoint ─► [ATTITUDE ctrl, fast inner loop] ─► motor speeds
```
- **Inner attitude loop** runs fastest (stabilize tilt) — PID.
- **Outer position loop** slower (go to waypoint).
- **MPC** for aggressive/agile flight with constraints.
- Build the cascade yourself with [`robotics/control.py`](../../robotics/control.py).

## Planning (Part 8)
- **Minimum-snap trajectories:** smooth 3D polynomials minimizing snap (4th
  derivative) for feasible, efficient flight (Mellinger & Kumar).
- **Obstacle avoidance:** onboard depth/LiDAR → local replanning; RRT* / motion
  primitives in 3D.

## Autopilot stacks & sims
- **PX4** and **ArduPilot:** the open-source flight-controller firmwares (the
  drone's "OS") — estimation + control + safety, battle-tested. Learn one.
- **Ground control:** QGroundControl / Mission Planner.
- **Simulators:** **gym-pybullet-drones** (RL-friendly, 💻 laptop), **Flightmare**,
  Microsoft **AirSim**, PX4 SITL + Gazebo.

## Learning for drones (Part 1)
- **RL for agile flight & racing:** Kaufmann et al. (2023, *Nature*) — RL +
  VIO beat human champions at drone racing. Trained in sim, deployed real
  (sim-to-real, Part 4).
- **RL hover/waypoint/wind-rejection policies:** great projects in
  gym-pybullet-drones with your Part-1 PPO/SAC.

## Applications
Aerial mapping/survey (RTK + photogrammetry), inspection (powerlines, wind
turbines), delivery, agriculture (spraying/monitoring), search-and-rescue,
cinematography, defense.

## Project ideas (matched to your hardware)
- 💻 Install **gym-pybullet-drones**; implement the cascaded PID; tune a stable
  hover. Then train **PPO** to hold position under wind.
- 💻 Run **PX4 SITL** + QGroundControl; fly a waypoint mission in sim.
- 🅰️ Reproduce an RL **waypoint/racing** policy; add domain randomization; study
  the VIO→control loop.

## Go deeper
- Mellinger & Kumar (2011), *Minimum-snap trajectory generation*.
- Kaufmann et al. (2023), *Champion-level drone racing using deep RL* (Nature).
- PX4 & ArduPilot docs; gym-pybullet-drones; *Small Unmanned Aircraft* (Beard & McLain, free).
