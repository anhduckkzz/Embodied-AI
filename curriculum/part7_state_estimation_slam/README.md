# Part 7 — State Estimation & SLAM: where am I, and what's around me?

> Code: [`robotics/filters.py`](../../robotics/filters.py), demo:
> [`ekf_localization_demo.py`](ekf_localization_demo.py). This part fuses the
> *motion model* (Part 2/3) with *noisy sensors* (Part 5/6) **over time** to
> answer the two questions every mobile robot must: **"Where am I?"**
> (localization) and **"What does the world look like?"** (mapping) — together,
> **SLAM**.

The unifying idea is the **Bayes filter**: maintain a probabilistic **belief**
over the state, **predict** it forward with motion, and **correct** it with
measurements. Kalman/EKF/particle filters are all the same loop with different
math. You already built the filters in code — here's the why.

---

## 1. Why probabilistic? Because everything is uncertain

Motion is imperfect (wheels slip), sensors are noisy (Part 5). So we never know
the state exactly — we track a **distribution** (mean + covariance). The width of
that distribution *is* the robot's uncertainty, and managing it well is the
difference between a robot that localizes and one that gets lost.

## 2. The Bayes filter loop (the master algorithm)

```
belief_0  (prior: where we think we start)
repeat:
   PREDICT:  push belief through the motion model with control u   → uncertainty GROWS
   UPDATE:   correct belief with measurement z (Bayes' rule)        → uncertainty SHRINKS
```

Every filter below is this loop. The choice of representation (Gaussian vs
particles, linear vs nonlinear) gives the variants.

## 3. Kalman Filter (KF) — the optimal linear case

When motion and sensors are **linear** and noise is **Gaussian**, the belief
stays Gaussian and the KF is provably optimal. Two steps:

```
predict:  x = F x + B u ;     P = F P Fᵀ + Q       (Q = process noise)
update:   K = P Hᵀ (H P Hᵀ + R)⁻¹                  (Kalman gain)
          x = x + K (z − H x) ;  P = (I − K H) P    (R = measurement noise)
```

The **Kalman gain `K`** is the star: it blends prediction vs measurement by their
relative confidence. Trust the sensor (small `R`) → `K` large → follow `z`.
Trust the model (small `Q`) → `K` small → ignore noisy `z`. Implemented and
tested in [`KalmanFilter`](../../robotics/filters.py) (it tracks a constant-velocity
target from noisy position measurements).

## 4. Extended Kalman Filter (EKF) — the real-robot workhorse

Real motion (turning!) and sensors (range/bearing, cameras) are **nonlinear**.
The EKF handles this by **linearizing** `f` and `h` with their Jacobians at the
current estimate each step — otherwise identical to the KF.

```python
from robotics.filters import ExtendedKalmanFilter
ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R, x0, P0)
ekf.predict(u);  ekf.update(z)
```

EKF powers GPS+IMU fusion, wheel-odometry+LiDAR localization, and classic
EKF-SLAM. The runnable [`ekf_localization_demo.py`](ekf_localization_demo.py)
localizes a moving robot from **noisy range-bearing to known landmarks** — and
shows the estimate beating raw dead-reckoning. (Variants worth knowing: **UKF**
for stronger nonlinearity without Jacobians; **error-state EKF** for IMU/VIO.)

## 5. Particle Filter (Monte Carlo Localization)

When the belief is **non-Gaussian** (e.g. "I could be in any of these 3
corridors"), represent it with **many weighted samples (particles)**:

```
1. propagate each particle through the motion model (+ noise)
2. weight each by how well it explains the measurement
3. resample: keep high-weight particles, drop low-weight ones
```

This is **MCL/AMCL** — the standard way a ROS robot localizes in a known map. It
handles the "kidnapped robot" and multimodal cases the EKF can't.

## 6. SLAM — Simultaneous Localization And Mapping

The chicken-and-egg problem: to build a map you need to know where you are; to
know where you are you need a map. **SLAM solves both at once.**

- **Filter-based SLAM** (EKF-SLAM, FastSLAM): the state includes the robot pose
  *and* all landmark positions; the filter updates them jointly.
- **Graph-based SLAM** (modern standard): build a **pose graph** — nodes are
  poses/landmarks, edges are measured constraints — and solve a big nonlinear
  least-squares (**bundle adjustment** / factor graphs, e.g. **g2o, GTSAM,
  Ceres**). **Loop closure** (recognizing a previously seen place) snaps
  accumulated drift back into a consistent map.

By sensor:
- **Visual SLAM / VO** — cameras: **ORB-SLAM** (features), **DSO** (direct).
  **VIO** adds an IMU (used on drones, AR/VR, the Apple Vision Pro).
- **LiDAR SLAM** — point clouds + ICP/NDT scan matching: **LOAM**, **Cartographer**,
  **LIO-SAM** (LiDAR-inertial). The geometric, accurate route for cars/robots.

The front-end (perception: features/scan-matching, Part 6) + back-end
(optimization) = a live map and a drift-corrected trajectory.

## 7. The estimation → action loop

State estimation closes the loop with everything else:
- Perception (6) provides measurements (landmarks, scans, detections).
- The filter/SLAM fuses them into pose + map.
- Planning (8) uses the map + pose to choose a path.
- Control (9) executes it; the motion becomes the next predict step. Repeat.

## 🛠️ Project
1. Run [`ekf_localization_demo.py`](ekf_localization_demo.py): compare EKF
   localization error vs raw dead-reckoning as the robot drives. Tune `Q`/`R`
   and watch the trade-off.
2. Implement a tiny **particle filter** for 1D localization in a known map (10
   landmarks). Visualize the particles collapsing onto the true pose.
3. (Stretch) Run **slam_toolbox** (ROS 2) or **ORB-SLAM3** on a public dataset
   (TUM RGB-D / KITTI) and watch the map build in RViz.

## ✅ Check your understanding
1. State the two steps of the Bayes filter and what each does to uncertainty.
2. What does the Kalman gain trade off, and how do `Q`/`R` control it?
3. Why do real robots need the EKF instead of the plain KF?
4. When is a particle filter better than an EKF?
5. What is loop closure and why is it essential in SLAM?

## 📖 Go deeper
- Thrun, Burgard & Fox, *Probabilistic Robotics* — the bible (filters, MCL, SLAM).
- Cadena et al. (2016), "Past, Present, and Future of SLAM" (survey).
- ORB-SLAM3, Cartographer, LIO-SAM papers; GTSAM/Ceres tutorials.

➡️ **Next:** [Part 8 — Navigation & Planning](../part8_navigation_planning/):
use the pose + map to decide *where to go* and *how to move there*.
