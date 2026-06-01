# Part 5 — Sensors: how robots perceive the world

> Code: [`robotics/sensors.py`](../../robotics/sensors.py). A robot is only as
> good as what it can measure. This part covers every major sensor — what it
> measures, its physics, its noise, and its coordinate frame — because **all
> perception (Part 6) and state estimation (Part 7) start with raw sensor data**.

The golden rule: **every sensor lies a little.** It has noise, bias, limited
range/resolution, and lives in its own frame. Knowing *how* each lies is what
lets you fuse them into a reliable picture (Part 7).

---

## 1. Cameras (RGB) — rich, cheap, ambiguous

A camera maps the 3D world to a 2D grid of pixels via the **pinhole model**:

```
[u]       [fx  0  cx] [X/Z]
[v]  =  K  [ 0 fy  cy] [Y/Z]     (project a 3D point (X,Y,Z) in the camera frame
[1]       [ 0  0   1] [ 1 ]       to pixel (u,v); K = intrinsic matrix)
```

- **Intrinsics** `K` (focal lengths `fx,fy`, principal point `cx,cy`) + **lens
  distortion** come from **camera calibration** (a checkerboard + OpenCV).
- **Strengths:** dense color/texture, semantics (you can *recognize* things),
  cheap, high-res. The backbone of VLA vision.
- **Weakness:** **loses depth** — a single image can't tell scale (is it a small
  near object or a big far one?). Lighting/motion blur hurt. Solutions: stereo,
  depth sensors, or learned monocular depth.

## 2. Depth & stereo — recovering the third dimension

- **Stereo cameras:** two cameras a known baseline `b` apart. The horizontal
  pixel shift (**disparity** `d`) of the same point gives depth:
  `Z = f · b / d`. Bigger baseline → better far-range depth.
- **RGB-D (structured light / ToF):** e.g. Intel RealSense, Kinect. Project IR
  pattern or measure light **time-of-flight** to get per-pixel depth directly.
  Great indoors/short range; struggles in sunlight and on shiny/black surfaces.
- Output is a **depth image**, which un-projects to a **point cloud** (Part 6)
  using the same intrinsics `K`.

## 3. LiDAR — geometry you can trust

**Li**ght **D**etection **a**nd **R**anging: fire laser pulses, time the echo,
get precise range. A spinning/solid-state unit produces a 3D **point cloud**
(often 100k–2M points at 10–20 Hz), each point `(x, y, z, intensity)`.

- **Strengths:** accurate metric 3D geometry, works in darkness, long range
  (100m+). The geometric backbone of most self-driving stacks.
- **Weaknesses:** expensive (dropping fast), sparse at range, no color/semantics,
  degraded by rain/fog/dust, mirror-like surfaces.
- **2D LiDAR** (single plane) is cheap and ubiquitous on indoor robots/vacuums —
  exactly what [`robotics.sensors.simulate_lidar_2d`](../../robotics/sensors.py)
  models (ray-casting to ranges, then to points).

## 4. Radar — robust velocity, coarse shape

Radio waves instead of light. **Doppler** gives **instantaneous radial velocity**
directly (huge for tracking moving cars), and it **sees through rain/fog/dust**
where LiDAR/cameras fail. Lower spatial resolution. Standard on production cars
(adaptive cruise, blind-spot).

## 5. GPS / GNSS — absolute position, slowly

Trilateration from satellites gives global `(lat, lon, alt)`.
- **Plain GPS:** ~1–5 m accuracy, 1–10 Hz, **no orientation**, fails indoors / in
  "urban canyons" / under trees (multipath).
- **RTK / differential GPS:** centimeter accuracy with a base station — used by
  agriculture robots, survey drones, AV ground truth.
- Modeled by [`robotics.sensors.GPS`](../../robotics/sensors.py) (Gaussian noise).
  Because it's noisy and low-rate, GPS is always **fused** with IMU/odometry.

## 6. IMU — fast motion, drifts

An **Inertial Measurement Unit** = **accelerometer** (linear acceleration +
gravity) + **gyroscope** (angular velocity), often + magnetometer (heading).
- **Strengths:** high rate (100–1000 Hz), self-contained, no external infra.
- **Weakness:** you must **integrate** to get velocity/position, so **bias and
  noise accumulate into unbounded drift** (see
  [`robotics.sensors.IMU`](../../robotics/sensors.py)). Seconds of dead-reckoning
  are fine; minutes are not.
- Therefore IMU is **fused** with vision (VIO), GPS, or wheels. The cheap
  **complementary filter** ([`robotics.filters`](../../robotics/filters.py))
  blends gyro + accel for attitude; the EKF does the full job (Part 7).

## 7. Proprioception — the robot sensing itself

- **Joint encoders** — joint angles (feed FK, Part 3). Very accurate.
- **Wheel odometry** — integrate wheel rotation for position; drifts with slip.
- **Force/torque sensors** — contact forces for manipulation (Part 9).

## 8. Calibration & synchronization — the unglamorous essentials

Multiple sensors are useless until you know **where each is** and **when each
fired**:
- **Intrinsic calibration** — a sensor's internal params (camera `K`, distortion).
- **Extrinsic calibration** — the rigid transform between sensors,
  `T_camera_from_lidar`, so a LiDAR point lands on the right pixel (Part 3 math!).
- **Time sync** — align timestamps across sensors moving at different rates;
  errors here look like phantom misalignments.

> 80% of real-robot perception bugs are a bad **extrinsic** or a **timing**
> offset — not the fancy neural net.

## 9. Choosing a sensor suite (by application)

| Application | Typical suite | Why |
|-------------|---------------|-----|
| Self-driving car | cameras + LiDAR + radar + GPS/IMU + wheel odom | redundancy & all-weather |
| Indoor mobile robot | 2D/3D LiDAR + camera + IMU + wheel odom | cheap, GPS-denied |
| Drone | camera + IMU + GPS (+ optical flow + barometer) | weight-limited, outdoor |
| Manipulator arm | wrist camera + joint encoders + F/T sensor | precise local sensing |

## 🛠️ Project
Use [`robotics/sensors.py`](../../robotics/sensors.py): place circular obstacles,
`simulate_lidar_2d` from a robot pose, convert with `ranges_to_points`, and plot
the scan. Then add `GPS` and `IMU` noise to a moving trajectory and *see* the
drift — motivating the filters in Part 7. (Runnable demo:
[`../part6_perception/perception_demo.py`](../part6_perception/perception_demo.py).)

## ✅ Check your understanding
1. Why can't a single RGB image recover absolute scale/depth?
2. Stereo depth: write `Z` in terms of focal length, baseline, disparity.
3. LiDAR vs radar vs camera — one strength and one weakness of each.
4. Why does an IMU drift, and what do we fuse it with?
5. Extrinsic vs intrinsic calibration — what does each give you?

## 📖 Go deeper
- Szeliski, *Computer Vision*, Ch. 2 (image formation) + camera models.
- Thrun et al., *Probabilistic Robotics*, Ch. 6 (sensor models).
- OpenCV camera-calibration tutorial; Intel RealSense / Velodyne docs.

➡️ **Next:** [Part 6 — Perception](../part6_perception/): turn these raw streams
into objects, depth, point clouds, and a usable 3D understanding.
