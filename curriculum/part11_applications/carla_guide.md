# CARLA: a simulator for autonomous driving

CARLA is the standard open-source simulator for self-driving research. It
provides a 3D urban world built on Unreal Engine, realistic sensors (RGB, depth,
semantic segmentation, LiDAR, radar, GNSS, IMU), traffic, pedestrians, and
weather. It is where you can build and test the full driving stack from Part 11
without a real car.

Reference: the CARLA documentation (https://carla.org) and the Python API docs.

## 1. What CARLA is for

- Generate sensor data with ground-truth labels for perception (Part 6) and 3D
  vision (Part 15).
- Develop and test the driving pipeline: perception, localization, planning
  (Part 8), and control (Part 14) in closed loop.
- Train and evaluate driving policies (imitation, Part 12, or RL, Part 1),
  including end-to-end and driving-VLA approaches.
- Run reproducible scenarios for safety evaluation (the CARLA Leaderboard and
  scenario_runner).

## 2. Architecture: server and client

CARLA runs as a server (the simulator process, which needs a GPU) and you connect
to it with a Python client over a port. Your code is a client that spawns actors,
attaches sensors, reads data, and sends control commands.

## 3. Install

- Download the CARLA package (the server binary) for your platform from the CARLA
  releases. This is a large download and needs a discrete GPU.
- Install the Python client API to match the server version:

```bash
pip install carla        # the client library (match the server version)
```

On the RTX 4060, run the server at reduced quality (`-quality-level=Low`) and a
modest resolution. The 4060 is sufficient for development; large-scale training
benefits from a stronger or cloud GPU.

## 4. The client workflow

```
connect to server -> get world -> spawn a vehicle -> attach sensors
   -> drive (autopilot or your controller) -> read sensors each tick -> close
```

See [`carla_starter.py`](carla_starter.py) for a complete, commented example that
connects, spawns a car, attaches an RGB camera, enables the built-in autopilot,
and saves a few frames. It is written to run on a machine with CARLA installed and
to explain clearly if CARLA is not present.

## 5. Key API objects (the 20 percent you use most)

- `carla.Client` and `world`: connect and access the simulation.
- Blueprints: templates for vehicles, sensors, pedestrians.
- Actors: spawned entities (your vehicle, sensors). Remember to destroy them.
- Sensors: `sensor.camera.rgb`, `sensor.lidar.ray_cast`, `sensor.other.gnss`,
  `sensor.other.imu`, and more. Each delivers data via a callback.
- Vehicle control: `carla.VehicleControl(throttle, steer, brake)` for your own
  controller (Part 14), or `vehicle.set_autopilot(True)` to use the Traffic
  Manager.
- Traffic Manager: populates and controls background traffic.

## 6. Building the driving stack on CARLA

1. Perception (Part 6, 15): run a detector or segmenter on the RGB and LiDAR
   streams; calibrate sensor extrinsics (Part 3, 5).
2. Localization (Part 7): fuse GNSS and IMU, optionally map matching.
3. Planning (Part 8): A* or lattice global plan, then a local planner.
4. Control (Part 14): PID or MPC to track the trajectory using
   `carla.VehicleControl`.
5. Or end-to-end: record expert driving (autopilot) and clone it (Part 12), or
   train a policy with RL (Part 1). This is the path toward driving VLAs.

## 7. Datasets and benchmarks beyond CARLA

For real-world data, pair CARLA experiments with the public datasets in
[`autonomous_driving.md`](autonomous_driving.md): nuScenes, Waymo Open, KITTI,
Argoverse. Train perception on real data, validate behavior in CARLA.

## 8. Suggested progression

1. Run [`carla_starter.py`](carla_starter.py): spawn a car, attach a camera,
   enable autopilot, save frames. Confirm the loop works.
2. Add a LiDAR sensor; visualize the point cloud and run the Part 6 pipeline
   (RANSAC ground, clustering) on real CARLA scans.
3. Replace autopilot with your own controller: a simple lane-follow using
   perception plus a PID (Part 14).
4. Record autopilot driving and train a behavioral-cloning policy (Part 12) to
   imitate it from camera input.

## Check your understanding

1. Why does CARLA use a server-client split?
2. Which CARLA sensors map to the real sensors in Part 5?
3. How would you implement the Part 8 plus Part 14 stack against the CARLA API?
4. What is the role of the Traffic Manager?
5. How would you collect data for an imitation-learning driving policy in CARLA?
