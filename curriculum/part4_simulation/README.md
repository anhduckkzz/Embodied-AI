# Part 4 — Simulation: MuJoCo, Isaac, and ROS/Gazebo

> Robots learn and are tested in **simulation** first — it's faster, safer,
> cheaper, and infinitely repeatable. This part gets the major simulators
> running on **your** machine and explains *which to use when*. Heavy installs
> (Isaac) run on your RTX 4060 / A100, not in a notebook sandbox.

**Why simulate?** A real robot collects ~1 trajectory per real second and breaks
when it falls. A simulator runs thousands of robots in parallel, faster than
real time, with perfect ground-truth labels — then you cross the **sim-to-real
gap** (Part 4.6). Modern legged robots and many manipulation policies are
trained *entirely* in sim.

---

## 1. The landscape — which simulator, and why

| Simulator | Best for | Physics | GPU-parallel | Your hardware |
|-----------|----------|---------|--------------|---------------|
| **MuJoCo** | RL research, manipulation, legged, contact-rich | excellent, fast, stable contacts | via MJX (JAX) | 💻 4060 fine (great start) |
| **PyBullet** | quick prototyping, education, free | good | no | 💻 trivial install |
| **Isaac Sim** | photorealistic, sensors (RGB/LiDAR/depth), digital twins | PhysX | yes | 🅰️ needs RTX GPU (4060 ok-ish, A100 better) |
| **Isaac Lab** | large-scale RL on Isaac Sim (1000s of envs) | PhysX (GPU) | **massively** | 🅰️ RTX GPU required |
| **Gazebo (+ROS 2)** | full robot stack, sensor sim, ROS integration | ODE/Bullet/DART | no | 💻 4060 fine |
| **Genesis / Brax / others** | newest fast/differentiable sims | varies | yes | situational |

**Recommended path for you:**
1. **Start with MuJoCo** on the laptop — easiest, the RL standard, runs today.
   (Detailed guide: [`mujoco_guide.md`](mujoco_guide.md), demo:
   [`mujoco_demo.py`](mujoco_demo.py).)
2. **Add ROS 2 + Gazebo + RViz** to learn the real robot software stack and
   visualization. ([`ros2_gazebo_guide.md`](ros2_gazebo_guide.md).)
3. **Graduate to Isaac Sim/Lab** for photoreal sensors and large-scale RL when
   you need them. ([`isaac_guide.md`](isaac_guide.md).)

## 2. MuJoCo (Multi-Joint dynamics with Contact) 💻

The de-facto standard for RL and robot control research (now open-source, by
Google DeepMind). Tiny install, superb contact physics, the engine behind most
locomotion/manipulation papers and the Gymnasium MuJoCo envs.

```bash
pip install mujoco                 # the simulator + Python bindings
pip install "gymnasium[mujoco]"    # standard RL tasks (Ant, Humanoid, ...)
python curriculum/part4_simulation/mujoco_demo.py   # runnable: drop & control a body
```

You describe a robot in an **MJCF** XML file (bodies, joints, geoms, actuators,
sensors). MuJoCo integrates the dynamics; you read state and write actuator
commands each step — exactly the `env.step(action)` loop from Part 1, but now
with real rigid-body physics. **Train your Part-1 SAC/PPO on `Ant-v5` or
`Humanoid-v5` and you're doing modern continuous-control robotics.**

## 3. Isaac Sim & Isaac Lab (NVIDIA) 🅰️

- **Isaac Sim** — a photorealistic robotics simulator built on NVIDIA Omniverse.
  Its superpower is **high-fidelity sensors**: physically-based RGB cameras,
  depth, **LiDAR**, IMU, plus domain randomization — ideal for training/testing
  *perception* (Parts 5–6), not just control.
- **Isaac Lab** (successor to Isaac Gym / OmniIsaacGymEnvs) — a framework on top
  of Isaac Sim for **GPU-parallel RL**: thousands of robot instances stepping on
  the GPU at once, so a quadruped locomotion policy trains in *minutes*. This is
  how Unitree/ANYmal-style policies are made.

Install needs an RTX GPU + the Omniverse stack (several GB). See
[`isaac_guide.md`](isaac_guide.md). On your 4060 you can run Isaac Sim for
perception and small RL; for big parallel RL, use a cloud A100/L4. Workflow:
train PPO in Isaac Lab (massively parallel) → deploy with domain randomization.

## 4. ROS 2 + Gazebo + RViz — the real robot stack 💻

If MuJoCo/Isaac are the *physics*, **ROS 2** is the *nervous system* real robots
run on, and you should know it:
- **ROS 2 (Robot Operating System)** — middleware: nodes communicate over
  **topics** (sensor streams, commands) and **services/actions**. Standardizes
  how perception, planning, and control modules talk. The **tf2** library
  manages the live transform tree (Part 3!).
- **Gazebo** — physics + sensor simulator that plugs into ROS 2; spawn a robot,
  simulate its cameras/LiDAR, and run your real ROS code against it.
- **RViz** ("Viz") — the **3D visualization** tool: see the robot model, live
  sensor data (point clouds, camera, laser scans), TF frames, planned paths, and
  costmaps. *Indispensable for debugging perception and navigation* — when your
  LiDAR points land in the wrong place, RViz is where you'll see it.

```bash
# Ubuntu (native or WSL2). See ros2_gazebo_guide.md for the full setup.
# After install:
ros2 launch gazebo_ros gazebo.launch.py      # simulator
rviz2                                         # visualization
ros2 topic list                               # see the data streams
```

## 5. Anatomy of a simulation loop (universal)

Every simulator, under the hood, is the loop you already know:

```
load world (robot model + objects + sensors)
for each timestep dt:
    apply actuator commands (torques/positions) ← your controller / policy
    step physics: solve dynamics + contacts, integrate state
    read sensors (joint encoders, camera, LiDAR, IMU) ← perception input
    render (optional)
```

`env.step(action)` in Gymnasium *is* this loop. MuJoCo/Isaac/Gazebo differ in
*physics fidelity*, *sensor realism*, and *parallelism* — not in concept.

## 6. The sim-to-real gap (the central challenge)

A policy perfect in sim can fail on hardware because sim ≠ reality:
- **Dynamics mismatch** — friction, motor lag, mass, latency are slightly wrong.
- **Perception mismatch** — real cameras have noise, blur, lighting; sim is clean.
- **Unmodeled effects** — cable drag, gear backlash, deformable contact.

Bridging techniques (you'll use these constantly):
- **Domain randomization** — randomize masses, friction, textures, lighting,
  latency during training so the policy is *robust* to the real value.
- **System identification** — measure the real robot and match the sim to it.
- **Real-world fine-tuning** — a little on-robot data (Part 9 teleop, Part 1 RL).

This is *the* reason the whole stack matters: good physics (Part 2), good
sensor models (Part 5), and robust policies (Part 1) all serve sim-to-real.

## 🛠️ Project
1. Run [`mujoco_demo.py`](mujoco_demo.py): watch a body fall under gravity, then
   add a PID ([`robotics/control.py`](../../robotics/control.py)) to hold a joint
   at a target angle.
2. `pip install "gymnasium[mujoco]"` and train your Part-1 **SAC** on
   `Ant-v5` — your first physically-simulated robot locomotion.
3. (Stretch) Install ROS 2 + Gazebo, spawn TurtleBot, and view its LiDAR in RViz.

## ✅ Check your understanding
1. Give three reasons to train robots in simulation.
2. MuJoCo vs Isaac Lab — when would you pick each?
3. What does RViz show you, and why is it essential for debugging perception?
4. Describe the universal simulation loop in your own words.
5. What is the sim-to-real gap, and name two ways to bridge it.

## 📖 Go deeper
- MuJoCo docs + tutorials; `dm_control` suite. · Isaac Lab docs (NVIDIA).
- ROS 2 "Humble/Jazzy" tutorials; *A Gentle Introduction to ROS* (O'Kane, free).
- Tobin et al. (2017), *Domain Randomization*; OpenAI (2019), *Dexterous Hand*.

➡️ **Next:** [Part 5 — Sensors](../part5_sensors/): how robots perceive — RGB,
depth, LiDAR, GPS, IMU — and the noise you must handle.
