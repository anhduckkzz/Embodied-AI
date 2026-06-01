# Part 9 — Manipulation & Teleoperation

> Code: [`robotics/kinematics.py`](../../robotics/kinematics.py),
> [`robotics/control.py`](../../robotics/control.py), demo:
> [`manipulation_demo.py`](manipulation_demo.py). Manipulation = making a robot
> **arm/hand do useful physical work**. Teleoperation = a human driving the robot
> — which is how we **collect the demonstration data that trains VLA models**
> (Part 10). This part is the direct on-ramp to robot foundation models.

---

## 1. The manipulation pipeline

```
perceive object (Part 6) → choose a grasp → plan a path (Part 8) →
   compute joint targets (IK, Part 3) → control the joints (this part) → grip → move
```

Each stage uses earlier parts; manipulation is where the whole stack meets a
physical object.

## 2. Controlling an arm — from joint angles to motion

You command an arm in one of several **control modes**, low to high level:
- **Joint position control** — "set joint 2 to 30°." A PID per joint tracks it
  (reuse [`robotics/control.py`](../../robotics/control.py)). Simple, stiff.
- **Joint velocity / torque control** — command speeds or torques; needed for
  compliant, dynamic, contact-rich tasks.
- **Cartesian (task-space) control** — command the *end-effector* pose; the
  controller uses **IK** (Part 3) or the **Jacobian** (`τ = Jᵀ F`, "operational-
  space control") to translate to joints. How you say "move the hand 5 cm right."
- **Impedance / compliance control** — control the *relationship between force and
  motion* ("be springy"), so the arm yields on contact instead of smashing —
  essential for safe insertion, wiping, human contact.

> Gravity compensation + a Cartesian impedance controller is the modern baseline
> for a safe, teachable arm.

## 3. Trajectories — smooth, feasible motion

You don't jump to a target; you follow a **trajectory** (position over time) that
respects velocity/acceleration limits:
- **Joint-space** interpolation (cubic/quintic splines, trapezoidal velocity).
- **Cartesian** interpolation: straight-line position + **SLERP** for orientation
  (Part 3's [`quat_slerp`](../../robotics/transforms.py)).
- **Time parameterization** to stay within actuator limits.

## 4. Grasping — the hard, beautiful problem

Deciding *how to hold* an object:
- **Analytic / geometric** — reason about contact points, **force closure**
  (can the grasp resist any wrench?), friction cones. Works when you have a model.
- **Learned grasping** — predict grasp poses from vision (**Dex-Net**, **GraspNet**,
  **Contact-GraspNet**); train on millions of simulated grasps. The dominant
  approach for novel objects.
- **End-to-end** — a policy (RL/imitation/VLA) maps pixels straight to gripper
  actions, skipping explicit grasp planning.

Grippers range from simple **parallel-jaw** (2 fingers) to **dexterous multi-
finger hands** (hard to control, high-DOF) to **suction** (great for warehouses).

## 5. Teleoperation — humans in the loop (and the data engine for VLA)

A human controls the robot remotely/directly. Interfaces:
- **Kinematic / leader-follower** (e.g. **ALOHA**, **GELLO**) — move a small
  replica arm, the real arm copies. Intuitive, high-quality demos.
- **VR / spatial** controllers — map hand pose to end-effector (Cartesian
  control + IK).
- **Space-mouse / joystick** — 6-DOF jogging for precise tasks.

**Why this is the key to VLA:** every modern robot foundation model (RT-2,
OpenVLA, Octo, π0) is trained mostly on **teleoperated demonstrations**. A human
teleoperates hundreds/thousands of episodes of "pick up the cup," each logged as
`(camera images, language instruction, robot actions)`. That dataset is the fuel.
This is exactly **Behavioral Cloning** from Part 1 / Module 06 — at robot scale.

```
teleoperation  ──collect──►  (obs, instruction, action) dataset  ──BC (Part 1)──►  policy / VLA
     ▲                                                                                  │
     └──────────────── DAgger / RL fine-tuning closes the loop (Parts 1 & covariate shift) ──┘
```

## 6. Data, datasets, and tooling
- **Open X-Embodiment** — ~1M+ trajectories across 22 robot types; the
  "ImageNet of robot manipulation."
- **LeRobot** (Hugging Face) — open library + datasets + VLA models; the easiest
  place to start training a real policy on your A100.
- **RoboMimic / RoboSuite, LIBERO, ManiSkill, Meta-World** — manipulation
  benchmarks and demo datasets you can run on the 4060/A100.

## 7. From classical to learned manipulation
- **Classical** (model-based): perception → grasp planning → motion planning →
  control. Reliable when models are good; brittle to novelty/clutter.
- **Learned** (imitation/RL/VLA): demonstrations + a policy net; generalizes to
  novel objects/scenes, needs lots of data. The field is rapidly moving here —
  which is exactly why you're heading to Part 10.

## 🛠️ Project
1. Run [`manipulation_demo.py`](manipulation_demo.py): a planar arm uses **IK**
   to reach a sequence of targets and traces a **pick-and-place** trajectory —
   then a per-joint **PID** "executes" it. Visualizes the arm and path.
2. `pip install gymnasium-robotics` and try `FetchPickAndPlace` — train your
   Part-1 **SAC** (sparse reward + hindsight replay) or clone a scripted expert
   with **BC** (Part 1 `bc.py`).
3. (Stretch) Install **LeRobot**, load an Open-X dataset episode, and inspect the
   `(image, instruction, action)` tuples — the raw material of a VLA.

## ✅ Check your understanding
1. Name the manipulation pipeline stages and which earlier part each uses.
2. Joint-space vs Cartesian vs impedance control — when do you want each?
3. What is force closure, and how do learned graspers sidestep needing it?
4. Why is teleoperation the linchpin of modern VLA training?
5. How does covariate shift (Part 1) limit pure-BC manipulation, and what fixes it?

## 📖 Go deeper
- Lynch & Park, *Modern Robotics*, Ch. 11–13 (control, grasping).
- ALOHA / Mobile-ALOHA, GELLO papers (low-cost teleop); Dex-Net, Contact-GraspNet.
- Hugging Face **LeRobot**; Open X-Embodiment; RoboSuite/ManiSkill docs.

➡️ **Next:** [Part 10 — VLA Models](../part10_vla_models/): combine perception,
language, and these actions into one foundation-model policy.
