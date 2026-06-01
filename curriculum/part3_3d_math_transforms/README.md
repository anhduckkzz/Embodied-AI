# Part 3 — 3D Math & Transforms (the language of robots)

> Code: [`robotics/transforms.py`](../../robotics/transforms.py),
> [`robotics/kinematics.py`](../../robotics/kinematics.py).
> This is the single most-used math in all of robotics, perception, and graphics.
> Master it and 90% of "why is my sensor data in the wrong place" bugs vanish.

Everything a robot knows lives in some **coordinate frame**: the camera sees in
*camera* coordinates, the LiDAR in *LiDAR* coordinates, the arm moves in *base*
coordinates, the map is in *world* coordinates. To do anything useful you
constantly **transform** data between frames. This part makes that second nature.

---

## 1. Frames and points

A **frame** is an origin + three perpendicular axes (x, y, z), right-handed. A
point's coordinates are *relative to a frame*: the same physical point has
different numbers in the camera frame vs the world frame. The notation we'll use:

```
p_world = T_world_from_camera · p_camera
```

Read right-to-left: "a point in camera coords, transformed into world coords."
Get the subscript discipline right and transforms compose like a chain.

## 2. Rotations — four representations, one rotation

Orientation is the hard part. There are several ways to write the *same*
rotation; each has a use and a failure mode. (All implemented and cross-checked
in `transforms.py`.)

### (a) Rotation matrix `R` (3×3) — SO(3)
Columns are the rotated axes. **Pros:** composes by multiplication
(`R_total = R2 @ R1`), rotates a point by `R @ p`, no singularities. **Cons:** 9
numbers for 3 DOF; drifts from orthonormal under accumulation.
- Orthonormal: `Rᵀ R = I`, so the inverse is just `Rᵀ` (free!).
- `det(R) = +1` (a pure rotation, no reflection).

### (b) Euler angles (roll, pitch, yaw) — 3 numbers
Intuitive ("nose up 10°, bank left 5°"). **Pros:** human-readable, compact.
**Cons:** **gimbal lock** — at certain orientations two axes align and you lose a
degree of freedom; interpolation is ugly. Great for *display*, dangerous for
*computation*.

### (c) Axis-angle / rotation vector — the Lie-algebra view
"Rotate by angle θ about unit axis **n**." `Rodrigues' formula` turns it into a
matrix. This is `so(3)`, the *tangent space* — where you do calculus on
rotations (integrate angular velocity, optimize poses in SLAM/bundle adjustment).

### (d) Quaternion `[w, x, y, z]` — what robots actually store
A 4-number unit vector encoding a rotation. **Pros:** no gimbal lock, compact,
numerically stable, **smooth interpolation (SLERP)**, cheap to compose and
renormalize. **Cons:** unintuitive; `q` and `−q` are the same rotation. This is
the default in ROS, game engines, IMUs, and flight controllers.

> **Rule of thumb:** compute and store in **quaternions or matrices**, *display*
> in Euler, *optimize* in axis-angle/Lie algebra. Never integrate Euler angles.

```python
from robotics.transforms import euler_to_matrix, matrix_to_quat, quat_slerp
R = euler_to_matrix(roll=0.1, pitch=0.2, yaw=0.3)   # build from intuitive angles
q = matrix_to_quat(R)                                # store/transmit as quaternion
q_mid = quat_slerp(q0, q1, 0.5)                       # smooth halfway orientation
```

## 3. Rigid transforms — SE(3) (rotation **and** translation)

A full pose = orientation + position. Pack both into a **4×4 homogeneous
transform**:

```
T = [ R  t ]      p_homogeneous = [x, y, z, 1]ᵀ
    [ 0  1 ]      p' = T · p     # rotates then translates in one matmul
```

Why 4×4? So that translation becomes a *matrix multiply* and transforms
**compose by multiplication** — the whole robot is one chain:

```
T_world_from_tool = T_world_from_base @ T_base_from_link1 @ ... @ T_linkN_from_tool
```

Key operations (all in `transforms.py`, all O(1)):
- **Compose:** `compose(A, B, C)` = `A @ B @ C`.
- **Invert:** `invert_transform(T)` uses `Rᵀ` — never call `np.linalg.inv` on a
  rigid transform.
- **Apply:** `transform_points(T, pts)` for an (N,3) cloud (used on LiDAR data!).

This is exactly what ROS's **tf2** library manages: a live tree of transforms
between every frame on the robot, queried at any timestamp.

## 4. Forward kinematics (FK) — joints → hand pose

A serial arm is a chain of links connected by joints. Multiply each link's
transform and you get the **end-effector pose** from the joint angles:

```python
from robotics.kinematics import DHChain
arm = DHChain([{"d":0,"a":1.0,"alpha":0}, {"d":0,"a":1.0,"alpha":0}])
T_ee = arm.forward([0.3, -0.5])      # 4x4 pose of the hand
```

We use **Denavit–Hartenberg (DH) parameters**, the standard way to describe a
manipulator with 4 numbers per link. FK is always solvable and unique.

## 5. Inverse kinematics (IK) — desired pose → joints

The reverse: "put the hand *here* — what joint angles do I need?" This is **hard**
because it's nonlinear: there may be **no** solution (out of reach), **one**,
**several** (elbow-up vs elbow-down), or **infinitely many** (redundant arms).

Two approaches, both in the code:
- **Closed-form** (when geometry allows): `planar_2link_ik` solves a 2-link arm
  exactly with the law of cosines — see the geometry directly.
- **Numerical (Jacobian-based)**: `DHChain.inverse` iteratively descends the
  pose error using the **Jacobian** (damped least squares). Works for any chain;
  this is what MoveIt and most real IK solvers do.

## 6. The Jacobian — the bridge between velocities (and forces)

The **Jacobian** `J(q)` maps joint velocities to end-effector velocity:

```
ẋ = J(q) · q̇          (q̇ = joint speeds → ẋ = hand twist: linear + angular)
```

It's everywhere:
- **Velocity control:** want the hand to move at velocity `ẋ`? Solve `q̇ = J⁻¹ ẋ`.
- **Numerical IK:** step joints along `Jᵀ`/`J⁻¹` to reduce pose error.
- **Statics:** joint torques from a hand force: `τ = Jᵀ · F`.
- **Singularities:** where `J` loses rank (the arm "locks up") — `det(J J ᵀ)→0`.

```python
J = arm.jacobian([0.3, -0.5])   # 6 x n_joints (numerical, in robotics/kinematics.py)
```

## 7. Where this shows up later
- **Perception (Part 6):** projecting a 3D LiDAR point into a 2D camera pixel is
  a transform `T_cam_from_lidar` + a camera projection. Sensor *calibration*
  (Part 5) is literally estimating these transforms.
- **SLAM (Part 7):** the robot's pose is an SE(3) element; optimization happens
  in the Lie algebra `se(3)`.
- **VLA (Part 10):** the actions a VLA outputs are end-effector **poses/deltas**
  in SE(3) — exactly these objects.

## 🛠️ Project
1. Build a 3-link planar `DHChain`. Pick joint angles, run FK, then feed the
   resulting position to `inverse()` and confirm you recover a valid solution.
2. Animate `quat_slerp` between two orientations and plot the axes — see why
   SLERP is the right way to interpolate rotation.
3. Transform a synthetic LiDAR cloud from sensor frame to world frame with a
   known `T` and verify with `invert_transform`.

## ✅ Check your understanding
1. Why store rotations as quaternions but display them as Euler angles?
2. What is gimbal lock, and which representation avoids it?
3. Why is `invert_transform` cheaper than a general 4×4 inverse?
4. Give two reasons IK is harder than FK.
5. Write the relation between joint torques and an end-effector force.

## 📖 Go deeper
- Lynch & Park, *Modern Robotics*, Ch. 2–5 (configuration space, rigid motions,
  FK, velocity kinematics) — the definitive, free treatment.
- "A tutorial on SE(3) transformation parameterizations" (Blanco) — for SLAM.
- 3Blue1Brown, *Quaternions* (visual intuition).

➡️ **Next:** [Part 4 — Simulation](../part4_simulation/): put this math to work
in MuJoCo, Isaac Sim/Lab, and ROS/Gazebo.
