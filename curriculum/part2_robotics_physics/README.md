# Part 2 — The Physics of Robots (from zero)

> No code in this part — it's the mental model. You cannot debug a wobbling
> robot, tune a controller, or trust a simulator until you understand the
> physics it's approximating. We build it from everyday intuition.

A robot is a **physical body** that obeys Newton's laws. Simulators (MuJoCo,
Isaac) are fast, approximate solvers of exactly the equations below. Control
(Part 9) is the art of commanding forces to make that body do what you want.

---

## 1. Kinematics vs Dynamics (know the difference)

- **Kinematics** = motion *without* forces. Positions, velocities, angles. "If
  the elbow is at 30°, where is the hand?" (Part 3 is all kinematics.)
- **Dynamics** = motion *caused by* forces and torques. "How much torque does
  the elbow motor need to lift a 2 kg payload?" This part.

You need both: kinematics to know *where things are*, dynamics to know *what it
takes to move them*.

## 2. Translation: Newton's second law

For a point mass (or a body's center of mass):

```
F = m · a            force = mass × acceleration
```

Position, velocity, acceleration are linked by calculus:
`a = dv/dt`, `v = dx/dt`. A simulator **integrates** these forward in time:

```
v ← v + a·dt          (a = F/m)
x ← x + v·dt          (semi-implicit Euler — what most sims use)
```

> You already wrote this loop in [`robotics/control.py:simulate_mass_pid`](../../robotics/control.py).
> That toy *is* a physics engine for one particle.

**Momentum** `p = m·v` is conserved without external forces — the principle
behind a drone's reaction to spinning rotors and a walking robot's balance.

## 3. Rotation: the part that trips everyone up

Rotational motion mirrors translation, with rotational analogues:

| Translation | Rotation | Meaning |
|-------------|----------|---------|
| force `F` | **torque** `τ = r × F` | twisting effort |
| mass `m` | **moment of inertia** `I` | resistance to angular accel |
| accel `a` | angular accel `α` | how fast spin changes |
| `F = m a` | **`τ = I α`** | Newton's law for rotation |
| velocity `v` | angular velocity `ω` | rad/s |

- **Torque** is force applied at a lever arm: pushing a door far from the hinge
  (large `r`) needs less force. A motor's spec sheet lists its max torque.
- **Moment of inertia** `I` depends on how mass is *distributed*: mass far from
  the axis resists rotation more (why a figure skater spins faster with arms in,
  and why long robot links are hard to swing). In 3D, `I` is a **3×3 inertia
  tensor**, not a scalar.
- **Gyroscopic effects:** spinning bodies resist changes to their axis — central
  to how drones and reaction wheels stabilize.

## 4. The pendulum: the "hello world" of robot dynamics

A single rigid link swinging under gravity:

```
I · θ̈ = −m g L sin(θ) + τ_motor
```

This one equation contains the whole challenge of robotics: it's **nonlinear**
(`sin θ`), **underactuated** if `τ` is limited, and unstable at the top. Every
manipulator and legged robot is a tree of coupled pendulums. (The RL `Pendulum`
and `CartPole` envs you trained in Part 1 are literally this.)

## 5. Forces a robot actually fights

- **Gravity** — constant `m·g` down; gravity *compensation* is the first job of
  an arm controller.
- **Friction** — static (must be overcome to start moving) vs kinetic; makes
  contact possible (a wheel grips, a gripper holds) but wastes energy and is
  hard to model. Simulators approximate it (Coulomb friction cones).
- **Contact & collision** — the hardest part of simulation. When foot meets
  ground or finger meets object, forces spike instantaneously. MuJoCo's claim to
  fame is a *stable, differentiable* contact solver.
- **Damping / drag** — velocity-dependent resistance (air on a drone, fluid in a
  joint); adds stability.

## 6. Actuators: how robots make force

- **Electric motors** (most robots): torque ∝ current. Often through **gears**
  that trade speed for torque (and add backlash/friction).
- **Direct-drive / quasi-direct-drive** (modern legged robots, e.g. MIT Cheetah):
  fewer gears → "transparent," force-controllable, robust to impacts.
- **Hydraulics** (Atlas-class): huge force density, hard to control.
- **Tendons / series-elastic** (soft, safe robots): a spring between motor and
  load lets you *measure and control force* and absorb shocks.

The actuator model is **half of why sim-to-real is hard** (Part 4): a perfect
torque in sim becomes a laggy, saturated, backlash-y reality.

## 7. Energy, stability, and why it matters for learning

- **Energy** `E = kinetic + potential`. Efficient gaits minimize energy; reward
  functions in RL often penalize energy use.
- **Stability / balance:** a legged robot stays up by keeping its center of mass
  over its support polygon (static) or managing momentum (dynamic). The
  **Zero-Moment Point (ZMP)** is the classic criterion.
- **Why an RL agent needs this:** the reward, the action space (torques!), and
  the sim's fidelity all come from this physics. A policy trained on bad physics
  won't transfer.

## 8. State of a rigid body (the bridge to Part 3)

A free rigid body in 3D has **6 degrees of freedom (DOF):** 3 translation + 3
rotation. Its full state is:

```
position (3)  +  orientation (3, e.g. roll/pitch/yaw)  +  linear vel (3)  +  angular vel (3)  =  12 numbers
```

Representing that **orientation** correctly — without singularities or drift — is
so important and so error-prone that the whole next part is devoted to it.

## 🛠️ Project
On paper (or in a 30-line NumPy script), simulate a pendulum with
`I·θ̈ = −mgL·sin θ`. Integrate with Euler. Add a PID `τ_motor` (reuse
[`robotics/control.py`](../../robotics/control.py)) to swing it up and hold it
vertical. You've just built and controlled your first dynamical robot.

## ✅ Check your understanding
1. Difference between kinematics and dynamics — one sentence each.
2. What's the rotational analogue of `F = ma`, term by term?
3. Why does moment of inertia depend on mass *distribution*, not just mass?
4. Why is contact the hardest thing for a physics simulator?
5. Name two reasons a torque command in sim differs from reality.

## 📖 Go deeper
- Lynch & Park, *Modern Robotics*, Ch. 8 (dynamics) — free book + videos.
- Featherstone, *Rigid Body Dynamics Algorithms* (the math inside every sim).
- MuJoCo documentation, "Computation" chapter — how a real solver works.

➡️ **Next:** [Part 3 — 3D Math & Transforms](../part3_3d_math_transforms/):
how to represent and compose positions and orientations without losing your mind.
