# Part 14 — Control Theory: tracking what the planner commands

> Code: [`robotics/control.py`](../../robotics/control.py) (PID, LQR), demo:
> [`control_demo.py`](control_demo.py). Planning (Part 8) says *where* to go;
> **control** makes the real, dynamic, noisy robot actually *get there*. This is
> the classical-engineering counterpart to RL — and the two increasingly blend.

## 1. The control problem

A controller computes actuator commands `u` to make the system state `x` track a
desired reference, despite disturbances and model error. The system evolves as
`ẋ = f(x, u)` (continuous) or `x_{k+1} = f(x_k, u_k)` (discrete). **Feedback** —
measuring `x` and reacting — is what makes control robust.

```
reference ─►(error)─► CONTROLLER ─► u ─► PLANT (robot dynamics) ─► x ─┐
                ▲                                                     │
                └───────────────── sensor measurement ───────────────┘
```

## 2. PID — the workhorse (you already have it)

`u = Kp·e + Ki·∫e + Kd·ė` (error `e = reference − measurement`). P pushes toward
the target, I removes steady-state offset, D damps oscillation. Tunes 3 numbers,
needs no model, runs 90% of industry. Limits: single-input/single-output,
manual tuning, no notion of optimality or constraints. (See
[`robotics/control.py:PID`](../../robotics/control.py); the
[`control_demo.py`](control_demo.py) balances an inverted pendulum with it.)

## 3. State-space & stability (the language of modern control)

- **State-space model:** `ẋ = Ax + Bu`, `y = Cx`. Multi-input/multi-output, the
  basis for everything below. (Nonlinear systems are *linearized* about an
  operating point to get `A, B` — exactly what the demo does for the pendulum.)
- **Stability:** does the system return to equilibrium? For discrete `x←(A−BK)x`,
  stable iff all eigenvalues lie **inside the unit circle**. **Lyapunov** theory
  proves stability via an "energy" function that only decreases — the rigorous
  tool, and the bridge to provably-safe learned control.
- **Controllability / observability:** can you *steer* every state with `u`? Can
  you *infer* every state from `y`? (Observability is why you need estimation,
  Part 7.)

## 4. LQR — optimal linear control (runnable)

For a linear system with a **quadratic cost** `J = Σ xᵀQx + uᵀRu`, the optimal
controller is **linear state feedback** `u = −Kx`, where `K` comes from solving
the **Riccati equation**. You *specify what you care about* (Q penalizes state
error, R penalizes effort) and get the **optimal** multi-state gains — no manual
tuning.

```python
from robotics.control import lqr_gain
K = lqr_gain(A, B, Q, R)     # optimal gain; control u = -K x
```

In [`control_demo.py`](control_demo.py), LQR balances the pendulum optimally; PID
also works but you *tuned* it, while LQR *derived* the gains from your cost. LQR is
the principled step up from PID and the core of MPC.

## 5. MPC — Model Predictive Control (the modern powerhouse)

At each step, **optimize a sequence of controls over a finite horizon** against a
model, subject to **constraints** (actuator limits, obstacles, comfort), apply the
first control, then **re-optimize next step** (receding horizon).
- Handles constraints and nonlinearities LQR can't (LQR ≈ unconstrained,
  infinite-horizon MPC of a linear system).
- Needs a model and solves an optimization *online* (real-time compute is the
  cost). Dominant in self-driving trajectory control, legged locomotion,
  drone agility, process control.
- **iLQR / DDP** — iterative LQR for *nonlinear* trajectory optimization (linearize
  around the current trajectory, solve an LQR sub-problem, repeat); the backbone
  of model-based control and trajectory optimization (links to Module 10).

## 6. Trajectory optimization
Solve for an entire optimal `x(t), u(t)` offline: **direct collocation** /
**shooting** methods (CasADi, Drake, OCS2). Used to generate dynamic motions
(jumps, gaits) that are then tracked by MPC/feedback. The optimization mindset of
Part 0 applied to motion.

## 7. Classical control vs RL (when to use which, and the blend)
| | **Classical (PID/LQR/MPC)** | **RL (Parts 1, 5)** |
|---|---|---|
| Needs a model? | yes (except PID) | no (model-free) |
| Guarantees | stability/optimality proofs | empirical |
| Constraints | explicit (MPC) | via reward/penalties |
| Nonlinear/contact | hard | learns it |
| Sample cost | none (analytic) | high |

**They blend:** RL policies trained in sim then deployed with a safety controller
(Module 15); MPC with *learned* dynamics (Module 10); residual RL on top of a
classical controller; LQR/MPC as the low level under an RL/VLA high level.

## 🛠️ Project
1. Run [`control_demo.py`](control_demo.py): compare LQR vs PID balancing the
   pendulum; change `Q`/`R` and watch the LQR gains and behavior change
   (aggressive vs gentle).
2. Implement **LQR on cartpole** (linearize the 4-state dynamics) and balance the
   real Gymnasium `CartPole`/MuJoCo `InvertedPendulum`. Compare to your Part-1
   PPO/DQN policy — classical vs learned on the same task.
3. (Stretch) Implement **iLQR** on `Pendulum` swing-up; then a simple **MPC**.

## ✅ Check your understanding
1. What do P, I, and D each contribute, and what can't PID do?
2. What does LQR optimize, and what do Q and R control?
3. How is MPC related to LQR, and what does it add?
4. State the discrete-time stability condition on the closed-loop matrix.
5. Give two concrete ways classical control and RL are combined in practice.

## 📖 Go deeper
- Lynch & Park, *Modern Robotics*, Ch. 11 (control). Åström & Murray, *Feedback
  Systems* (free). Underactuated Robotics (Russ Tedrake, MIT, free course — LQR,
  iLQR, trajectory opt, the best resource here).
- Borrelli/Bemporad, *Predictive Control* (MPC).

➡️ **Next:** [Part 15 — 3D Vision](../part15_3d_vision/): the geometry and neural
methods that turn images into 3D scenes.
