"""Manipulation, runnable: a planar arm reaches targets via IK, then 'executes'
the joint trajectory with per-joint PID control.

Run:  python curriculum/part9_manipulation_teleop/manipulation_demo.py
      (saves manipulation_demo.png if matplotlib is available)

Ties together Part 3 (kinematics/IK) and Part 9 (control):
  1. Define a 3-link planar arm (DH).
  2. For a sequence of Cartesian targets, solve INVERSE KINEMATICS for joints.
  3. Interpolate a smooth joint trajectory and 'execute' it with PID per joint.
This is a miniature of the perceive -> IK -> control manipulation pipeline.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotics.control import PID
from robotics.kinematics import DHChain


def make_arm():
    # 3-link planar arm (all in the XY plane): reach ~ 2.3 units.
    return DHChain([{"d": 0, "a": 1.0, "alpha": 0},
                    {"d": 0, "a": 0.8, "alpha": 0},
                    {"d": 0, "a": 0.5, "alpha": 0}])


def solve_targets(arm, targets):
    print("1) Inverse kinematics for each Cartesian target")
    configs, q = [], np.array([0.3, 0.3, 0.3])
    for i, (tx, ty) in enumerate(targets):
        q, ok = arm.inverse([tx, ty, 0.0], q_init=q, iters=300)
        reached = arm.forward(q)[:2, 3]
        err = np.linalg.norm(reached - [tx, ty])
        print(f"   target {i} ({tx:+.1f},{ty:+.1f}) -> IK {'ok' if ok else 'approx'}, "
              f"reached ({reached[0]:+.2f},{reached[1]:+.2f}), err={err:.3f}")
        configs.append(q.copy())
    return configs


def execute_pid(q_start, q_goal, steps=200, dt=0.02):
    """'Execute' a joint move with one PID per joint (position control)."""
    pids = [PID(kp=12.0, ki=0.5, kd=3.0, setpoint=g, output_limits=(-8, 8))
            for g in q_goal]
    q = np.array(q_start, float)
    qd = np.zeros_like(q)
    traj = [q.copy()]
    for _ in range(steps):
        for j in range(len(q)):
            tau = pids[j](q[j], dt)        # torque-ish command
            qd[j] += tau * dt              # toy joint dynamics (unit inertia)
            qd[j] *= 0.9                    # damping
            q[j] += qd[j] * dt
        traj.append(q.copy())
    return np.array(traj)


def main():
    print("=== Manipulation: IK + PID execution on a 3-link planar arm ===\n")
    arm = make_arm()
    targets = [(1.8, 0.5), (1.0, 1.5), (-0.5, 1.6), (0.5, -1.2)]
    configs = solve_targets(arm, targets)

    print("2) Execute the joint trajectory through all targets with PID")
    full_traj, q = [], configs[0]
    for goal in configs[1:]:
        seg = execute_pid(q, goal)
        full_traj.append(seg)
        q = seg[-1]
    full_traj = np.vstack(full_traj)
    print(f"   executed {len(full_traj)} control steps through {len(targets)} targets")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        # draw the end-effector path traced during execution
        ee = np.array([arm.forward(qq)[:2, 3] for qq in full_traj])
        ax.plot(ee[:, 0], ee[:, 1], "b-", alpha=0.6, label="end-effector path")
        # draw the arm at each target config
        for q in configs:
            pts = arm.link_positions(q)[:, :2]
            ax.plot(pts[:, 0], pts[:, 1], "-o", lw=2, alpha=0.7)
        tx = np.array(targets)
        ax.scatter(tx[:, 0], tx[:, 1], marker="x", c="red", s=90, label="targets")
        ax.set_aspect("equal"); ax.legend(); ax.set_title("3-link arm: IK reach + PID execute")
        ax.grid(alpha=0.3)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manipulation_demo.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nSaved -> {out}")
    except ImportError:
        pass
    print("\nperceive -> IK -> control: the manipulation loop, in 80 lines.")


if __name__ == "__main__":
    main()
