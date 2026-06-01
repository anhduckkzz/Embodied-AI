"""Control comparison: LQR vs PID stabilizing an inverted pendulum.

Run:  python curriculum/part14_control_theory/control_demo.py
      (saves control_demo.png if matplotlib is available)

We balance a pendulum UPRIGHT (an unstable equilibrium). We design an optimal
LQR controller on the *linearized* dynamics (robotics.control.lqr_gain), then
apply it to the *nonlinear* pendulum - and compare against a hand-tuned PID.
Shows the jump from "tune three numbers" (PID) to "specify a cost, get the
optimal multi-state feedback" (LQR).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotics.control import PID, lqr_gain

G, L, M = 9.81, 1.0, 1.0      # gravity, pole length, mass
DT = 0.02


def nonlinear_step(theta, omega, u):
    """True pendulum dynamics (theta measured from UPRIGHT): unstable at 0."""
    theta_ddot = (G / L) * np.sin(theta) + u / (M * L * L)
    omega = omega + theta_ddot * DT
    theta = theta + omega * DT
    return theta, omega


def design_lqr():
    # Linearize about theta=0 (upright): sin(theta) ~ theta.
    Ac = np.array([[0.0, 1.0], [G / L, 0.0]])
    Bc = np.array([[0.0], [1.0 / (M * L * L)]])
    A = np.eye(2) + Ac * DT            # forward-Euler discretization
    B = Bc * DT
    Q = np.diag([10.0, 1.0])           # penalize angle error most
    R = np.array([[0.1]])              # mild control penalty
    return lqr_gain(A, B, Q, R)


def simulate(controller, steps=400, theta0=0.4):
    theta, omega = theta0, 0.0
    traj = []
    for _ in range(steps):
        u = controller(theta, omega)
        theta, omega = nonlinear_step(theta, omega, u)
        traj.append(theta)
    return np.array(traj)


def main():
    print("=== LQR vs PID: balancing an inverted pendulum upright ===\n")
    K = design_lqr()
    print(f"LQR gain K = {K.ravel()}  (optimal full-state feedback u = -K x)")

    lqr_ctrl = lambda th, om: -float((K @ np.array([th, om])).item())
    # PID on the angle: u = kp*(0-theta) + kd*d(0-theta)/dt  ->  -kp*theta - kd*omega,
    # which is stabilizing when kp > g/L. (Negative feedback toward upright.)
    pid = PID(kp=40.0, ki=0.0, kd=8.0, setpoint=0.0, output_limits=(-100, 100))
    pid_ctrl = lambda th, om: pid(th, DT)

    traj_lqr = simulate(lqr_ctrl)
    traj_pid = simulate(pid_ctrl)
    print(f"final |theta|  LQR: {abs(traj_lqr[-1]):.4f} rad   "
          f"PID: {abs(traj_pid[-1]):.4f} rad   (0 = perfectly upright)")
    assert abs(traj_lqr[-1]) < 0.05, "LQR should stabilize the pendulum"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(traj_lqr)) * DT
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, np.rad2deg(traj_lqr), label="LQR (optimal)")
        ax.plot(t, np.rad2deg(traj_pid), "--", label="PID (hand-tuned)")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("time (s)"); ax.set_ylabel("pole angle from upright (deg)")
        ax.set_title("Stabilizing an inverted pendulum"); ax.legend(); ax.grid(alpha=0.3)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_demo.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nSaved -> {out}")
    except ImportError:
        pass
    print("\nPID = 3 tuned gains on one signal. LQR = specify a cost, get optimal "
          "feedback over ALL states. MPC = re-solve a constrained LQR each step.")


if __name__ == "__main__":
    main()
