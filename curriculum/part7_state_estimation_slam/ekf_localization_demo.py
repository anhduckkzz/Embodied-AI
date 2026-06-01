"""EKF localization vs dead-reckoning - watch sensor fusion beat drift.

Run:  python curriculum/part7_state_estimation_slam/ekf_localization_demo.py
      (saves ekf_localization_demo.png if matplotlib is available)

A differential-drive robot drives a loop. Its controls (v, omega) are noisy, so
pure **dead-reckoning** (integrating the motion model) drifts away from truth.
An **EKF** corrects that drift using noisy **range+bearing** measurements to a
few known landmarks - exactly the Probabilistic-Robotics setup.

State x = [px, py, theta];  control u = [v, omega].
Nonlinear motion -> we linearize with Jacobians (that's what makes it an EKF).
We use robotics.filters.ExtendedKalmanFilter directly.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotics.filters import ExtendedKalmanFilter

DT = 0.1
LANDMARKS = np.array([[6, 4], [2, 8], [8, 8], [1, 1]], dtype=float)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# --- motion model and its Jacobian (w.r.t. state) ---
def motion(x, u):
    v, w = u
    return np.array([x[0] + v * np.cos(x[2]) * DT,
                     x[1] + v * np.sin(x[2]) * DT,
                     wrap(x[2] + w * DT)])


def motion_jac(x, u):
    v, w = u
    return np.array([[1, 0, -v * np.sin(x[2]) * DT],
                     [0, 1, v * np.cos(x[2]) * DT],
                     [0, 0, 1]], dtype=float)


# --- measurement model for ONE landmark and its Jacobian ---
def make_measurement(lm):
    def h(x):
        dx, dy = lm[0] - x[0], lm[1] - x[1]
        return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])

    def H(x):
        dx, dy = lm[0] - x[0], lm[1] - x[1]
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        return np.array([[-dx / r, -dy / r, 0],
                         [dy / q, -dx / q, -1]], dtype=float)
    return h, H


def run():
    rng = np.random.default_rng(0)
    Q = np.diag([0.02, 0.02, 0.01]) ** 2        # process noise (motion uncertainty)
    R = np.diag([0.3, 0.05]) ** 2               # measurement noise (range, bearing)

    x_true = np.array([1.0, 1.0, 0.0])
    x_dead = x_true.copy()                       # dead-reckoning estimate
    ekf = ExtendedKalmanFilter(motion, None, motion_jac, None, Q, R,
                               x0=x_true.copy(), P0=np.diag([0.1, 0.1, 0.1]))

    u = np.array([1.0, 0.3])                     # commanded control (drive in a circle)
    # The real robot is miscalibrated: it actually goes a bit faster and turns a
    # bit more than commanded (systematic bias) - the classic cause of odometry
    # drift. The estimators only know the *commanded* u.
    bias = np.array([1.08, 1.06])
    truth, dead, est = [], [], []
    for step in range(200):
        # --- true motion: biased control + small random noise ---
        u_true = u * bias + rng.normal(0, [0.02, 0.01])
        x_true = motion(x_true, u_true)

        # --- dead-reckoning: integrate the nominal control, no correction ---
        x_dead = motion(x_dead, u)

        # --- EKF predict, then update from each landmark ---
        ekf.predict(u)
        for lm in LANDMARKS:
            h, H = make_measurement(lm)
            z = h(x_true) + rng.normal(0, [0.3, 0.05])   # noisy real measurement
            z[1] = wrap(z[1])
            ekf.h, ekf.H_jac = h, H
            # wrap the bearing innovation by predicting then correcting manually-safe
            y = z - h(ekf.x)
            y[1] = wrap(y[1])
            Hk = H(ekf.x)
            S = Hk @ ekf.P @ Hk.T + R
            K = ekf.P @ Hk.T @ np.linalg.inv(S)
            ekf.x = ekf.x + K @ y
            ekf.x[2] = wrap(ekf.x[2])
            ekf.P = (np.eye(3) - K @ Hk) @ ekf.P

        truth.append(x_true.copy()); dead.append(x_dead.copy()); est.append(ekf.x.copy())

    truth, dead, est = map(np.array, (truth, dead, est))
    dead_err = np.linalg.norm(dead[:, :2] - truth[:, :2], axis=1).mean()
    ekf_err = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1).mean()
    print("Mean position error over the trajectory:")
    print(f"   dead-reckoning : {dead_err:.3f} m")
    print(f"   EKF fusion     : {ekf_err:.3f} m   ({dead_err / max(ekf_err,1e-9):.1f}x better)")
    return truth, dead, est


def main():
    print("=== EKF localization vs dead-reckoning ===\n")
    truth, dead, est = run()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(truth[:, 0], truth[:, 1], "g-", lw=2, label="true path")
        ax.plot(dead[:, 0], dead[:, 1], "r--", label="dead-reckoning (drifts)")
        ax.plot(est[:, 0], est[:, 1], "b-", label="EKF estimate")
        ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker="*", s=200,
                   c="orange", label="landmarks")
        ax.legend(); ax.axis("equal"); ax.set_title("EKF localization")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ekf_localization_demo.png")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"\nSaved -> {out}")
    except ImportError:
        pass
    print("\nFusing noisy motion + noisy measurements beats either alone. That is the EKF.")


if __name__ == "__main__":
    main()
