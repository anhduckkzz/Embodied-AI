"""Feedback control: making the robot actually track a target.

Perception tells you *where you are*; planning tells you *where to go*; control
is *how you get there* given real dynamics, noise, and disturbances. The PID
controller is the 90%-of-industry workhorse and the right first thing to
understand before LQR / MPC / learned control.

A controller computes an actuator command from the error between desired and
measured state. PID combines three terms:

* **P** (proportional) - push harder the further you are from the target.
* **I** (integral)      - accumulate steady-state error to eliminate offset.
* **D** (derivative)    - damp; react to the rate of change (anticipation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PID:
    """A scalar (or vectorized) PID controller with anti-windup and output limits."""

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.0
    output_limits: tuple = (-np.inf, np.inf)
    integral_limit: float = np.inf      # anti-windup clamp on the integral term
    _integral: float = field(default=0.0, init=False)
    _prev_error: float = field(default=None, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def __call__(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        # Integral with anti-windup.
        self._integral = np.clip(self._integral + error * dt,
                                 -self.integral_limit, self.integral_limit)
        # Derivative on error (use derivative-on-measurement to avoid setpoint kick).
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        u = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(u, *self.output_limits))


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
             iters: int = 1000, tol: float = 1e-9) -> np.ndarray:
    """Infinite-horizon discrete LQR gain K for the system x_{k+1}=A x + B u.

    LQR is the *optimal* controller for a linear system under a quadratic cost
    ``sum x'Q x + u'R u`` - the principled step up from hand-tuned PID, and the
    backbone of MPC (which re-solves a finite-horizon version each step).

    Solves the discrete algebraic Riccati equation by iteration, then returns
    the state-feedback gain so the optimal control is ``u = -K x``.
    """
    A, B = np.atleast_2d(A).astype(float), np.atleast_2d(B).astype(float)
    Q, R = np.atleast_2d(Q).astype(float), np.atleast_2d(R).astype(float)
    P = Q.copy()
    for _ in range(iters):
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)        # (R+B'PB)^-1 B'PA
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    BtP = B.T @ P
    return np.linalg.solve(R + BtP @ B, BtP @ A)


def simulate_mass_pid(target: float = 1.0, kp: float = 20.0, ki: float = 5.0,
                      kd: float = 8.0, mass: float = 1.0, dt: float = 0.02,
                      steps: int = 500, disturbance: float = 0.0):
    """Toy 1-D point-mass tracking a setpoint under a PID force command.

    Returns (time, position) arrays - plot them to *see* P vs PD vs PID behavior
    (overshoot, steady-state offset, settling). Great first control experiment.
    """
    pid = PID(kp, ki, kd, setpoint=target, output_limits=(-50, 50))
    x, v = 0.0, 0.0
    ts, xs = [], []
    for i in range(steps):
        f = pid(x, dt) + disturbance
        a = f / mass
        v += a * dt
        x += v * dt
        ts.append(i * dt)
        xs.append(x)
    return np.array(ts), np.array(xs)
