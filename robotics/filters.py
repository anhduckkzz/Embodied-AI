"""State estimation: knowing where you are from noisy sensors.

No real sensor is exact and no real actuator is perfect. The Bayes filter — and
its Gaussian incarnations, the **Kalman filter** (KF) and **Extended Kalman
filter** (EKF) — fuse a motion *prediction* with noisy *measurements* to track a
belief (mean + covariance) over the robot's state. This is the mathematical
core of GPS/IMU fusion, wheel odometry, visual-inertial odometry, and SLAM.

The two-step loop, forever:
  1. **Predict**  - push the belief through the motion model (uncertainty grows).
  2. **Update**   - correct with a measurement (uncertainty shrinks).
"""
from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Linear Kalman filter for x_{k+1} = F x_k + B u_k + w,  z_k = H x_k + v.

    The optimal estimator when dynamics/measurements are linear and noise is
    Gaussian. ``Q`` is process-noise covariance, ``R`` measurement-noise.
    """

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F = np.atleast_2d(F).astype(float)
        self.H = np.atleast_2d(H).astype(float)
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)
        self.B = None if B is None else np.atleast_2d(B).astype(float)
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.atleast_2d(P0).astype(float)

    def predict(self, u=None):
        """Time update: propagate mean and covariance through the motion model."""
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.atleast_1d(u)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """Measurement update: correct the belief using observation z."""
        z = np.atleast_1d(np.asarray(z, float))
        y = z - self.H @ self.x                       # innovation (surprise)
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x


class ExtendedKalmanFilter:
    """EKF for *nonlinear* models, via first-order (Jacobian) linearization.

    You provide the nonlinear functions ``f(x,u)`` and ``h(x)`` plus their
    Jacobians ``F_jac(x,u)`` and ``H_jac(x)``. This is what real robots use,
    because motion (rotations!) and sensors (range/bearing, cameras) are
    nonlinear. The KF is just the special case where f, h are already linear.
    """

    def __init__(self, f, h, F_jac, H_jac, Q, R, x0, P0):
        self.f, self.h = f, h
        self.F_jac, self.H_jac = F_jac, H_jac
        self.Q = np.atleast_2d(Q).astype(float)
        self.R = np.atleast_2d(R).astype(float)
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.atleast_2d(P0).astype(float)

    def predict(self, u=None):
        F = np.atleast_2d(self.F_jac(self.x, u))
        self.x = np.asarray(self.f(self.x, u), float).reshape(-1)
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def update(self, z):
        z = np.atleast_1d(np.asarray(z, float))
        H = np.atleast_2d(self.H_jac(self.x))
        y = z - np.atleast_1d(np.asarray(self.h(self.x), float))
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
        return self.x


class ComplementaryFilter:
    """Cheap, beloved IMU attitude filter: blend gyro (fast) and accel (stable).

    angle = alpha * (angle + gyro*dt) + (1 - alpha) * accel_angle.
    The gyro is accurate short-term but drifts; the accelerometer is noisy but
    drift-free. The complementary filter trusts the gyro at high frequency and
    the accelerometer at low frequency. A drone's flight controller does this
    thousands of times per second.
    """

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.angle = 0.0

    def update(self, gyro_rate: float, accel_angle: float, dt: float) -> float:
        self.angle = self.alpha * (self.angle + gyro_rate * dt) + (1 - self.alpha) * accel_angle
        return self.angle
