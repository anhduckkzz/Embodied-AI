"""Simple simulated sensors - so you can reason about noise before real hardware.

Every robot perceives through imperfect sensors. Before plugging in a real
LiDAR or GPS it pays to *simulate* their characteristic errors and feed them to
your filters (``robotics.filters``) and perception code. These toy models
capture the essence: a 2D LiDAR that ray-casts against obstacles, a GPS with
Gaussian position noise, an IMU with bias + drift, and a landmark range-bearing
sensor (the classic SLAM measurement).
"""
from __future__ import annotations

import numpy as np


def simulate_lidar_2d(pose, obstacles, n_beams: int = 180, fov: float = 2 * np.pi,
                      max_range: float = 10.0, noise_std: float = 0.02, seed: int = 0):
    """Cast ``n_beams`` rays from a 2D pose and return per-beam ranges.

    pose = (x, y, theta); obstacles = list of (cx, cy, radius) circles. Returns
    ``(angles, ranges)``. This is exactly the 2D scan a TurtleBot or a Roomba
    sees, and the input to 2D SLAM (gmapping / Cartographer).
    """
    x, y, theta = pose
    rng = np.random.default_rng(seed)
    angles = np.linspace(-fov / 2, fov / 2, n_beams) + theta
    ranges = np.full(n_beams, max_range)
    for i, a in enumerate(angles):
        d = np.array([np.cos(a), np.sin(a)])
        best = max_range
        for cx, cy, r in obstacles:
            # Ray-circle intersection: solve |o + t d - c|^2 = r^2 for smallest t>0.
            oc = np.array([x - cx, y - cy])
            b = 2 * d @ oc
            c = oc @ oc - r * r
            disc = b * b - 4 * c
            if disc >= 0:
                t = (-b - np.sqrt(disc)) / 2
                if 0 < t < best:
                    best = t
        ranges[i] = best + rng.normal(0, noise_std) if best < max_range else max_range
    return angles, ranges


def ranges_to_points(pose, angles, ranges, max_range: float = 10.0) -> np.ndarray:
    """Convert a 2D scan (angles, ranges) into XY points in the world frame.

    The bridge from raw LiDAR to a point cloud you can register with ICP.
    """
    x, y, _ = pose
    valid = ranges < max_range
    a, r = angles[valid], ranges[valid]
    return np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)


class GPS:
    """GPS position fix with Gaussian noise (meters). Real GPS ~ 1-5 m; RTK ~ cm."""

    def __init__(self, noise_std: float = 2.0, seed: int = 0):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def measure(self, true_xy) -> np.ndarray:
        return np.asarray(true_xy, float) + self.rng.normal(0, self.noise_std, size=2)


class IMU:
    """Gyro/accel with constant bias + random walk - why integration drifts.

    Integrating a biased gyro turns a tiny error into unbounded heading drift,
    which is exactly why we fuse the IMU with absolute sensors (GPS, vision).
    """

    def __init__(self, bias: float = 0.01, noise_std: float = 0.005, seed: int = 0):
        self.bias = bias
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def measure_rate(self, true_rate: float) -> float:
        return true_rate + self.bias + self.rng.normal(0, self.noise_std)


def range_bearing(pose, landmark, range_std: float = 0.1, bearing_std: float = 0.02,
                  rng: np.random.Generator = None):
    """Noisy (range, bearing) to a landmark - the canonical SLAM measurement.

    bearing is relative to the robot's heading. EKF-SLAM linearizes exactly this
    nonlinear model (see ``robotics.filters.ExtendedKalmanFilter``).
    """
    rng = rng or np.random.default_rng(0)
    dx, dy = landmark[0] - pose[0], landmark[1] - pose[1]
    r = np.hypot(dx, dy) + rng.normal(0, range_std)
    b = (np.arctan2(dy, dx) - pose[2]) + rng.normal(0, bearing_std)
    return np.array([r, np.arctan2(np.sin(b), np.cos(b))])  # wrap bearing to [-pi,pi]
