"""Kinematics: from joint angles to where the hand is (and back).

* **Forward kinematics (FK):** given joint angles, where is the end-effector?
  A deterministic chain of transforms - always solvable.
* **Inverse kinematics (IK):** given a desired hand pose, what joint angles get
  there? Generally nonlinear, may have zero / one / many / infinite solutions.

We implement FK via Denavit-Hartenberg (DH) parameters (the classic robotics
parameterization) and a numerical IK via the Jacobian (gradient descent in pose
space) that works for any chain - plus a closed-form 2-link planar IK so you can
*see* the geometry. This is the maths under MuJoCo / Isaac arm control.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from robotics.transforms import make_transform


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Single DH link transform (standard / distal convention).

    Parameters describe how to get from link i-1 to link i:
      theta - joint angle (revolute variable), d - link offset,
      a - link length, alpha - link twist.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ], dtype=float)


class DHChain:
    """A serial manipulator defined by a list of DH links.

    Each link is a dict with keys d, a, alpha and (optionally) a constant
    theta_offset. ``theta`` is supplied per-joint at FK time (revolute joints).
    """

    def __init__(self, links: Sequence[dict]):
        self.links = list(links)

    @property
    def n_joints(self) -> int:
        return len(self.links)

    def forward(self, q: Sequence[float]) -> np.ndarray:
        """Forward kinematics -> 4x4 pose of the end-effector in the base frame."""
        T = np.eye(4)
        for link, theta in zip(self.links, q):
            T = T @ dh_transform(theta + link.get("theta_offset", 0.0),
                                 link["d"], link["a"], link["alpha"])
        return T

    def link_positions(self, q: Sequence[float]) -> np.ndarray:
        """Return (n+1, 3) positions of the base and every joint - handy for plots."""
        T = np.eye(4)
        pts = [T[:3, 3].copy()]
        for link, theta in zip(self.links, q):
            T = T @ dh_transform(theta + link.get("theta_offset", 0.0),
                                 link["d"], link["a"], link["alpha"])
            pts.append(T[:3, 3].copy())
        return np.array(pts)

    def jacobian(self, q: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        """Numerical 6xN geometric Jacobian (linear+angular) via finite differences.

        The Jacobian maps joint velocities to end-effector velocity; it is the
        workhorse of velocity control and numerical IK.
        """
        q = np.asarray(q, float)
        T0 = self.forward(q)
        p0, R0 = T0[:3, 3], T0[:3, :3]
        J = np.zeros((6, self.n_joints))
        for i in range(self.n_joints):
            dq = q.copy()
            dq[i] += eps
            Ti = self.forward(dq)
            J[:3, i] = (Ti[:3, 3] - p0) / eps            # linear part
            dR = (Ti[:3, :3] - R0) / eps
            W = dR @ R0.T                                 # angular velocity (skew)
            J[3:, i] = np.array([W[2, 1], W[0, 2], W[1, 0]])
        return J

    def inverse(self, target_pos: np.ndarray, q_init: Sequence[float] = None,
                iters: int = 200, step: float = 0.5, tol: float = 1e-4):
        """Numerical position-only IK via damped least squares on the Jacobian.

        Iteratively nudges joints to reduce end-effector position error. Returns
        ``(q, converged)``. Orientation IK adds the angular rows analogously.
        """
        q = np.zeros(self.n_joints) if q_init is None else np.asarray(q_init, float).copy()
        target = np.asarray(target_pos, float)
        for _ in range(iters):
            err = target - self.forward(q)[:3, 3]
            if np.linalg.norm(err) < tol:
                return q, True
            Jp = self.jacobian(q)[:3]                      # position rows
            # Damped least squares (Levenberg-Marquardt) for stability near singularities
            lam = 0.05
            dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + lam ** 2 * np.eye(3), err)
            q = q + step * dq
        return q, False


# --------------------------------------------------------------------------
# Closed-form 2-link planar arm - the geometry, made explicit
# --------------------------------------------------------------------------
def planar_2link_fk(l1: float, l2: float, q1: float, q2: float) -> np.ndarray:
    """End-effector (x, y) of a 2-link planar arm."""
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, y])


def planar_2link_ik(l1: float, l2: float, x: float, y: float, elbow_up: bool = True):
    """Analytic IK for a 2-link planar arm. Returns (q1, q2) or None if unreachable."""
    r2 = x * x + y * y
    cos_q2 = (r2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    if abs(cos_q2) > 1:
        return None  # target outside the reachable annulus
    q2 = np.arccos(np.clip(cos_q2, -1, 1))
    if elbow_up:
        q2 = -q2
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])
