"""Rigid-body transforms: the language every robot speaks.

A robot is a tree of coordinate frames (world → base → shoulder → ... → camera).
To fuse a LiDAR point with a camera pixel, or to command a gripper pose, you are
*always* converting between frames. This module implements the core objects:

* **Rotations** in SO(3): rotation matrices, quaternions, Euler angles,
  axis-angle, and the conversions between them.
* **Rigid transforms** in SE(3): 4x4 homogeneous matrices that combine a
  rotation and a translation, and how to compose / invert / apply them.

Conventions (state them or suffer):
* Right-handed frames, column vectors, ``p_world = T_world_from_body @ p_body``.
* Quaternions are ``[w, x, y, z]`` (scalar-first), unit norm.
* Euler angles here are intrinsic XYZ (roll, pitch, yaw) in radians.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Elementary rotations
# --------------------------------------------------------------------------
def Rx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic XYZ Euler angles (rad) -> rotation matrix R = Rz·Ry·Rx."""
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def matrix_to_euler(R: np.ndarray):
    """Rotation matrix -> (roll, pitch, yaw), intrinsic XYZ. Inverse of above."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    if np.isclose(np.cos(pitch), 0.0):  # gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


# --------------------------------------------------------------------------
# Axis-angle (the Lie-algebra view: so(3) -> SO(3))
# --------------------------------------------------------------------------
def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]_x so that [v]_x @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula: rotate by ``angle`` about unit ``axis``."""
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def matrix_to_axis_angle(R: np.ndarray):
    """Rotation matrix -> (unit axis, angle). Inverse of Rodrigues."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(angle)), angle


# --------------------------------------------------------------------------
# Quaternions  [w, x, y, z]  (the representation robots actually store)
# --------------------------------------------------------------------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, float) / (np.linalg.norm(q) + 1e-12)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product: the rotation q1 followed by q2 -> q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion [w,x,y,z] (Shepperd's stable method)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation - smooth rotation between two orientations.

    Used everywhere from animation to trajectory generation for end-effectors.
    """
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    dot = np.dot(q0, q1)
    if dot < 0:           # take the shorter arc
        q1, dot = -q1, -dot
    if dot > 0.9995:      # nearly identical -> linear interp + renormalize
        return quat_normalize(q0 + t * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


# --------------------------------------------------------------------------
# SE(3): homogeneous 4x4 transforms
# --------------------------------------------------------------------------
def make_transform(R: np.ndarray = None, t: np.ndarray = None) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from rotation R and translation t."""
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, float).ravel()
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform - cheaper & more stable than np.linalg.inv.

    T^{-1} = [[R^T, -R^T t], [0, 1]] because R is orthonormal.
    """
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N,3) array of points: p' = R p + t."""
    pts = np.atleast_2d(np.asarray(pts, float))
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Chain transforms left-to-right: compose(A, B, C) = A @ B @ C.

    Reads like a kinematic chain: T_world_from_tool = compose(T_w_b, T_b_a, ...).
    """
    out = np.eye(4)
    for T in transforms:
        out = out @ T
    return out
