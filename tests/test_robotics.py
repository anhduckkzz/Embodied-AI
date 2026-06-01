"""Sanity tests for the robotics.* toolkit (pure NumPy, no heavy deps)."""
import numpy as np
import pytest

from robotics import transforms as tf
from robotics.kinematics import DHChain, planar_2link_fk, planar_2link_ik
from robotics.control import PID, simulate_mass_pid
from robotics.filters import KalmanFilter, ComplementaryFilter
from robotics.planning import astar, dijkstra, RRT
from robotics.pointcloud import voxel_downsample, fit_plane_ransac, icp
from robotics import sensors


# ----------------------------------------------------------- transforms ----
def test_rotation_roundtrips():
    R = tf.euler_to_matrix(0.3, -0.5, 1.1)
    # Orthonormal with det 1.
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    # Euler -> matrix -> euler.
    r, p, y = tf.matrix_to_euler(R)
    assert np.allclose(tf.euler_to_matrix(r, p, y), R, atol=1e-9)
    # Matrix <-> quaternion.
    q = tf.matrix_to_quat(R)
    assert np.allclose(tf.quat_to_matrix(q), R, atol=1e-9)
    # Axis-angle roundtrip.
    axis, ang = tf.matrix_to_axis_angle(R)
    assert np.allclose(tf.axis_angle_to_matrix(axis, ang), R, atol=1e-8)


def test_transform_inverse_and_apply():
    T = tf.make_transform(tf.Rz(0.7), [1, 2, 3])
    p = np.array([0.5, -1.0, 2.0])
    p_w = tf.transform_points(T, p)[0]
    p_back = tf.transform_points(tf.invert_transform(T), p_w)[0]
    assert np.allclose(p_back, p, atol=1e-9)


def test_slerp_endpoints():
    q0 = tf.matrix_to_quat(tf.Rz(0.0))
    q1 = tf.matrix_to_quat(tf.Rz(1.0))
    assert np.allclose(tf.quat_slerp(q0, q1, 0.0), q0, atol=1e-9)
    mid = tf.quat_slerp(q0, q1, 0.5)
    assert np.allclose(tf.quat_to_matrix(mid), tf.Rz(0.5), atol=1e-6)


# ----------------------------------------------------------- kinematics ----
def test_planar_2link_fk_ik_consistency():
    l1, l2 = 1.0, 0.8
    q = planar_2link_ik(l1, l2, 1.2, 0.5)
    assert q is not None
    xy = planar_2link_fk(l1, l2, q[0], q[1])
    assert np.allclose(xy, [1.2, 0.5], atol=1e-9)


def test_dhchain_numeric_ik_reaches_target():
    # 3-link planar arm via DH (all alpha=0, in the XY plane).
    chain = DHChain([{"d": 0, "a": 1.0, "alpha": 0},
                     {"d": 0, "a": 1.0, "alpha": 0},
                     {"d": 0, "a": 0.5, "alpha": 0}])
    target = chain.forward([0.3, -0.4, 0.2])[:3, 3]
    q, ok = chain.inverse(target, q_init=[0, 0, 0])
    assert ok
    assert np.allclose(chain.forward(q)[:3, 3], target, atol=1e-3)


# -------------------------------------------------------------- control ----
def test_pid_converges_to_setpoint():
    _, xs = simulate_mass_pid(target=1.0, steps=1000)
    assert abs(xs[-1] - 1.0) < 0.02  # settles at the setpoint


# -------------------------------------------------------------- filters ----
def test_kalman_tracks_constant_velocity():
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    kf = KalmanFilter(F, H, Q=np.eye(2) * 1e-4, R=[[0.5]], x0=[0, 0], P0=np.eye(2))
    rng = np.random.default_rng(0)
    true_x, true_v = 0.0, 1.0
    errs = []
    for _ in range(100):
        true_x += true_v * dt
        kf.predict()
        kf.update(true_x + rng.normal(0, 0.5))
        errs.append(abs(kf.x[0] - true_x))
    assert np.mean(errs[-20:]) < 0.3       # estimate stays close to truth
    assert abs(kf.x[1] - true_v) < 0.3     # velocity recovered


def test_complementary_filter_reduces_drift():
    cf = ComplementaryFilter(alpha=0.98)
    angle = cf.update(gyro_rate=0.0, accel_angle=0.5, dt=0.01)
    assert 0.0 < angle <= 0.5


# ------------------------------------------------------------- planning ----
def test_astar_finds_path_around_wall():
    grid = np.zeros((10, 10))
    grid[3:8, 5] = 1  # vertical wall with a gap at the top
    path = astar(grid, (0, 0), (9, 9))
    assert path is not None and path[0] == (0, 0) and path[-1] == (9, 9)
    # No path cell is an obstacle.
    assert all(grid[r, c] == 0 for r, c in path)


def test_astar_no_path_when_blocked():
    grid = np.zeros((5, 5))
    grid[:, 2] = 1  # full wall
    assert astar(grid, (0, 0), (0, 4)) is None


def test_rrt_reaches_goal():
    rrt = RRT(bounds=((0, 10), (0, 10)), obstacles=[(5, 5, 1.5)], step=0.5, seed=1)
    path = rrt.plan((1, 1), (9, 9), max_iters=5000)
    assert path is not None
    assert np.linalg.norm(np.asarray(path[-1]) - np.array([9, 9])) < 1.0


# ----------------------------------------------------------- pointcloud ----
def test_voxel_downsample_reduces_count():
    pts = np.random.default_rng(0).random((1000, 3))
    ds = voxel_downsample(pts, voxel_size=0.25)
    assert len(ds) < len(pts) and ds.shape[1] == 3


def test_ransac_finds_ground_plane():
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-5, 5, 500), rng.uniform(-5, 5, 500),
                              rng.normal(0, 0.01, 500)])           # z ~ 0 plane
    objects = rng.uniform(-5, 5, (50, 3)) + np.array([0, 0, 3])    # floating points
    pts = np.vstack([ground, objects])
    plane, inliers = fit_plane_ransac(pts, threshold=0.1)
    assert inliers.sum() >= 480                                    # most ground found
    assert abs(abs(plane[2]) - 1.0) < 0.05                         # normal ~ +/-z


def test_icp_recovers_known_transform():
    rng = np.random.default_rng(0)
    dst = rng.random((100, 3))
    R = tf.Rz(0.2)
    t = np.array([0.3, -0.2, 0.1])
    src = (R.T @ (dst - t).T).T   # src such that R@src + t = dst
    R_est, t_est, aligned = icp(src, dst, max_iters=60)
    assert np.mean(np.linalg.norm(aligned - dst, axis=1)) < 0.05


# ------------------------------------------------------------- sensors ----
def test_lidar_detects_obstacle_ahead():
    angles, ranges = sensors.simulate_lidar_2d((0, 0, 0), [(3, 0, 0.5)],
                                               n_beams=181, noise_std=0.0)
    # The forward beam (angle 0) should hit the circle at ~2.5 m (3 - radius).
    forward = ranges[np.argmin(np.abs(angles))]
    assert abs(forward - 2.5) < 0.1


def test_range_bearing_zero_noise():
    m = sensors.range_bearing((0, 0, 0), (1, 1), range_std=0, bearing_std=0)
    assert np.isclose(m[0], np.sqrt(2))
    assert np.isclose(m[1], np.pi / 4)
