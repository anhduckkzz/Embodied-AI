"""Point clouds: the native language of LiDAR and depth cameras.

A point cloud is just an (N, 3) array of XYZ points (optionally with intensity
or RGB). Everything a self-driving car or a robot does with LiDAR starts here:
downsample for speed, segment the ground plane, cluster objects, and register
two scans together (odometry / mapping). These are teaching-grade NumPy
implementations of what Open3D/PCL do at scale.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """Reduce density by averaging all points within each voxel cube.

    LiDAR returns millions of points; downsampling to a regular grid keeps shape
    while making downstream processing tractable. One representative per voxel.
    """
    points = np.asarray(points, float)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3))
    counts = np.zeros(inv.max() + 1)
    np.add.at(out, inv, points)
    np.add.at(counts, inv, 1)
    return out / counts[:, None]


def fit_plane_ransac(points: np.ndarray, threshold: float = 0.05,
                     iters: int = 200, seed: int = 0):
    """Segment the dominant plane (e.g. the ground/road) with RANSAC.

    Repeatedly fit a plane to 3 random points and count inliers; keep the best.
    Returns ``(plane, inlier_mask)`` where plane = (a, b, c, d) for
    ``a x + b y + c z + d = 0`` with unit normal (a, b, c). Robust to the huge
    fraction of "outlier" object points sitting above the road.
    """
    points = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    best_inliers = None
    best_plane = None
    n = len(points)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ p1
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = np.array([*normal, d])
    return best_plane, best_inliers


def nearest_neighbors(src: np.ndarray, dst: np.ndarray):
    """For each point in src, the index & distance of the closest dst point.

    O(N*M) brute force - clear, but use a KD-tree (scipy.spatial.cKDTree) for
    real clouds. This is the inner loop of ICP.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    diff = src[:, None, :] - dst[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    idx = np.argmin(d2, axis=1)
    return idx, np.sqrt(d2[np.arange(len(src)), idx])


def icp(src: np.ndarray, dst: np.ndarray, max_iters: int = 50, tol: float = 1e-6):
    """Iterative Closest Point: rigid-align src onto dst.

    Alternates (1) match each src point to its nearest dst point, (2) solve the
    optimal rotation+translation (Procrustes / Kabsch) that aligns the matches.
    Returns ``(R, t, aligned_src)``. This is the backbone of LiDAR odometry and
    scan-to-map registration in SLAM.
    """
    src = np.asarray(src, float).copy()
    dst = np.asarray(dst, float)
    R_total = np.eye(3)
    t_total = np.zeros(3)
    prev_err = np.inf
    for _ in range(max_iters):
        idx, dists = nearest_neighbors(src, dst)
        matched = dst[idx]
        # Kabsch: best rigid transform between src and matched.
        mu_s, mu_d = src.mean(0), matched.mean(0)
        H = (src - mu_s).T @ (matched - mu_d)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:        # reflection fix
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = mu_d - R @ mu_s
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        err = dists.mean()
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return R_total, t_total, src


def passthrough_filter(points: np.ndarray, axis: int = 2,
                       lo: float = -np.inf, hi: float = np.inf) -> np.ndarray:
    """Keep points whose coordinate along ``axis`` is within [lo, hi].

    The simplest, most-used crop: e.g. drop everything below the sensor or
    beyond max range before heavier processing.
    """
    points = np.asarray(points, float)
    mask = (points[:, axis] >= lo) & (points[:, axis] <= hi)
    return points[mask]
