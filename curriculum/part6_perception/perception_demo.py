"""The classical perception pipeline, end to end, with no GPU and no heavy deps.

Run:  python curriculum/part6_perception/perception_demo.py
      (optional) it saves a plot to perception_demo.png if matplotlib is present.

Pipeline demonstrated (all from robotics/):
  1. Simulate a 2D LiDAR scan in a scene of obstacles  -> ranges
  2. Convert the scan to a point cloud (sensor -> world frame)
  3. On a synthetic 3D cloud: RANSAC-segment the ground plane from objects
  4. Voxel-downsample a dense cloud
  5. ICP-align two overlapping scans (the core of LiDAR odometry / mapping)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from robotics.pointcloud import fit_plane_ransac, icp, voxel_downsample
from robotics.sensors import ranges_to_points, simulate_lidar_2d
from robotics.transforms import Rz


def step1_lidar_scan():
    print("1) Simulate a 2D LiDAR among three obstacles")
    pose = (0.0, 0.0, 0.0)
    obstacles = [(3.0, 0.0, 0.5), (2.0, 2.0, 0.4), (1.5, -1.5, 0.3)]
    angles, ranges = simulate_lidar_2d(pose, obstacles, n_beams=180, noise_std=0.01)
    points = ranges_to_points(pose, angles, ranges)
    print(f"   -> {len(points)} hit points; nearest return = {ranges.min():.2f} m")
    return points


def step2_ransac_ground():
    print("2) RANSAC ground-plane segmentation on a synthetic 3D cloud")
    rng = np.random.default_rng(0)
    ground = np.column_stack([rng.uniform(-5, 5, 800), rng.uniform(-5, 5, 800),
                              rng.normal(0, 0.02, 800)])           # z ~ 0 road
    car = rng.uniform(-1, 1, (120, 3)) * [1.0, 0.5, 0.7] + [3, 0, 0.8]  # a "vehicle"
    cloud = np.vstack([ground, car])
    plane, ground_mask = fit_plane_ransac(cloud, threshold=0.1)
    objects = cloud[~ground_mask]
    print(f"   -> plane normal ~ {np.round(plane[:3], 2)}; "
          f"ground={ground_mask.sum()} pts, objects={len(objects)} pts")
    return cloud, ground_mask


def step3_voxel():
    print("3) Voxel-downsample a dense cloud")
    rng = np.random.default_rng(1)
    dense = rng.random((20000, 3)) * [10, 10, 2]
    sparse = voxel_downsample(dense, voxel_size=0.5)
    print(f"   -> {len(dense)} -> {len(sparse)} points "
          f"({100 * len(sparse) / len(dense):.1f}% kept)")
    return dense, sparse


def step4_icp():
    print("4) ICP-align two overlapping scans (LiDAR odometry)")
    rng = np.random.default_rng(2)
    scan_a = rng.random((300, 3)) * [5, 5, 1]
    R_true = Rz(np.deg2rad(12))               # the robot turned 12 deg + moved
    t_true = np.array([0.4, -0.3, 0.0])
    scan_b = (R_true @ scan_a.T).T + t_true   # second scan of the same scene
    # ICP recovers the motion that maps the first scan onto the second.
    R_est, t_est, aligned = icp(scan_a, scan_b, max_iters=60)
    resid = np.mean(np.linalg.norm(aligned - scan_b, axis=1))
    ang_est = np.rad2deg(np.arctan2(R_est[1, 0], R_est[0, 0]))
    print(f"   -> recovered rotation ~ {ang_est:.1f} deg (true 12), "
          f"residual = {resid:.4f} m")
    return scan_a, scan_b, aligned


def maybe_plot(lidar_pts, cloud, ground_mask):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].scatter(lidar_pts[:, 0], lidar_pts[:, 1], s=6)
    ax[0].scatter([0], [0], c="red", marker="^", s=80, label="robot")
    ax[0].set_title("2D LiDAR scan (world frame)"); ax[0].axis("equal"); ax[0].legend()
    ax[1].scatter(cloud[ground_mask, 0], cloud[ground_mask, 1], s=4, c="green", label="ground")
    ax[1].scatter(cloud[~ground_mask, 0], cloud[~ground_mask, 1], s=8, c="red", label="objects")
    ax[1].set_title("RANSAC ground vs objects (top view)"); ax[1].legend()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perception_demo.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    return out


def main():
    print("=== Classical perception pipeline (CPU, no heavy deps) ===\n")
    lidar_pts = step1_lidar_scan()
    cloud, ground_mask = step2_ransac_ground()
    step3_voxel()
    step4_icp()
    path = maybe_plot(lidar_pts, cloud, ground_mask)
    if path:
        print(f"\nSaved visualization -> {path}")
    print("\nThis is exactly what Open3D/PCL do at scale - now you know the internals.")


if __name__ == "__main__":
    main()
