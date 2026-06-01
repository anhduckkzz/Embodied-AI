"""robotics: a from-scratch, NumPy-only toolkit for the Embodied-AI curriculum.

Like the ``rl`` package, this is written to be *read*. Every function maps to a
concept in the curriculum (rigid-body transforms, kinematics, filtering,
planning, point clouds) and avoids heavy dependencies so it runs on a laptop and
on Colab alike. For production use real libraries (Open3D, PyTorch3D, ROS tf2,
Pinocchio); for *understanding*, read these.

Modules
-------
- ``robotics.transforms``  : rotations & rigid-body transforms (SO(3), SE(3))
- ``robotics.kinematics``  : forward/inverse kinematics, Jacobians
- ``robotics.control``     : PID and friends
- ``robotics.filters``     : Kalman filter, EKF, complementary filter
- ``robotics.pointcloud``  : voxel grids, RANSAC planes, a tiny ICP
- ``robotics.planning``    : Dijkstra, A*, RRT
- ``robotics.sensors``     : simple simulated range/GPS/IMU sensor models
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
