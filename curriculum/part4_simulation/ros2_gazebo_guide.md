# ROS 2 + Gazebo + RViz setup 💻 (the real robot software stack)

**ROS 2** is the middleware almost every research/industrial robot runs.
**Gazebo** simulates physics + sensors and plugs into ROS. **RViz** ("Viz") is
the 3D visualizer where you *see* sensor data, transforms, and plans. Learning
this is essential for real robotics and for understanding autonomy stacks.

> Best on **Ubuntu** (native or **WSL2** on Windows). Your 4060 is plenty.
> macOS users: use Docker or a VM.

## Install (Ubuntu 24.04 → ROS 2 Jazzy; 22.04 → Humble)
```bash
# Follow docs.ros.org for the exact apt steps, then:
sudo apt install ros-jazzy-desktop          # ROS 2 + RViz + demos
sudo apt install ros-jazzy-ros-gz           # Gazebo + ROS bridge
source /opt/ros/jazzy/setup.bash            # do this in every shell
```

## Core concepts (the 20% you use constantly)
- **Node** — a process that does one thing (a camera driver, a planner).
- **Topic** — a named stream nodes publish/subscribe to (e.g. `/scan`,
  `/camera/image`, `/cmd_vel`). Pub/sub = how data flows.
- **Message** — typed data on a topic (`LaserScan`, `Image`, `PointCloud2`, `Twist`).
- **Service / Action** — request/response and long-running goals (e.g. "navigate
  to X").
- **tf2** — the live tree of coordinate transforms between every frame (Part 3!).
- **URDF** — XML robot description (links, joints, meshes) — like MJCF for ROS.

## First commands
```bash
ros2 run demo_nodes_cpp talker            # publisher
ros2 run demo_nodes_py listener           # subscriber (other terminal)
ros2 topic list                           # what streams exist
ros2 topic echo /chatter                  # print messages on a topic
rviz2                                     # open the 3D visualizer
```

## Simulate a robot with sensors, view in RViz
```bash
# Example: a TurtleBot-style robot with a LiDAR in Gazebo, visualized in RViz.
ros2 launch ros_gz_sim_demos ...          # (pick a demo from ros_gz_sim_demos)
rviz2                                      # add displays: RobotModel, LaserScan,
                                           #   PointCloud2, TF, Camera
```
In RViz you'll *see* the laser scan as points around the robot, the TF frames as
little axes, and (after Part 8) the planned path and costmap. **When perception
is wrong, RViz is where you debug it.**

## Where ROS meets the rest of this curriculum
- **Nav2** — ROS 2's navigation stack (costmaps + planners from Part 8).
- **MoveIt 2** — motion planning / IK for arms (Part 9).
- **slam_toolbox / cartographer** — SLAM (Part 7) producing a live map.
- **ros2_control** — controllers (Part 9) talking to real/sim hardware.

## Tips
- Always `source` the setup file; "command not found" usually means you didn't.
- Use `rqt_graph` to visualize which nodes talk to which topics.
- `ros2 bag record` saves real sensor streams to replay offline — gold for
  developing perception without the robot.
