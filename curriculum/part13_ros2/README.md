# Part 13 — ROS 2: the software framework real robots run on

> Setup guide: [`../part4_simulation/ros2_gazebo_guide.md`](../part4_simulation/ros2_gazebo_guide.md).
> Example node: [`minimal_nodes.py`](minimal_nodes.py).
> If MuJoCo/Isaac are the *physics*, **ROS 2** is the *nervous system* — the
> middleware that connects perception, estimation, planning, and control into one
> running robot. Almost every research and industrial robot uses it. This is a
> from-zero tour of the concepts, the code, and the ecosystem.

## 1. Why a robot needs a framework

A robot is **many programs** running at once: camera driver, LiDAR driver, a
detector, a localizer, a planner, motor controllers, a UI. They run at different
rates, maybe on different computers, and must exchange data reliably with correct
timing and coordinate frames. Writing that plumbing yourself, per robot, is
madness. **ROS 2 standardizes it**: a common way to pass messages, manage
parameters, describe robots, and reuse a huge library of existing components
(SLAM, nav, manipulation, drivers).

> ROS 2 (not ROS 1) is the current standard: real-time-friendly, multi-robot,
> secure, built on **DDS** middleware. Distros track Ubuntu (Humble/Jazzy).

## 2. The core concepts (the 20% you use daily)

- **Node** — one process doing one job (`/camera_driver`, `/planner`). You compose
  a robot from many nodes.
- **Topic** — a named, typed **stream**; nodes **publish** and **subscribe**.
  Async, many-to-many. E.g. `/scan` (LiDAR), `/camera/image_raw`, `/cmd_vel`
  (velocity command), `/odom`.
- **Message** — the typed data on a topic: `sensor_msgs/LaserScan`,
  `sensor_msgs/Image`, `sensor_msgs/PointCloud2`, `geometry_msgs/Twist`,
  `nav_msgs/Odometry`.
- **Service** — synchronous **request/response** (e.g. "reset the map"). One-shot.
- **Action** — a **long-running goal** with feedback + cancel (e.g. "navigate to
  (x,y)", "follow this trajectory"). The right tool for tasks that take time.
- **Parameter** — runtime config of a node (e.g. a gain, a topic name).
- **tf2** — the live tree of **coordinate transforms** between every frame
  (`map → odom → base_link → camera_link …`), queryable at any timestamp. This is
  Part 3's SE(3) math, as infrastructure. *Most "my sensor data is in the wrong
  place" bugs are tf bugs.*

## 3. Your first nodes (Python / rclpy)

See [`minimal_nodes.py`](minimal_nodes.py) for a complete, commented
publisher+subscriber. The essence:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__("talker")
        self.pub = self.create_publisher(String, "chatter", 10)
        self.create_timer(0.5, self.tick)          # 2 Hz
    def tick(self):
        self.pub.publish(String(data="hello robot"))

def main():
    rclpy.init(); rclpy.spin(Talker()); rclpy.shutdown()
```

Run it, then inspect from the shell — this is most of day-to-day ROS:
```bash
ros2 run my_pkg talker          # start a node
ros2 topic list                 # what streams exist
ros2 topic echo /chatter        # print messages
ros2 topic hz /scan             # measure a topic's rate
ros2 node info /talker          # a node's pubs/subs/services
rqt_graph                       # visualize the node/topic graph
ros2 bag record -a              # record ALL topics to replay offline (gold!)
```

## 4. Describing a robot: URDF & ros2_control

- **URDF/Xacro** — XML describing the robot's **links** (rigid bodies) and
  **joints** (how they connect + limits) + meshes/inertia. The ROS analogue of
  MJCF (Part 4). `robot_state_publisher` turns joint angles + URDF into the live
  **tf** tree (FK from Part 3, automated).
- **ros2_control** — a standard interface between controllers (position/velocity/
  effort, e.g. a PID from Part 14) and hardware *or* simulation. Swap sim↔real
  without changing your controller.

## 5. The big reusable stacks (don't reinvent these)

- **Nav2** — the navigation stack: costmaps (from perception), global planner
  (A*/Dijkstra, Part 8), local planner (DWA/MPC/TEB), recovery behaviors,
  behavior trees for orchestration. Give it a map + goal; it drives the robot.
- **MoveIt 2** — motion planning, IK (Part 3), and collision-aware trajectories
  for arms (Part 9). The manipulation counterpart to Nav2.
- **slam_toolbox / Cartographer** — 2D/3D **SLAM** (Part 7) producing a live map +
  pose.
- **image_pipeline, perception_pcl** — camera rectification, point-cloud tools
  (Part 6), bridging to OpenCV/PCL.
- **ros_gz / gazebo_ros** — simulate the robot + sensors and run your real ROS
  code against it (Part 4).

## 6. Visualization & debugging

- **RViz2** — the 3D viewer: robot model, live LiDAR/point clouds, camera, **tf**
  frames, planned paths, costmaps. **Your primary perception/nav debugging tool.**
- **rqt** — GUI plugins: `rqt_graph` (node/topic graph), `rqt_plot`, `rqt_console`.
- **ros2 bag** — record real sensor streams, replay them offline to develop
  perception/SLAM **without the robot**. Indispensable.

## 7. How ROS 2 ties the whole curriculum together
```
 drivers ─► /scan,/image (Part 5) ─► perception nodes (Part 6) ─► /detections
        └► tf2 (Part 3) ─► slam_toolbox (Part 7) ─► /map,/pose
                                          └► Nav2 (Part 8) ─► /cmd_vel ─► ros2_control (Part 14) ─► motors
```
Every box you built conceptually in Parts 3–14 has a battle-tested ROS 2
implementation. Knowing ROS 2 is how you assemble them into a working robot.

## 8. Realities & advanced bits
- **QoS (Quality of Service)** — reliability/durability settings per topic;
  mismatched QoS = "why isn't my subscriber getting data?" (a classic gotcha).
- **Executors & callback groups** — how callbacks are scheduled (single vs
  multi-threaded); matters for real-time.
- **Lifecycle nodes** — managed states (unconfigured→active) for deterministic
  startup/shutdown.
- **DDS** — the pluggable middleware (Fast-DDS, Cyclone) doing discovery + transport.
- **Micro-ROS** — ROS 2 on microcontrollers (drones, embedded).

## 🛠️ Project (do this on your laptop — Ubuntu/WSL2)
1. Write a `talker`/`listener` pair (use [`minimal_nodes.py`](minimal_nodes.py));
   inspect with `ros2 topic echo` and `rqt_graph`.
2. Spawn a TurtleBot in **Gazebo**, teleop it, and visualize its LiDAR + tf in
   **RViz2**. Then `ros2 bag record` a drive and replay it.
3. Run **Nav2**: give a map + goal and watch global+local planning (Part 8) drive
   the robot. Run **slam_toolbox** to build the map live (Part 7).

## ✅ Check your understanding
1. Topic vs service vs action — when do you use each?
2. What does tf2 manage, and which part's math underlies it?
3. What is URDF, and how does it relate to MJCF and to FK (Part 3)?
4. What problem does ros2_control solve for sim-to-real code reuse?
5. Why is `ros2 bag` so valuable for developing perception?

## 📖 Go deeper
- Official **docs.ros.org** tutorials (Humble/Jazzy) — start here.
- *A Gentle Introduction to ROS* (Jason O'Kane, free book).
- Nav2, MoveIt 2, slam_toolbox documentation; *Programming Robots with ROS* (O'Reilly).

➡️ **Next:** [Part 14 — Control Theory](../part14_control_theory/): the math of
making the robot track what the planner commands.
