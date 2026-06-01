"""Minimal ROS 2 (rclpy) publisher + subscriber - the 'hello world' of ROS.

This is illustrative reference code: it runs on a machine with ROS 2 installed
(`pip`/apt `rclpy`), NOT in this curriculum's plain-Python sandbox. Read it to
learn the node/topic/spin pattern, then run it on your Ubuntu/WSL2 ROS 2 setup.

Run (after `source /opt/ros/<distro>/setup.bash`):
    python minimal_nodes.py talker      # terminal 1
    python minimal_nodes.py listener    # terminal 2
    # then inspect:  ros2 topic echo /chatter   |   rqt_graph
"""
from __future__ import annotations

import sys

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # so the file imports cleanly even without ROS installed
    rclpy = None
    Node = object


class Talker(Node):
    """Publishes a String on /chatter at 2 Hz."""

    def __init__(self):
        super().__init__("talker")
        # publisher: (msg_type, topic_name, queue_depth)
        self.pub = self.create_publisher(String, "chatter", 10)
        self.count = 0
        # a timer is the standard way to do periodic work in a node
        self.create_timer(0.5, self.on_timer)
        self.get_logger().info("talker up - publishing on /chatter")

    def on_timer(self):
        msg = String()
        msg.data = f"hello robot {self.count}"
        self.pub.publish(msg)
        self.count += 1


class Listener(Node):
    """Subscribes to /chatter and logs what it hears."""

    def __init__(self):
        super().__init__("listener")
        # subscription: (msg_type, topic, callback, queue_depth)
        self.create_subscription(String, "chatter", self.on_msg, 10)
        self.get_logger().info("listener up - subscribed to /chatter")

    def on_msg(self, msg):
        self.get_logger().info(f"heard: {msg.data}")


def main():
    if rclpy is None:
        print("rclpy not found. Install ROS 2 and `source` its setup.bash, then "
              "run this on your robot/dev machine. See README.md.")
        return
    role = sys.argv[1] if len(sys.argv) > 1 else "talker"
    rclpy.init()
    node = Talker() if role == "talker" else Listener()
    try:
        rclpy.spin(node)          # process callbacks until Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
