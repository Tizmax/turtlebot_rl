import math
import time
import csv
import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from turtlebot_rl.TDMPC2Controller import TDMPC2GoToController
from turtlebot_rl.PIDController import PIDGoToController
from turtlebot_rl.PPOController import PPOGoToController


class GoToNode(Node):
    def __init__(self):
        super().__init__("goto_node")

        # ---- ROS2 parameters ----
        self.declare_parameter("controller", "tdmpc2")  # "tdmpc2", "pid", or "ppo"
        self.declare_parameter("goal_timeout", 60.0)     # seconds
        self.declare_parameter("log_dir", "")            # CSV log directory
        self.declare_parameter("cmd_topic", "/mux/autoCommand")

        ctrl_name = self.get_parameter("controller").value
        self.goal_timeout = self.get_parameter("goal_timeout").value
        log_dir = self.get_parameter("log_dir").value
        cmd_topic = self.get_parameter("cmd_topic").value

        # ---- Publishers / Subscribers ----
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.result_pub = self.create_publisher(String, "/goto/result", 10)
        self.marker_pub = self.create_publisher(Marker, "/goto/goal_marker", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_callback, 10
        )

        # ---- Control loop ----
        self.dt = 0.05  # ~20 Hz, matching odom publish rate
        self.timer = self.create_timer(self.dt, self.timer_callback)

        # ---- Controller ----
        if ctrl_name == "pid":
            self.controller = PIDGoToController(dt=self.dt)
        elif ctrl_name == "ppo":
            self.controller = PPOGoToController(dt=self.dt)
        else:
            self.controller = TDMPC2GoToController(dt=self.dt)
        self.ctrl_name = ctrl_name

        # ---- Robot state (from odometry) ----
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_actual = 0.0
        self.w_actual = 0.0
        self.pose_valid = False  # True once first odom message received

        # ---- Goal state ----
        self.x_goal = None
        self.y_goal = None
        self.goal_active = False
        self.goal_start_time = None
        self.path_length = 0.0
        self.prev_x = None
        self.prev_y = None

        # ---- CSV logging ----
        self.csv_writer = None
        self.csv_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_dir = log_dir
        else:
            self.log_dir = None

        self.get_logger().info(
            f"GoTo node ready  [controller={ctrl_name}]. "
            f"Publish goals to /goal_pose (odom frame)."
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def quaternion_to_yaw(self, x, y, z, w):
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        # Pose and velocities from odometry
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.theta = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.v_actual = msg.twist.twist.linear.x
        self.w_actual = msg.twist.twist.angular.z
        self.pose_valid = True

    def goal_callback(self, msg):
        self.x_goal = msg.pose.position.x
        self.y_goal = msg.pose.position.y
        self.goal_active = True
        self.goal_start_time = time.monotonic()
        self.path_length = 0.0
        self.prev_x = self.x
        self.prev_y = self.y

        self._open_csv_log()

        # Publish goal marker as a cylinder
        marker = Marker()
        marker.header = msg.header
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = msg.pose.position.x
        marker.pose.position.y = msg.pose.position.y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.30  # diameter = 2 * 0.15m radius
        marker.scale.y = 0.30
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)

        self.get_logger().info(
            f"New goal received: ({self.x_goal:.2f}, {self.y_goal:.2f})"
        )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def timer_callback(self):
        if not self.goal_active:
            return

        if not self.pose_valid:
            return  # Waiting for first odom message

        # Accumulate path length
        if self.prev_x is not None:
            self.path_length += math.hypot(self.x - self.prev_x, self.y - self.prev_y)
        self.prev_x = self.x
        self.prev_y = self.y

        # Body-frame goal
        dx_w = self.x_goal - self.x
        dy_w = self.y_goal - self.y
        c, s = math.cos(self.theta), math.sin(self.theta)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        dist = math.hypot(dx_b, dy_b)
        bearing = math.atan2(dy_b, dx_b)

        obs = [self.v_actual, self.w_actual,
               math.cos(bearing), math.sin(bearing), dist]

        v, w = self.controller.get_action(obs)

        # Publish twist
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_pub.publish(twist)

        elapsed = time.monotonic() - self.goal_start_time

        # Log to CSV
        self._log_step(elapsed, v, w, dist, bearing)

        # Check success
        if v == 0.0 and w == 0.0:
            self._finish_goal("success", elapsed, dist)
            return

        # Check timeout
        if elapsed > self.goal_timeout:
            self._finish_goal("timeout", elapsed, dist)
            return

    # ------------------------------------------------------------------
    # Goal lifecycle
    # ------------------------------------------------------------------

    def _finish_goal(self, outcome, elapsed, final_dist):
        # Stop the robot
        self.cmd_pub.publish(Twist())
        self.goal_active = False

        result_msg = String()
        result_msg.data = (
            f"{outcome},"
            f"{self.ctrl_name},"
            f"{self.x_goal:.3f},{self.y_goal:.3f},"
            f"{elapsed:.3f},"
            f"{self.path_length:.3f},"
            f"{final_dist:.3f}"
        )
        self.result_pub.publish(result_msg)

        self._close_csv_log()

        self.get_logger().info(
            f"Goal {outcome}: time={elapsed:.2f}s  "
            f"path={self.path_length:.2f}m  final_dist={final_dist:.3f}m"
        )

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _open_csv_log(self):
        self._close_csv_log()
        if self.log_dir is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.ctrl_name}_{stamp}.csv"
        path = os.path.join(self.log_dir, fname)
        self.csv_file = open(path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "t", "x", "y", "theta", "v_actual", "w_actual",
            "v_cmd", "w_cmd", "dist", "bearing",
            "x_goal", "y_goal",
        ])
        self.get_logger().info(f"Logging to {path}")

    def _log_step(self, t, v_cmd, w_cmd, dist, bearing):
        if self.csv_writer is None:
            return
        self.csv_writer.writerow([
            f"{t:.4f}",
            f"{self.x:.4f}", f"{self.y:.4f}", f"{self.theta:.4f}",
            f"{self.v_actual:.4f}", f"{self.w_actual:.4f}",
            f"{v_cmd:.4f}", f"{w_cmd:.4f}",
            f"{dist:.4f}", f"{bearing:.4f}",
            f"{self.x_goal:.4f}", f"{self.y_goal:.4f}",
        ])

    def _close_csv_log(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._close_csv_log()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GoToNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down.")
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()