"""
Experiment runner: sends a sequence of goals to the GoTo node.

Usage:
    ros2 run turtlebot_rl experiment_runner --ros-args \
        -p goals:="0.5,0.5; 1.0,0.0; 0.0,1.0" \
        -p settle_time:=3.0
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Empty


class ExperimentRunner(Node):
    def __init__(self):
        super().__init__("experiment_runner")

        # ---- Parameters ----
        self.declare_parameter("goals", "2.0,0.0; 1.732,1.0; 1.732,-1.0")
        self.declare_parameter("settle_time", 3.0)  # pause between trials
        self.declare_parameter("reset_odom_enabled", True)  # reset odom between goals
        self.declare_parameter("frame_id", "map")  # frame for goal poses

        goals_str = self.get_parameter("goals").value
        self.settle_time = self.get_parameter("settle_time").value
        self.reset_odom_enabled = self.get_parameter("reset_odom_enabled").value
        self.frame_id = self.get_parameter("frame_id").value

        # Parse goals: "x1,y1; x2,y2; ..."
        base_goals = []
        for pair in goals_str.split(";"):
            pair = pair.strip()
            if pair:
                x, y = pair.split(",")
                base_goals.append((float(x.strip()), float(y.strip())))

        if not base_goals:
            self.get_logger().error("No goals provided. Exiting.")
            return

        self.goals = base_goals

        # ---- Pub / Sub ----
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.result_sub = self.create_subscription(
            String, "/goto/result", self.result_callback, 10
        )
        self.odom_reset_pub = self.create_publisher(
            Empty, "/commands/reset_odometry", 10
        )

        # ---- State ----
        self.current_idx = 0
        self.waiting_for_result = False

        self.num_base_goals = len(base_goals)

        # Wait for goto node to be ready (subscribed to /goal_pose)
        # and for settle_time to elapse
        self._startup_timer = self.create_timer(0.1, self._check_goto_ready)

        self.get_logger().info(
            f"Experiment: {len(self.goals)} goals, {len(self.goals)} trials"
        )

    def _send_first_goal(self):
        """One-shot timer callback to send the first goal."""
        self._startup_timer.cancel()
        self._send_goal(self.current_idx)

    def _check_goto_ready(self):
        """Check if goto_odom is subscribed and settle_time has elapsed."""
        # Check if anyone is subscribed to /goal_pose
        num_subscribers = self.goal_pub.get_subscription_count()
        if num_subscribers > 0:
            # goto node is ready, now wait settle_time before sending first goal
            self._startup_timer.cancel()
            self._startup_timer = self.create_timer(
                self.settle_time, self._send_first_goal
            )
            self.get_logger().info(
                f"GoTo node detected (subscribers={num_subscribers}). "
                f"Waiting {self.settle_time}s before sending first goal."
            )

    def _send_goal(self, idx):
        if idx >= len(self.goals):
            self._finish_experiment()
            return

        x, y = self.goals[idx]
        goal_in_seq = idx % self.num_base_goals + 1

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        self.waiting_for_result = True
        self.get_logger().info(
            f"[Trial {idx + 1}/{len(self.goals)}  "
            f"goal {goal_in_seq}] "
            f"Sent goal ({x:.2f}, {y:.2f})"
        )

    def result_callback(self, msg):
        if not self.waiting_for_result:
            return
        self.waiting_for_result = False

        trial = self.current_idx + 1
        goal_in_seq = self.current_idx % self.num_base_goals + 1

        self.get_logger().info(
            f"[Trial {trial}/{len(self.goals)}  goal {goal_in_seq}] "
            "Goal result received, sending next goal."
        )

        # Reset odometry if enabled (for odom-frame experiments)
        if self.reset_odom_enabled:
            self.odom_reset_pub.publish(Empty())
            self.get_logger().info(
                f"[Trial {trial}/{len(self.goals)}] Odometry reset."
            )

        self.current_idx += 1
        if self.current_idx < len(self.goals):
            # Wait for robot to settle, then send next goal
            idx = self.current_idx  # capture for closure
            timer = self.create_timer(
                self.settle_time,
                lambda: (timer.cancel(), self._send_goal(idx)),
            )
        else:
            self._finish_experiment()

    def _finish_experiment(self):
        self.get_logger().info("Experiment complete: all goals sent.")

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Experiment interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
