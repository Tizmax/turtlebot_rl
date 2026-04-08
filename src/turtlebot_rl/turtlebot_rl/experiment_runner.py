"""
Experiment runner: sends a sequence of goals to the GoTo node and collects
per-trial results.

Usage:
    ros2 run turtlebot_rl experiment_runner --ros-args \
        -p goals:="0.5,0.5; 1.0,0.0; 0.0,1.0" \
        -p repeats:=3 \
        -p experiment_name:="tdmpc2_run1" \
        -p settle_time:=3.0
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class ExperimentRunner(Node):
    def __init__(self):
        super().__init__("experiment_runner")

        # ---- Parameters ----
        self.declare_parameter("goals", "2.40,0.60; 4.50,0.80; 6.0,1.80; 7.61,2.77; 8.95,3.50; 9.15,5.77; 8.84,7.99; 7.15,6.67")
        self.declare_parameter("repeats", 1)
        self.declare_parameter("settle_time", 3.0)  # pause between trials
        self.declare_parameter("experiment_name", "experiment")

        goals_str = self.get_parameter("goals").value
        repeats = self.get_parameter("repeats").value
        self.settle_time = self.get_parameter("settle_time").value
        experiment_name = self.get_parameter("experiment_name").value

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

        # Repeat the full goal sequence N times
        self.goals = base_goals * repeats

        # CSV filename from experiment_name
        self.results_file = f"{experiment_name}.csv"

        # ---- Pub / Sub ----
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.result_sub = self.create_subscription(
            String, "/goto/result", self.result_callback, 10
        )

        # ---- State ----
        self.current_idx = 0
        self.waiting_for_result = False
        self.results = []

        # Open results CSV
        self.csv_file = open(self.results_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "trial", "repeat", "outcome", "controller",
            "x_goal", "y_goal",
            "time_s", "path_length_m", "final_dist_m",
        ])

        self.num_base_goals = len(base_goals)

        # Delay initial goal to let goto node settle
        self._startup_timer = self.create_timer(5.0, self._send_first_goal)

        self.get_logger().info(
            f"Experiment '{experiment_name}': "
            f"{len(base_goals)} goals x {repeats} repeats = "
            f"{len(self.goals)} trials"
        )

    def _send_first_goal(self):
        """One-shot timer callback to send the first goal."""
        self._startup_timer.cancel()
        self._send_goal(self.current_idx)

    def _send_goal(self, idx):
        if idx >= len(self.goals):
            self._finish_experiment()
            return

        x, y = self.goals[idx]
        repeat_num = idx // self.num_base_goals + 1
        goal_in_seq = idx % self.num_base_goals + 1

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

        self.waiting_for_result = True
        self.get_logger().info(
            f"[Trial {idx + 1}/{len(self.goals)}  "
            f"repeat {repeat_num}, goal {goal_in_seq}] "
            f"Sent goal ({x:.2f}, {y:.2f})"
        )

    def result_callback(self, msg):
        if not self.waiting_for_result:
            return
        self.waiting_for_result = False

        # Parse: "outcome,controller,x_goal,y_goal,time,path_length,final_dist"
        parts = msg.data.split(",")
        outcome = parts[0]
        controller = parts[1]
        x_goal = parts[2]
        y_goal = parts[3]
        elapsed = parts[4]
        path_length = parts[5]
        final_dist = parts[6]

        trial = self.current_idx + 1
        repeat_num = self.current_idx // self.num_base_goals + 1
        self.results.append(parts)

        self.csv_writer.writerow([
            trial, repeat_num, outcome, controller,
            x_goal, y_goal,
            elapsed, path_length, final_dist,
        ])
        self.csv_file.flush()

        self.get_logger().info(
            f"[Trial {trial}] {outcome}: "
            f"time={elapsed}s  path={path_length}m  final_dist={final_dist}m"
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
        successes = sum(1 for r in self.results if r[0] == "success")
        total = len(self.results)
        self.get_logger().info(
            f"Experiment complete: {successes}/{total} goals reached."
        )
        self.csv_file.close()
        self.get_logger().info(f"Results saved to {self.results_file}")

    def destroy_node(self):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
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
