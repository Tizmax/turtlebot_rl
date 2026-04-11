"""
Experiment metrics node: listens to /goto/result and generates metrics outputs.

Usage (live run):
    ros2 run turtlebot_rl experiment_metrics --ros-args \
        -p experiment_name:=tdmpc2_run1

Usage (rosbag playback):
    ros2 run turtlebot_rl experiment_metrics --ros-args \
        -p experiment_name:=bag_eval \
        -p expected_trials:=9
"""

import csv
import math
from collections import defaultdict
from statistics import mean, stdev

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

import matplotlib.pyplot as plt


class ExperimentMetrics(Node):
    def __init__(self):
        super().__init__("experiment_metrics")

        # Parameters used for grouping and output naming.
        self.declare_parameter("experiment_name", "experiment")
        self.declare_parameter("expected_trials", 2)

        experiment_name = self.get_parameter("experiment_name").value
        self.expected_trials = int(self.get_parameter("expected_trials").value)

        self.results_file = f"{experiment_name}.csv"
        self.summary_file = f"{experiment_name}_summary.csv"
        self.curves_file = f"{experiment_name}_curves.png"

        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.result_sub = self.create_subscription(
            String, "/goto/result", self.result_callback, 10
        )

        self.results = []
        self.results_by_goal = defaultdict(list)
        self.goal_to_idx = {}
        self.next_goal_idx = 1
        self.finalized = False
        self.active_goal_idx = None
        self.active_goal_name = None
        self.active_goal_pose = None
        self.current_pose_samples = []
        self.active_agent = None
        self.goal_pose_files = {}
        self.goal_pose_writers = {}

        self.csv_file = open(self.results_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "trial", "repeat", "goal_in_sequence", "outcome", "controller",
            "x_goal", "y_goal",
            "time_s", "path_length_m", "final_dist_m",
        ])

        self.get_logger().info(
            f"Experiment metrics started. Listening to /goto/result. "
            f"expected_trials={self.expected_trials}"
        )

    def goal_callback(self, msg):
        goal_key = (
            round(msg.pose.position.x, 3),
            round(msg.pose.position.y, 3),
        )
        if goal_key not in self.goal_to_idx:
            self.goal_to_idx[goal_key] = self.next_goal_idx
            self.next_goal_idx += 1

        self.active_goal_idx = self.goal_to_idx[goal_key]
        self.active_goal_name = f"goal_{self.active_goal_idx}"
        q = msg.pose.orientation
        self.active_goal_pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            self._quaternion_to_yaw(q.x, q.y, q.z, q.w),
        )
        self.current_pose_samples = []

    def odom_callback(self, msg):
        if self.finalized or self.active_goal_idx is None or self.active_goal_pose is None:
            return

        stamp = msg.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        q = msg.pose.pose.orientation
        theta = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)
        goal_x, goal_y, goal_theta = self.active_goal_pose
        self.current_pose_samples.append((
            self.active_goal_idx,
            self.active_goal_name,
            timestamp_s,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            theta,
            goal_x,
            goal_y,
            goal_theta,
        ))

    def result_callback(self, msg):
        if self.finalized:
            return

        parts = msg.data.split(",")
        if len(parts) != 7:
            self.get_logger().warning(
                f"Unexpected result payload '{msg.data}'. Expected 7 comma-separated fields."
            )
            return

        outcome = parts[0]
        controller = parts[1]
        x_goal = float(parts[2])
        y_goal = float(parts[3])
        elapsed = float(parts[4])
        path_length = float(parts[5])
        final_dist = float(parts[6])

        goal_key = (round(x_goal, 3), round(y_goal, 3))
        if goal_key not in self.goal_to_idx:
            self.goal_to_idx[goal_key] = self.next_goal_idx
            self.next_goal_idx += 1
        goal_in_seq = self.goal_to_idx[goal_key]

        trial = len(self.results) + 1
        repeat_num = (trial - 1) // len(self.goal_to_idx) + 1

        record = {
            "trial": trial,
            "repeat": repeat_num,
            "goal_in_sequence": goal_in_seq,
            "outcome": outcome,
            "controller": controller,
            "x_goal": x_goal,
            "y_goal": y_goal,
            "time_s": elapsed,
            "path_length_m": path_length,
            "final_dist_m": final_dist,
        }
        self.results.append(record)
        self.results_by_goal[goal_in_seq].append(record)
        self.active_agent = controller
        self._append_goal_pose_samples(goal_in_seq, trial, repeat_num, controller)
        self.active_goal_idx = None
        self.active_goal_name = None
        self.active_goal_pose = None
        self.current_pose_samples = []

        self.csv_writer.writerow([
            trial, repeat_num, goal_in_seq, outcome, controller,
            f"{x_goal:.3f}", f"{y_goal:.3f}",
            f"{elapsed:.3f}", f"{path_length:.3f}", f"{final_dist:.3f}",
        ])
        self.csv_file.flush()

        self.get_logger().info(
            f"[Trial {trial}] {outcome}: time={elapsed:.3f}s "
            f"path={path_length:.3f}m final_dist={final_dist:.3f}m"
        )

        if self.expected_trials > 0 and len(self.results) >= self.expected_trials:
            self.get_logger().info("Reached expected_trials; finalizing metrics outputs.")
            self.finalize()

    def finalize(self):
        if self.finalized:
            return
        self.finalized = True

        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()

        self._write_summary_csv()
        self._write_curves_plot()

        successes = sum(1 for r in self.results if r["outcome"] == "success")
        total = len(self.results)
        self.get_logger().info(f"Metrics complete: {successes}/{total} successes.")
        self.get_logger().info(f"Results saved to {self.results_file}")
        self.get_logger().info(f"Summary saved to {self.summary_file}")

        for pose_file in self.goal_pose_files.values():
            if pose_file and not pose_file.closed:
                pose_file.close()

    def _write_summary_csv(self):
        with open(self.summary_file, "w", newline="") as summary_file:
            writer = csv.writer(summary_file)
            writer.writerow([
                "goal_in_sequence", "x_goal", "y_goal", "n",
                "success_rate", "mean_time_s", "std_time_s",
                "mean_path_length_m", "std_path_length_m",
                "mean_final_dist_m", "std_final_dist_m",
            ])

            for goal_idx in sorted(self.results_by_goal.keys()):
                rows = self.results_by_goal[goal_idx]
                times = [row["time_s"] for row in rows]
                path_lengths = [row["path_length_m"] for row in rows]
                final_dists = [row["final_dist_m"] for row in rows]
                success_rate = sum(
                    1 for row in rows if row["outcome"] == "success"
                ) / len(rows)

                writer.writerow([
                    goal_idx,
                    f"{rows[0]['x_goal']:.3f}",
                    f"{rows[0]['y_goal']:.3f}",
                    len(rows),
                    f"{success_rate:.3f}",
                    f"{mean(times):.3f}", f"{self._safe_std(times):.3f}",
                    f"{mean(path_lengths):.3f}",
                    f"{self._safe_std(path_lengths):.3f}",
                    f"{mean(final_dists):.3f}",
                    f"{self._safe_std(final_dists):.3f}",
                ])

            if self.results:
                writer.writerow([])
                writer.writerow([
                    "overall",
                    "",
                    "",
                    len(self.results),
                    f"{sum(1 for row in self.results if row['outcome'] == 'success') / len(self.results):.3f}",
                    f"{mean([row['time_s'] for row in self.results]):.3f}",
                    f"{self._safe_std([row['time_s'] for row in self.results]):.3f}",
                    f"{mean([row['path_length_m'] for row in self.results]):.3f}",
                    f"{self._safe_std([row['path_length_m'] for row in self.results]):.3f}",
                    f"{mean([row['final_dist_m'] for row in self.results]):.3f}",
                    f"{self._safe_std([row['final_dist_m'] for row in self.results]):.3f}",
                ])

    def _write_curves_plot(self):
        if plt is None:
            self.get_logger().warning(
                "matplotlib is not available; skipping curve generation."
            )
            return

        if not self.results_by_goal:
            return

        goal_indices = []
        mean_times = []
        std_times = []
        mean_paths = []
        std_paths = []
        mean_dists = []
        std_dists = []
        success_rates = []

        for goal_idx in sorted(self.results_by_goal.keys()):
            rows = self.results_by_goal[goal_idx]
            goal_indices.append(goal_idx)
            mean_times.append(mean([row["time_s"] for row in rows]))
            std_times.append(self._safe_std([row["time_s"] for row in rows]))
            mean_paths.append(mean([row["path_length_m"] for row in rows]))
            std_paths.append(self._safe_std([row["path_length_m"] for row in rows]))
            mean_dists.append(mean([row["final_dist_m"] for row in rows]))
            std_dists.append(self._safe_std([row["final_dist_m"] for row in rows]))
            success_rates.append(
                sum(1 for row in rows if row["outcome"] == "success") / len(rows)
            )

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

        self._plot_with_band(
            axes[0], goal_indices, mean_times, std_times,
            "Time per goal", "time_s"
        )
        self._plot_with_band(
            axes[1], goal_indices, mean_paths, std_paths,
            "Path length per goal", "path_length_m"
        )
        self._plot_with_band(
            axes[2], goal_indices, mean_dists, std_dists,
            "Final distance per goal", "final_dist_m"
        )
        axes[3].plot(goal_indices, success_rates, marker="o")
        axes[3].set_title("Success rate per goal")
        axes[3].set_xlabel("goal_in_sequence")
        axes[3].set_ylabel("success_rate")
        axes[3].set_ylim(0.0, 1.05)
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.curves_file, dpi=150)
        plt.close(fig)

    def _plot_with_band(self, axis, x_values, means, stds, title, ylabel):
        axis.plot(x_values, means, marker="o")
        lower = [mean_value - std_value for mean_value, std_value in zip(means, stds)]
        upper = [mean_value + std_value for mean_value, std_value in zip(means, stds)]
        axis.fill_between(x_values, lower, upper, alpha=0.2)
        axis.set_title(title)
        axis.set_xlabel("goal_in_sequence")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    def _safe_std(self, values):
        if len(values) < 2:
            return 0.0
        return stdev(values)

    def _append_goal_pose_samples(self, goal_idx, trial, repeat_num, agent):
        if not self.current_pose_samples:
            return

        writer = self._get_goal_pose_writer(goal_idx, agent)
        for measurement_idx, (
            goal_number,
            goal_name,
            timestamp_s,
            x_pos,
            y_pos,
            theta,
            goal_x,
            goal_y,
            goal_theta,
        ) in enumerate(self.current_pose_samples, start=1):
            writer.writerow([
                trial,
                repeat_num,
                measurement_idx,
                goal_number,
                goal_name,
                f"{timestamp_s:.6f}",
                f"{x_pos:.4f}",
                f"{y_pos:.4f}",
                f"{theta:.4f}",
                f"{goal_x:.4f}",
                f"{goal_y:.4f}",
                f"{goal_theta:.4f}",
            ])

        key = (goal_idx, agent)
        self.goal_pose_files[key].flush()

    def _get_goal_pose_writer(self, goal_idx, agent):
        key = (goal_idx, agent)
        if key in self.goal_pose_writers:
            return self.goal_pose_writers[key]

        path = f"{self.get_parameter('experiment_name').value}_{agent}_goal_{goal_idx}_positions.csv"
        pose_file = open(path, "w", newline="")
        writer = csv.writer(pose_file, delimiter=";")
        writer.writerow([
            "trial",
            "repeat",
            "measurement",
            "goal_number",
            "goal_name",
            "timestamp_s",
            "x",
            "y",
            "theta",
            "x_goal",
            "y_goal",
            "theta_goal",
        ])

        self.goal_pose_files[key] = pose_file
        self.goal_pose_writers[key] = writer
        return writer

    def _quaternion_to_yaw(self, x, y, z, w):
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def destroy_node(self):
        self.finalize()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExperimentMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Metrics collection interrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
