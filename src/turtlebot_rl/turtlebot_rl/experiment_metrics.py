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


class ExperimentMetrics(Node):
    def __init__(self):
        super().__init__("experiment_metrics")

        # Parameters used for grouping and output naming.
        self.declare_parameter("experiment_name", "experiment")
        self.declare_parameter("expected_trials", 2)

        experiment_name = self.get_parameter("experiment_name").value
        self.expected_trials = int(self.get_parameter("expected_trials").value)

        self.experiment_name = experiment_name
        self.results_file = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_initialized = False

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
        self.active_cumulative_reward = 0.0
        self.active_goal_start_time = None
        
        # In-memory tracking for trajectory aggregation (goal_idx, controller) -> list of samples
        self.position_samples_by_goal_controller = defaultdict(list)
        # Track reach times: (trial, goal_idx, controller) -> (reach_time, dist_at_reach)
        self.reach_times = {}
        # Track goal metadata: goal_idx -> {x_goal, y_goal}
        self.goal_metadata = {}

        self.get_logger().info(
            f"Experiment metrics started. Listening to /goto/result. "
            f"expected_trials={self.expected_trials}. CSV file will be created on first result."
        )

    def goal_callback(self, msg):
        goal_key = (
            round(msg.pose.position.x, 3),
            round(msg.pose.position.y, 3),
        )
        if goal_key not in self.goal_to_idx:
            self.goal_to_idx[goal_key] = self.next_goal_idx
            self.goal_metadata[self.next_goal_idx] = {
                "x_goal": msg.pose.position.x,
                "y_goal": msg.pose.position.y,
            }
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
        self.active_cumulative_reward = 0.0
        self.active_goal_start_time = None

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
        
        # Initialize CSV file on first result (now we know the agent)
        if not self.csv_file_initialized:
            self._initialize_csv_file(controller)
        
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

        # Write summary metrics before closing the file
        self._write_summary_csv()
        
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()

        successes = sum(1 for r in self.results if r["outcome"] == "success")
        total = len(self.results)
        self.get_logger().info(f"Metrics complete: {successes}/{total} successes.")
        if self.results_file:
            self.get_logger().info(f"Results and summary saved to {self.results_file}")

        for pose_file in self.goal_pose_files.values():
            if pose_file and not pose_file.closed:
                pose_file.close()

    def _initialize_csv_file(self, agent):
        """Initialize the main CSV file with the agent name in the filename."""
        if self.csv_file_initialized:
            return
        
        self.results_file = f"outputs/{self.experiment_name}_{agent}.csv"
        self.csv_file = open(self.results_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "trial", "repeat", "goal_in_sequence", "outcome", "controller",
            "x_goal", "y_goal", "time_s", "path_length_m", "final_dist_m",
            "n", "success_rate", "mean_reach_time_s", "std_reach_time_s",
            "mean_normalized_reach_time", "std_normalized_reach_time",
            "mean_path_length_m", "std_path_length_m", "mean_final_dist_m", "std_final_dist_m",
        ])
        self.csv_file_initialized = True
        self.get_logger().info(f"Created results file: {self.results_file}")

    def _write_summary_csv(self):
        """
        Write all trial rows to CSV with aggregated summary stats included.
        Each row contains both individual trial data and the per-goal-controller aggregate metrics.
        """
        if not self.csv_file_initialized:
            return
        
        # First, compute summary stats per (goal, controller) pair
        summary_stats = {}  # (goal_idx, controller) -> dict of stats
        
        by_goal_controller = defaultdict(list)
        for record in self.results:
            key = (record["goal_in_sequence"], record["controller"])
            by_goal_controller[key].append(record)
        
        for (goal_idx, controller) in by_goal_controller.keys():
            records = by_goal_controller[(goal_idx, controller)]
            
            # Compute aggregates
            success_rate = sum(1 for r in records if r["outcome"] == "success") / len(records)
            path_lengths = [r["path_length_m"] for r in records]
            final_dists = [r["final_dist_m"] for r in records]
            
            # Reach time metrics
            reach_times = []
            normalized_reach_times = []
            for trial_num, record in enumerate(self.results, 1):
                if record["goal_in_sequence"] != goal_idx or record["controller"] != controller:
                    continue
                
                reach_key = (trial_num, goal_idx, controller)
                if reach_key in self.reach_times:
                    reach_time, dist_at_reach = self.reach_times[reach_key]
                    reach_times.append(reach_time)
                    
                    key = (goal_idx, controller)
                    samples = self.position_samples_by_goal_controller[key]
                    trial_samples = [s for s in samples if s["trial"] == trial_num]
                    
                    if trial_samples:
                        initial_dist = trial_samples[0]["dist_to_goal"]
                        if initial_dist > 0:
                            normalized_reach_times.append(reach_time / initial_dist)
            
            summary_stats[(goal_idx, controller)] = {
                "n": len(records),
                "success_rate": success_rate,
                "mean_reach_time": mean(reach_times) if reach_times else 0.0,
                "std_reach_time": self._safe_std(reach_times),
                "mean_norm_reach": mean(normalized_reach_times) if normalized_reach_times else 0.0,
                "std_norm_reach": self._safe_std(normalized_reach_times),
                "mean_path_length": mean(path_lengths),
                "std_path_length": self._safe_std(path_lengths),
                "mean_final_dist": mean(final_dists),
                "std_final_dist": self._safe_std(final_dists),
            }
        
        # Now write all trial rows with summary data included
        for record in self.results:
            goal_idx = record["goal_in_sequence"]
            controller = record["controller"]
            key = (goal_idx, controller)
            stats = summary_stats.get(key, {})
            
            self.csv_writer.writerow([
                record["trial"],
                record["repeat"],
                goal_idx,
                record["outcome"],
                controller,
                f"{record['x_goal']:.3f}",
                f"{record['y_goal']:.3f}",
                f"{record['time_s']:.3f}",
                f"{record['path_length_m']:.3f}",
                f"{record['final_dist_m']:.3f}",
                stats.get("n", ""),
                f"{stats.get('success_rate', 0):.3f}" if stats else "",
                f"{stats.get('mean_reach_time', 0):.3f}" if stats else "",
                f"{stats.get('std_reach_time', 0):.3f}" if stats else "",
                f"{stats.get('mean_norm_reach', 0):.3f}" if stats else "",
                f"{stats.get('std_norm_reach', 0):.3f}" if stats else "",
                f"{stats.get('mean_path_length', 0):.3f}" if stats else "",
                f"{stats.get('std_path_length', 0):.3f}" if stats else "",
                f"{stats.get('mean_final_dist', 0):.3f}" if stats else "",
                f"{stats.get('std_final_dist', 0):.3f}" if stats else "",
            ])

    def _safe_std(self, values):
        if len(values) < 2:
            return 0.0
        return stdev(values)

    def _append_goal_pose_samples(self, goal_idx, trial, repeat_num, agent):
        if not self.current_pose_samples:
            return

        writer = self._get_goal_pose_writer(goal_idx, agent)
        
        # Reward calculation coefficients
        lambda_1 = 35.0      # Distance progress
        lambda_2 = 0.02      # Bearing alignment
        lambda_3 = 0.3       # Smoothness penalty
        lambda_4 = -0.04     # Time penalty
        lambda_5 = 40.0      # Goal bonus
        k1 = -10.0           # Bearing quartic
        k2 = -0.1            # Bearing quadratic
        k3 = -0.33           # Smoothness (unused in this formulation)
        
        # Track previous state for progress and smoothness calculation
        prev_distance = None
        prev_theta = None
        
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
            # Initialize goal start time on first measurement
            if self.active_goal_start_time is None:
                self.active_goal_start_time = timestamp_s
            
            # Compute distance to goal
            dist_to_goal = math.sqrt((x_pos - goal_x)**2 + (y_pos - goal_y)**2)
            
            # Calculate distance progress (positive if getting closer)
            if prev_distance is not None:
                distance_progress = prev_distance - dist_to_goal
            else:
                distance_progress = 0.0
            
            # Calculate bearing to goal (angle pointing towards goal)
            bearing = math.atan2(goal_y - y_pos, goal_x - x_pos)
            
            # Calculate bearing error (normalized to [-π, π])
            bearing_error = bearing - theta
            bearing_error = math.atan2(math.sin(bearing_error), math.cos(bearing_error))
            
            # Calculate smoothness metric (angular velocity magnitude)
            if prev_theta is not None:
                delta_theta = theta - prev_theta
                delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))
                smoothness = abs(delta_theta)
            else:
                smoothness = 0.0
            
            # Calculate time penalty (per step)
            time_penalty = lambda_4
            
            # Check if goal is reached
            goal_reached = 1.0 if dist_to_goal <= 0.15 else 0.0
            
            # Calculate reward components
            distance_reward = lambda_1 * distance_progress
            bearing_reward = lambda_2 * (1.0 - abs(bearing_error) / math.pi)
            smoothness_penalty = -lambda_3 * smoothness
            goal_bonus = lambda_5 * goal_reached
            bearing_quartic = k1 * (bearing_error ** 4)
            bearing_quadratic = k2 * (bearing_error ** 2)
            
            # Total step reward
            step_reward = (
                distance_reward +
                bearing_reward +
                smoothness_penalty +
                time_penalty +
                goal_bonus +
                bearing_quartic +
                bearing_quadratic
            )
            
            # Accumulate reward
            self.active_cumulative_reward += step_reward
            
            # Store sample in memory for later aggregation
            self.position_samples_by_goal_controller[(goal_idx, agent)].append({
                "trial": trial,
                "time_s": timestamp_s,
                "x": x_pos,
                "y": y_pos,
                "dist_to_goal": dist_to_goal,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "cumulative_reward": self.active_cumulative_reward,
            })
            
            # Track reach time on first sample where dist <= 0.15m
            reach_key = (trial, goal_idx, agent)
            if reach_key not in self.reach_times and dist_to_goal <= 0.15:
                self.reach_times[reach_key] = (timestamp_s, dist_to_goal)
            
            # Write to CSV file
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
                f"{dist_to_goal:.6f}",
                f"{bearing_error:.6f}",
                f"{distance_reward:.6f}",
                f"{bearing_reward:.6f}",
                f"{smoothness_penalty:.6f}",
                f"{time_penalty:.6f}",
                f"{goal_bonus:.6f}",
                f"{bearing_quartic:.6f}",
                f"{bearing_quadratic:.6f}",
                f"{step_reward:.6f}",
                f"{self.active_cumulative_reward:.6f}",
            ])
            
            # Update previous state for next iteration
            prev_distance = dist_to_goal
            prev_theta = theta

        key = (goal_idx, agent)
        self.goal_pose_files[key].flush()

    def _get_goal_pose_writer(self, goal_idx, agent):
        key = (goal_idx, agent)
        if key in self.goal_pose_writers:
            return self.goal_pose_writers[key]

        path = f"outputs/{self.get_parameter('experiment_name').value}_{agent}_goal_{goal_idx}.csv"
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
            "dist_to_goal",
            "bearing_error",
            "distance_reward",
            "bearing_reward",
            "smoothness_penalty",
            "time_penalty",
            "goal_bonus",
            "bearing_quartic",
            "bearing_quadratic",
            "step_reward",
            "cumulative_reward",
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
