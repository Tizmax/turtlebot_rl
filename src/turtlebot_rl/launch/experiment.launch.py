"""
Launch file for GoTo experiments with experiment runner and metrics.

Brings up: slam_toolbox (sync), tb_slam_env (aruco), teleop mux, 
           goto_odom node, experiment_runner, and experiment_metrics.

Usage (odom frame with reset):
    ros2 launch turtlebot_rl experiment.launch.py \
        controller:=tdmpc2 \
        goals:="0.5,0.5; 1.0,0.0; 0.0,1.0" \
        reset_odom_enabled:=true \
        experiment_name:=tdmpc2_odom_test \
        expected_trials:=9

    ros2 launch turtlebot_rl experiment.launch.py \
        experiment_name:=5g \
        controller:=pid \
        goals:="1.473,0.393;1.473,-0.393;0.000,0.814;0.000,-0.814;0.203,0.409;0.203,-0.409;0.490,0.740;0.490,-0.740;1.153,0.387;1.153,-0.387" \ 
        expected_trials:=10

Usage (map frame without reset):
    ros2 launch turtlebot_rl experiment.launch.py \
        controller:=pid \
        goals:="0.5,0.5; 1.0,0.0" \
        reset_odom_enabled:=false \
        frame_id:=map \
        experiment_name:=pid_map_test \
        expected_trials:=6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    turtlebot_launch_dir = get_package_share_directory("turtlebot_launch")
    vrep_teleop_dir = get_package_share_directory("vrep_ros_teleop")

    # ---- Launch arguments ----
    controller_arg = DeclareLaunchArgument(
        "controller", default_value="pid",
        description="Controller type: tdmpc2, pid, or ppo")
    goal_timeout_arg = DeclareLaunchArgument(
        "goal_timeout", default_value="60.0",
        description="Goal timeout in seconds")
    log_dir_arg = DeclareLaunchArgument(
        "log_dir", default_value="",
        description="CSV log directory for per-trial data")
    goals_arg = DeclareLaunchArgument(
        "goals", default_value="2.0,0.0; 1.732,1.0; 1.732,-1.0",
        description="Goals as 'x1,y1; x2,y2; ...'")
    settle_time_arg = DeclareLaunchArgument(
        "settle_time", default_value="3.0",
        description="Settle time between goals (seconds)")
    reset_odom_enabled_arg = DeclareLaunchArgument(
        "reset_odom_enabled", default_value="true",
        description="Reset odometry between goals (true/false)")
    frame_id_arg = DeclareLaunchArgument(
        "frame_id", default_value="odom",
        description="Goal frame: 'map' or 'odom'")
    experiment_name_arg = DeclareLaunchArgument(
        "experiment_name", default_value="experiment",
        description="Experiment name for metric outputs")
    expected_trials_arg = DeclareLaunchArgument(
        "expected_trials", default_value="3",
        description="Expected number of trials for metrics")

    # ---- Include existing launch files ----
    slam_tb_sync = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot_launch_dir, "slam_tb_sync.launch.py")
        )
    )

    tb_slam_env = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot_launch_dir, "tb_slam_env.launch.py")
        )
    )

    teleop_mux_tb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(vrep_teleop_dir, "teleop_mux_tb.launch.py")
        )
    )

    # ---- GoTo node ----
    goto_node = Node(
        package="turtlebot_rl",
        executable="goto_odom",
        name="goto_node",
        parameters=[{
            "controller": LaunchConfiguration("controller"),
            "goal_timeout": LaunchConfiguration("goal_timeout"),
            "log_dir": LaunchConfiguration("log_dir"),
            "cmd_topic": "/mux/autoCommand",
        }],
        output="screen",
    )

    # ---- Experiment runner node ----
    experiment_runner_node = Node(
        package="turtlebot_rl",
        executable="experiment_runner",
        name="experiment_runner",
        parameters=[{
            "goals": LaunchConfiguration("goals"),
            "settle_time": LaunchConfiguration("settle_time"),
            "reset_odom_enabled": LaunchConfiguration("reset_odom_enabled"),
            "frame_id": LaunchConfiguration("frame_id"),
        }],
        output="screen",
    )

    # ---- Experiment metrics node ----
    experiment_metrics_node = Node(
        package="turtlebot_rl",
        executable="experiment_metrics",
        name="experiment_metrics",
        parameters=[{
            "experiment_name": LaunchConfiguration("experiment_name"),
            "expected_trials": LaunchConfiguration("expected_trials"),
        }],
        output="screen",
    )

    return LaunchDescription([
        controller_arg,
        goal_timeout_arg,
        log_dir_arg,
        goals_arg,
        settle_time_arg,
        reset_odom_enabled_arg,
        frame_id_arg,
        experiment_name_arg,
        expected_trials_arg,
        slam_tb_sync,
        tb_slam_env,
        teleop_mux_tb,
        goto_node,
        experiment_runner_node,
        experiment_metrics_node,
    ])
