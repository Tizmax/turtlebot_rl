"""
Launch file for GoTo experiments.

Brings up: slam_toolbox (sync), tb_slam_env (aruco), teleop mux, and the goto node.

Usage:
    ros2 launch turtlebot_rl experiment.launch.py controller:=tdmpc2 log_dir:=logs/tdmpc2
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
        "controller", default_value="tdmpc2",
        description="Controller type: tdmpc2 or pid")
    goal_timeout_arg = DeclareLaunchArgument(
        "goal_timeout", default_value="60.0",
        description="Goal timeout in seconds")
    log_dir_arg = DeclareLaunchArgument(
        "log_dir", default_value="",
        description="CSV log directory for per-trial data")

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
        executable="goto",
        name="goto_node",
        parameters=[{
            "controller": LaunchConfiguration("controller"),
            "goal_timeout": LaunchConfiguration("goal_timeout"),
            "log_dir": LaunchConfiguration("log_dir"),
            "cmd_topic": "/mux/autoCommand",
        }],
        output="screen",
    )

    return LaunchDescription([
        controller_arg,
        goal_timeout_arg,
        log_dir_arg,
        slam_tb_sync,
        tb_slam_env,
        teleop_mux_tb,
        goto_node,
    ])
