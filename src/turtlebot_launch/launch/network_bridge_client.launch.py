from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_dir = get_package_share_directory("turtlebot_launch")
    tcp_config = config_dir + "/config/network_bridge_tcp_client.yaml"

    return LaunchDescription(
        [
            Node(
                package="network_bridge",
                executable="network_bridge",
                name="network_bridge_client",
                output="screen",
                parameters=[tcp_config],
            ),
        ]
    )
