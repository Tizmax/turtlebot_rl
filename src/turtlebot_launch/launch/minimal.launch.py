import os

import ament_index_python.packages
from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


import yaml


def generate_launch_description():
    share_dir = ament_index_python.packages.get_package_share_directory('kobuki_node')
    # There are two different ways to pass parameters to a non-composed node;
    # either by specifying the path to the file containing the parameters, or by
    # passing a dictionary containing the key -> value pairs of the parameters.
    # When starting a *composed* node on the other hand, only the dictionary
    # style is supported.  To keep the code between the non-composed and
    # composed launch file similar, we use that style here as well.
    params_file = os.path.join(share_dir, 'config', 'kobuki_node_params.yaml')
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['kobuki_ros_node']['ros__parameters']
    kobuki_ros_node = launch_ros.actions.Node(package='kobuki_node',
                                              executable='kobuki_ros_node',
                                              output='both',
                                              parameters=[params])

    share_dir = ament_index_python.packages.get_package_share_directory('kobuki_velocity_smoother')

    # There are two different ways to pass parameters to a non-composed node;
    # either by specifying the path to the file containing the parameters, or by
    # passing a dictionary containing the key -> value pairs of the parameters.
    # When starting a *composed* node on the other hand, only the dictionary
    # style is supported.  To keep the code between the non-composed and
    # composed launch file similar, we use that style here as well.
    params_file = os.path.join(share_dir, 'config', 'velocity_smoother_params.yaml')
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)['kobuki_velocity_smoother']['ros__parameters']
    velocity_smoother_node = launch_ros.actions.Node(
        package='kobuki_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='both',
        parameters=[params],
        remappings=[
            ('~/smoothed', '/commands/velocity'),
            ],
        )

    return launch.LaunchDescription([
        kobuki_ros_node, 
        velocity_smoother_node,
        IncludeLaunchDescription(
              PythonLaunchDescriptionSource([get_package_share_directory('turtlebot_launch') ,
                        '/turtlebot_description.launch.py'])
              )
        ]
    )
