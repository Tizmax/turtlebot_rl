import os

import launch_ros
from launch_ros.actions.node import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

from launch.actions.declare_launch_argument import DeclareLaunchArgument
from launch.launch_description import LaunchDescription
from launch.substitutions.launch_configuration import LaunchConfiguration


def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package="kinect_ros2").find(
        "kinect_ros2"
    )
    default_rviz_config_path = os.path.join(pkg_share, "rviz/pointcloud.rviz")

    return LaunchDescription(
        [
            Node(
                package="kinect_ros2",
                executable="kinect_ros2_node",
                name="kinect_ros2",
                namespace="kinect",
                output="screen"
            ),
        ComposableNodeContainer( 
            name='container', 
            namespace='', 
            package='rclcpp_components', 
            executable='component_container', 
            composable_node_descriptions=[ 
                # Driver itself 
                ComposableNode( 
                    package='depth_image_proc', 
                    plugin='depth_image_proc::PointCloudXyzrgbNode', 
                    name='depth_to_pc', 
                    remappings=[('rgb/image_rect_color', '/kinect/image_raw'), 
                                ('rgb/camera_info', '/kinect/camera_info'), 
                                ('depth_registered/image_rect', '/kinect/depth/image_raw')]
                ), 
            ], 
            output='screen', 
        ),

        Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            remappings=[('cloud_in', '/points') ],
            parameters=[{
                # 'target_frame': 'cloud',
                'transform_tolerance': 0.01,
                'min_height': 0.0,
                'max_height': 1.0,
                'angle_min': -1.5708,  # -M_PI/2
                'angle_max': 1.5708,  # M_PI/2
                'angle_increment': 0.0087,  # M_PI/360.0
                'scan_time': 0.3333,
                'range_min': 0.45,
                'range_max': 4.0,
                'use_inf': True,
                'inf_epsilon': 1.0
            }],
            name='pointcloud_to_laserscan'
        )


        ]
    )
