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
                package="topic_tools",
                executable="throttle",
                name="astra_throttle",
                arguments=["messages","astra/color/image_raw","3.0"],
                parameters=[{'lazy': True}],
                output="screen"
            ),
            Node(
                package="topic_tools",
                executable="throttle",
                name="astra_compressed_throttle",
                arguments=["messages","astra/color/image_raw/compressed","3.0"],
                parameters=[{'lazy': True}],
                output="screen"
            ),
            Node(
                package="topic_tools",
                executable="throttle",
                name="point_throttle",
                arguments=["messages","astra/depth/points","1.0"],
                parameters=[{'lazy': True}],
                output="screen"
            ),
            Node(
                package="astra_camera",
                executable="astra_camera_node",
                name="astra_camera",
                namespace="astra",
            parameters=[{
                    "camera_name": "camera",
                    "depth_registration": True,
                    "serial_number": "",
                    "device_num": 1,
                    "vendor_id": "0x2bc5",
                    "product_id": "",
                    "enable_point_cloud": True,
                    "enable_colored_point_cloud": False,
                    "point_cloud_qos": "default", # or "SENSOR_DATA"
                    "connection_delay": 100,
                    "color_width": 640,
                    "color_height": 480,
                    "color_fps": 5,
                    "enable_color": True,
                    "flip_color": False,
                    "color_qos": "default", # or "SENSOR_DATA"
                    "color_camera_info_qos": "default", # or "SENSOR_DATA"
                    "depth_width": 640,
                    "depth_height": 480,
                    "depth_fps": 5,
                    "enable_depth": True,
                    "flip_depth": False,
                    "depth_qos": "default", # or "SENSOR_DATA"
                    "depth_camera_info_qos": "default", # or "SENSOR_DATA"
                    "ir_width": 640,
                    "ir_height": 480,
                    "ir_fps": 5,
                    "enable_ir": True,
                    "flip_ir": False,
                    "ir_qos": "default", # or "SENSOR_DATA"
                    "ir_camera_info_qos": "default", # or "SENSOR_DATA"
                    "publish_tf": True,
                    "tf_publish_rate": 10.0,
                    "ir_info_url": "",
                    "color_info_url": "",
                    "color_depth_synchronization": True,
                    "oni_log_level": "verbose",
                    "oni_log_to_console": False,
                    "oni_log_to_file": False,
                    "enable_d2c_viewer": False,
                    "enable_publish_extrinsic": False,
                }],
                output="screen"
            ),
        # ComposableNodeContainer( 
        #     name='container', 
        #     namespace='', 
        #     package='rclcpp_components', 
        #     executable='component_container', 
        #     composable_node_descriptions=[ 
        #         # Driver itself 
        #         ComposableNode( 
        #             package='depth_image_proc', 
        #             plugin='depth_image_proc::PointCloudXyzrgbNode', 
        #             name='depth_to_pc', 
        #             remappings=[('rgb/image_rect_color', '/kinect/image_raw'), 
        #                         ('rgb/camera_info', '/kinect/camera_info'), 
        #                         ('depth_registered/image_rect', '/kinect/depth/image_raw')]
        #         ), 
        #     ], 
        #     output='screen', 
        # ),

        # Node(
        #     package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
        #     remappings=[('cloud_in', '/points') ],
        #     parameters=[{
        #         # 'target_frame': 'cloud',
        #         'transform_tolerance': 0.01,
        #         'min_height': 0.0,
        #         'max_height': 1.0,
        #         'angle_min': -1.5708,  # -M_PI/2
        #         'angle_max': 1.5708,  # M_PI/2
        #         'angle_increment': 0.0087,  # M_PI/360.0
        #         'scan_time': 0.3333,
        #         'range_min': 0.45,
        #         'range_max': 4.0,
        #         'use_inf': True,
        #         'inf_epsilon': 1.0
        #     }],
        #     name='pointcloud_to_laserscan'
        # )


        ]
    )
