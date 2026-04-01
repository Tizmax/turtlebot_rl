import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Environment variable check
    env_var = os.environ.get('PERCEPTION_SENSOR', None)

    action = None
    # Conditional launch based on the environment variable
    if env_var == 'astra':
        included_launch_file = 'astra.launch.py'
        action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join('$(find turtlebot_launch)/launch', included_launch_file)]),
        )
    elif env_var =='kinect':
        included_launch_filE = 'kinect.launch.py'
        action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join('$(find turtlebot_launch)/launch', included_launch_file)]),
        )
    else:
        action = LogInfo(msg="3D_SENSOR environment variable set to an unsupported value. Should be either astra or kinect")


    return LaunchDescription([action])
