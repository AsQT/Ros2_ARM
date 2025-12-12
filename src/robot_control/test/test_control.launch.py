import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # 1. Bật Gazebo (Sim)
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("robot_description"), "launch", "gazebo.launch.py")
        )
    )

    # 2. Bật GUI Slider (Nút điều khiển)
    # Thêm use_sim_time để đồng bộ với Gazebo
    slider_control_node = Node(
        package="robot_control",
        executable="slider_control",
        output="screen",
        parameters=[{"use_sim_time": True}] 
    )

    return LaunchDescription([
        gazebo_sim,
        slider_control_node
    ])