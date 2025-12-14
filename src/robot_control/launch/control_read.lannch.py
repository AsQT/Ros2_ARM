import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import Command, LaunchConfiguration

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    desc_share = get_package_share_directory("robot_description")
    ctrl_share = get_package_share_directory("robot_control")

    model_arg = DeclareLaunchArgument(
        "model",
        default_value=os.path.join(desc_share, "urdf", "robot.urdf.xacro"),
    )
    controllers_arg = DeclareLaunchArgument(
        "controllers",
        default_value=os.path.join(ctrl_share, "config", "ros2_controllers.yaml"),
    )

    robot_description = ParameterValue(
        Command(["xacro ", LaunchConfiguration("model"), " use_sim:=false"]),
        value_type=str,
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            LaunchConfiguration("controllers"),
        ],
    )

    spawners = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ctrl_share, "launch", "controllers.launch.py"))
    )

    return LaunchDescription([
        model_arg,
        controllers_arg,
        ros2_control_node,
        spawners,
    ])
