import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. Tải cấu hình MoveIt
    moveit_config = MoveItConfigsBuilder("robot", package_name="robot_moveit").to_moveit_configs()

    # Lấy đường dẫn file
    pkg_share = get_package_share_directory("robot_moveit")
    rviz_config_file = os.path.join(pkg_share, "config", "moveit.rviz")

    # 2. INCLUDE CONTROL LAUNCH (Thay thế cho toàn bộ phần ros2_control cũ)
    # Phần này sẽ bật: Hardware Interface, Controller Manager, Spawners, Robot State Publisher
    launch_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("robot_control"), "launch", "control.launch.py")
        ),
        # is_sim="False" -> Chạy thật hoặc Mock Hardware (không dùng Gazebo)
        launch_arguments={"is_sim": "False"}.items()
    )

    # 3. Node Move Group

    trajectory_execution = {
    "moveit_manage_controllers": True,
    "trajectory_execution.allowed_execution_duration_scaling": 1.2,
    "trajectory_execution.allowed_goal_duration_margin": 0.5,
    "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    node_move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            trajectory_execution,
            {"use_sim_time": False},
        ],
    )

    # 4. Node RViz
    node_rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
    )

    return LaunchDescription([
        launch_control,
        node_move_group,
        node_rviz,
    ])