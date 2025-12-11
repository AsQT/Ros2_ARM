import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory # <--- Cần thêm cái này
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. Khai báo tên các gói
    pkg_moveit = 'robot_moveit'
    pkg_description = 'robot_description'
    
    # --- KHẮC PHỤC LỖI TẠI ĐÂY ---
    # Lấy đường dẫn tuyệt đối đến file URDF trong gói 'robot_description'
    pkg_description_path = get_package_share_directory(pkg_description)
    urdf_file_path = os.path.join(pkg_description_path, "urdf", "robot.urdf.xacro")

    # 2. Tải cấu hình MoveIt
    # Bỏ tham số 'package', truyền thẳng đường dẫn tuyệt đối vào 'file_path'
    moveit_config = MoveItConfigsBuilder("robot", package_name=pkg_moveit) \
        .robot_description(
            file_path=urdf_file_path,   # <--- Truyền đường dẫn full
            mappings={"use_sim": "true"}      
        ) \
        .to_moveit_configs()

    # 3. Include file gazebo.launch.py 
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(pkg_description), 'launch', 'gazebo.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': 'true', 
        }.items()
    )

    # 4. Node Move Group
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
            {
                "moveit_manage_controllers": True,
                "trajectory_execution.allowed_execution_duration_scaling": 1.2,
                "trajectory_execution.allowed_goal_duration_margin": 0.5,
                "trajectory_execution.allowed_start_tolerance": 0.01,
            }
        ],
    )

    # 5. Node RViz
    rviz_config_file = PathJoinSubstitution([FindPackageShare(pkg_moveit), "config", "moveit.rviz"])
    
    run_rviz_node = Node(
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
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([
        gazebo_launch,
        run_move_group_node,
        run_rviz_node
    ])
