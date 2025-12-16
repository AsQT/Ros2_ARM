# Chạy Moveit hiển thị trong Rviz và execute vào Gazebo
# File gốc ở pkg robot_moveit
import os
from launch                             import LaunchDescription
from launch.actions                     import IncludeLaunchDescription
from launch.launch_description_sources  import PythonLaunchDescriptionSource
from launch.substitutions               import PathJoinSubstitution
from launch_ros.actions                 import Node
from launch_ros.substitutions           import FindPackageShare
from ament_index_python.packages        import get_package_share_directory
from moveit_configs_utils               import MoveItConfigsBuilder

def generate_launch_description():

    pkg_moveit              = 'robot_moveit'
    pkg_description         = 'robot_description'
    pkg_description_path    = get_package_share_directory(pkg_description)
    urdf_file_path          = os.path.join(pkg_description_path, "urdf", "robot.urdf.xacro")

    moveit_config = (
        MoveItConfigsBuilder("robot", package_name =pkg_moveit)
        .robot_description(file_path =urdf_file_path, 
                           mappings  ={"use_sim": "true"})
        .to_moveit_configs()
    )
    gazebo_launch = IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                        PathJoinSubstitution([FindPackageShare(pkg_description), 
                                            'launch', 
                                            'gazebo.launch.py'])
                        ),
                        launch_arguments={ 'use_sim_time': 'true', }.items()
    )

    run_move_group_node = Node(
                        package     ="moveit_ros_move_group",
                        executable  ="move_group",
                        output      ="screen",
                        name        ="move_group",
                        parameters  =[
                                    moveit_config.to_dict(),
                                    {"use_sim_time": True},
                                    {
                                        "moveit_manage_controllers": True,
                                        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
                                        "trajectory_execution.allowed_goal_duration_margin": 0.5,
                                        "trajectory_execution.allowed_start_tolerance": 0.01
                                    }
                                ]
    )
    
    rviz_config_file = PathJoinSubstitution([FindPackageShare(pkg_moveit), 
                                             "config", 
                                             "moveit.rviz"])
    
    run_rviz_node = Node(
                    package         ="rviz2",
                    executable      ="rviz2",
                    name            ="rviz2",
                    output          ="log",
                    arguments       =["-d", rviz_config_file],
                    parameters      =[
                                    moveit_config.robot_description,
                                    moveit_config.robot_description_semantic,
                                    moveit_config.planning_pipelines,
                                    moveit_config.robot_description_kinematics,
                                    {"use_sim_time": True},
                                    ]
    )

    return LaunchDescription([
        gazebo_launch,
        run_move_group_node,
        run_rviz_node
    ])
