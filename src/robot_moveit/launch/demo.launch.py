import os
import yaml
import tempfile
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def load_yaml(package_name, file_path):
    try:
        package_path = get_package_share_directory(package_name)
        absolute_file_path = os.path.join(package_path, file_path)
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except Exception:
        return {}

def sanitize_config(config):
    if isinstance(config, dict):
        return {k: sanitize_config(v) for k, v in config.items() if v is not None}
    elif isinstance(config, list):
        return [sanitize_config(i) for i in config if i is not None]
    return config

def generate_launch_description():
    # 1. KHAI BÁO BIẾN USE_SIM (QUAN TRỌNG)
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim",
            default_value="false",
            description="False: Mock, True: Gazebo",
        )
    )
    use_sim = LaunchConfiguration("use_sim")
    
    robot_pkg_share = FindPackageShare("robot")
    moveit_pkg_share = FindPackageShare("robot_moveit")

    xacro_file = PathJoinSubstitution([robot_pkg_share, "urdf", "robot.urdf.xacro"])
    srdf_file = PathJoinSubstitution([moveit_pkg_share, "config", "robot.srdf"])
    rviz_config_file = PathJoinSubstitution([moveit_pkg_share, "config", "moveit.rviz"])

    # 2. LOAD ROBOT DESCRIPTION
    robot_description_content = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", xacro_file, " ", "use_sim:=", use_sim]),
        value_type=str
    )
    robot_description = {"robot_description": robot_description_content}

    robot_description_semantic_content = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", srdf_file]),
        value_type=str
    )
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    # 3. LOAD CONFIG YAML
    kinematics_info = load_yaml("robot_moveit", "config/kinematics.yaml")
    moveit_controllers_info = load_yaml("robot_moveit", "config/moveit_controllers.yaml")
    ompl_planning_pipeline_config = load_yaml("robot_moveit", "config/ompl_planning.yaml")
    
    # Load Controller Config (Dùng file gốc)
    ros2_controllers_file = PathJoinSubstitution([moveit_pkg_share, "config", "ros2_controllers.yaml"])

    # 4. CẤU HÌNH PLANNING PIPELINE
    planning_pipeline_parameters = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "start_state_max_bounds_error": 0.1,
        }
    }
    if ompl_planning_pipeline_config:
        planning_pipeline_parameters["ompl"].update(ompl_planning_pipeline_config)

    # QUAN TRỌNG: Chỉ định danh sách Adapter (Đã xóa cái gây lỗi crash)
    planning_pipeline_parameters["ompl"]["request_adapters"] = [
        "default_planning_request_adapters/ResolveConstraintFrames",
        "default_planning_request_adapters/ValidateWorkspaceBounds",
        "default_planning_request_adapters/CheckStartStateBounds",
        "default_planning_request_adapters/CheckStartStateCollision"
    ]

    # 5. CÁC NODE
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_info},
            moveit_controllers_info,
            planning_pipeline_parameters,
            {"use_sim_time": use_sim},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": kinematics_info},
            planning_pipeline_parameters,
            {"use_sim_time": use_sim},
            moveit_controllers_info,
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim}],
    )
    
    # Controller Manager (Load file trực tiếp)
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, ros2_controllers_file],
        output="both",
        condition=UnlessCondition(use_sim) 
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        condition=UnlessCondition(use_sim)
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        condition=UnlessCondition(use_sim)
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        condition=UnlessCondition(use_sim)
    )

    return LaunchDescription(declared_arguments + [
        robot_state_publisher,
        move_group_node,
        rviz_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
    ])