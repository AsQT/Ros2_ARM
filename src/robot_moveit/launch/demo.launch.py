import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1. Tải cấu hình MoveIt
    moveit_config = MoveItConfigsBuilder("robot", package_name="robot_moveit").to_moveit_configs()

    # Lấy đường dẫn file
    pkg_share = get_package_share_directory("robot_moveit")
    rviz_config_file = os.path.join(pkg_share, "config", "moveit.rviz")
    ros2_controllers_path = os.path.join(pkg_share, "config", "ros2_controllers.yaml")

    # 2. Đọc và TRÍCH XUẤT tham số cho Controller Manager
    # Chúng ta chỉ lấy phần bên trong ['controller_manager']['ros__parameters']
    with open(ros2_controllers_path, 'r') as file:
        full_config = yaml.safe_load(file)
        # Sửa lỗi 'Type not defined': Đưa params ra đúng cấp độ node mong muốn
        controller_manager_params = full_config['controller_manager']['ros__parameters']

    # 3. Node Robot State Publisher
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # 4. Node Move Group

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


    # 5. Node RViz
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

    # 6. Node ROS2 Control
    # - Truyền Dict đã trích xuất (Fix lỗi Type not defined)
    # - Truyền URDF (Fix lỗi phần cứng)
    # - KHÔNG truyền file path (Fix lỗi trailing flag)
    node_ros2_control = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[moveit_config.robot_description, controller_manager_params],
        output="screen",
    )

    # 7. Các Spawner
    # Truyền file path (-p) để controller con tự đọc cấu hình của riêng nó (PID, joints...)
    
    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager", "-p", ros2_controllers_path],
        output="screen",
    )

    spawn_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "-c", "/controller_manager", "-p", ros2_controllers_path],
        output="screen",
    )

    spawn_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "-c", "/controller_manager", "-p", ros2_controllers_path],
        output="screen",
    )

    # 8. Static TF
    node_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
    )

    return LaunchDescription([
        node_static_tf,
        node_robot_state_publisher,
        node_move_group,
        node_rviz,
        node_ros2_control,
        spawn_jsb,
        spawn_arm,
        spawn_gripper,
    ])