import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- 1. KHAI BÁO THAM SỐ ---
    is_sim_arg = DeclareLaunchArgument(
        "is_sim",
        default_value="False",
        description="True neu chay Gazebo, False neu chay Robot that"
    )
    is_sim = LaunchConfiguration("is_sim")

    # --- 2. ĐƯỜNG DẪN CONFIG ---
    # Lưu ý: Bạn đang để config controller trong robot_moveit
    moveit_pkg = get_package_share_directory("robot_moveit")
    description_pkg = get_package_share_directory("robot_description")
    
    xacro_file = os.path.join(description_pkg, "urdf", "robot.urdf.xacro")
    controllers_file = os.path.join(moveit_pkg, "config", "ros2_controllers.yaml")

    # --- 3. LOAD ROBOT DESCRIPTION ---
    robot_description_content = Command([
        "xacro ", xacro_file,
        " use_sim:=", is_sim 
    ])
    
    robot_description = {"robot_description": robot_description_content}

    # --- 4. ROBOT STATE PUBLISHER (Dùng chung cho cả Sim và Real) ---
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": is_sim}],
    )

    # --- 5. ROS2 CONTROL NODE (CHỈ CHẠY KHI KHÔNG PHẢI SIM) ---
    # Nếu là Sim, Gazebo sẽ tự load plugin controller, ta không chạy node này
    node_ros2_control = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, controllers_file],
        output="screen",
        condition=UnlessCondition(is_sim) 
    )

    # --- 6. SPAWNERS (Dùng chung) ---
    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    spawn_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    spawn_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    
    # --- 7. STATIC TF (Optional: Nếu robot thật cần tf cố định ra world) ---
    node_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base_link"],
        condition=UnlessCondition(is_sim)
    )

    return LaunchDescription([
        is_sim_arg,
        node_static_tf,
        node_robot_state_publisher,
        node_ros2_control,
        spawn_jsb,
        spawn_arm,
        spawn_gripper,
    ])