import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition

def generate_launch_description():
    # 1. Khai báo đường dẫn
    moveit_config_pkg = 'robot_moveit'
    robot_pkg = 'robot' # Package chứa file URDF gốc
    
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    # 2. Load Robot Description (Phải khớp với bên Gazebo: use_sim:=true)
    os.system(f"xacro {os.path.join(get_package_share_directory(robot_pkg), 'urdf', 'robot.urdf.xacro')} use_sim:=true > /tmp/robot_moveit.urdf")
    
    with open("/tmp/robot_moveit.urdf", "r") as f:
        robot_desc_content = f.read()
    
    robot_description = {"robot_description": robot_desc_content}

    # 3. Load URDF
    srdf_path = os.path.join(get_package_share_directory(moveit_config_pkg), 'config', 'robot.srdf')
    with open(srdf_path, 'r') as f:
        robot_description_semantic = {"robot_description_semantic": f.read()}

    # 4. Load Kinematics
    kinematics_yaml = os.path.join(get_package_share_directory(moveit_config_pkg), 'config', 'kinematics.yaml')

    # 5. Cấu hình MoveGroup
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics":  kinematics_yaml},
            {"use_sim_time": True}, # Quan trọng: Đồng bộ giờ với Gazebo
            os.path.join(get_package_share_directory(moveit_config_pkg), "config", "ompl_planning.yaml"),
            os.path.join(get_package_share_directory(moveit_config_pkg), "config", "moveit_controllers.yaml"),
        ],
    )

    # 6. Cấu hình RViz
    rviz_config_file = os.path.join(get_package_share_directory(moveit_config_pkg), 'config', 'moveit.rviz')
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"use_sim_time": True},
            {"robot_description_kinematics": kinematics_yaml},
        ],
    )

    return LaunchDescription([
        move_group_node,
        rviz_node
    ])
