import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare # <--- THÊM CÁI NÀY

def generate_launch_description():
    package_name = 'robot' 
    pkg_share = get_package_share_directory(package_name)

    # --- SỬA 1: TRỎ ĐÚNG FILE TỔNG (File có logic use_sim) ---
    # Thay vì 'urdf/robot.xacro', hãy trỏ vào file 'config/robot.urdf.xacro' trong gói moveit
    # Hoặc file robot.urdf.xacro trong gói robot (tùy bạn để đâu). 
    # Ở đây mình dùng cách an toàn nhất là trỏ vào file MoveIt Config đã sửa lúc nãy:
    default_model_path = PathJoinSubstitution([
        FindPackageShare("robot"), "urdf", "robot.urdf.xacro"
    ])
    
    model_arg = DeclareLaunchArgument(name='model', default_value=default_model_path)
    world_arg = DeclareLaunchArgument(name='world', default_value='arm_on_the_table.sdf')
    world_path = PathJoinSubstitution([pkg_share, 'worlds', LaunchConfiguration('world')])

    user_home = str(Path.home())
    custom_model_path = os.path.join(user_home, 'aws_robomaker_models', 'models')
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[os.path.join(pkg_share, 'worlds'), pathsep, str(os.path.dirname(pkg_share)), pathsep, custom_model_path]
    )

    # --- SỬA 2: THÊM use_sim:=true ĐỂ BẬT PLUGIN GAZEBO ---
    robot_description = ParameterValue(
        Command(['xacro ', LaunchConfiguration('model'), ' use_sim:=true']),
        value_type=str
    )

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}], output="screen"
    )
    
    joint_state_publisher = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        parameters=[{'use_sim_time': True}], output="screen"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={'gz_args': ['-r ', world_path]}.items()
    )

    # --- SỬA 3: XOAY ROBOT ĐỨNG DẬY (-P -1.57) ---
    spawn_entity = Node(
        package='ros_gz_sim', executable='create',
        arguments=[
            '-name', 'robot',
            '-topic', 'robot_description',
            '-x', '-0.4', '-y', '0.0', '-z', '1.05',
            '-R', '0.0', '-P', '0.0', '-Y', '0.0',
            '-J', 'joint_2', '0.0',  # Gài sẵn khớp 2 ngẩng lên
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock', '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model'],
        output='screen'
    )

    joint_state_broadcaster = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    robot_controller_spawner = Node(
        package="controller_manager", executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    start_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster, robot_controller_spawner],
        )
    )

    return LaunchDescription([
        gz_resource_path, model_arg, world_arg,
        gazebo, bridge, robot_state_publisher, joint_state_publisher,
        spawn_entity, start_controllers
    ])
