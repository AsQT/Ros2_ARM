import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    # 1. Khai báo đường dẫn
    pkg_description = get_package_share_directory('robot_description')
    pkg_moveit = get_package_share_directory('robot_moveit')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')
    
    # --- KHẮC PHỤC CHÍNH: Lấy đường dẫn tuyệt đối (String) ---
    # Thay vì dùng PathJoinSubstitution (dễ gây lỗi khi pass qua spawner), ta dùng os.path.join
    ros2_controllers_path = os.path.join(pkg_moveit, "config", "ros2_controllers.yaml")

    # File World
    world_file = os.path.join(pkg_description, "worlds", "arm_on_the_table.sdf")

    # 2. Thiết lập biến môi trường
    install_dir = os.path.dirname(pkg_description) 
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=install_dir
    )

    # 3. Xử lý Robot Description (XACRO)
    xacro_file = os.path.join(pkg_description, "urdf", "robot.urdf.xacro")
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        xacro_file,
        " use_sim:=true", 
        " initial_positions_file:=", 
        os.path.join(pkg_moveit, "config", "initial_positions.yaml")
    ])
    
    robot_description = {"robot_description": robot_description_content}

    # 4. Node Robot State Publisher
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # 5. Khởi động Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 6. Spawn Robot
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'robot',
                   '-x', '0.0', '-y', '0.0', '-z', '1.02'], 
        output='screen',
    )

    # 7. Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        output='screen'
    )

    # 8. Spawn Controllers (ĐÃ THÊM LẠI --param-file VỚI ĐƯỜNG DẪN CHUẨN)
    # Bây giờ spawner sẽ chịu trách nhiệm nạp cấu hình, không phụ thuộc vào plugin Gazebo nữa
    
    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--param-file", ros2_controllers_path],
        output="screen",
    )

    spawn_arm = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--param-file", ros2_controllers_path],
        output="screen",
    )

    spawn_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--param-file", ros2_controllers_path],
        output="screen",
    )

    # Nhóm Controller
    controller_group = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[spawn_jsb, spawn_arm, spawn_gripper],
        )
    )

    return LaunchDescription([
        gz_resource_path,
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        bridge,
        controller_group
    ])
