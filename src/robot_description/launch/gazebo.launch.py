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
    pkg_control = get_package_share_directory('robot_control')
    ros_gz_sim_pkg = get_package_share_directory('ros_gz_sim')
    
    world_file = os.path.join(pkg_description, "worlds", "arm_on_the_table.sdf")
    
    # --- KHẮC PHỤC QUAN TRỌNG: TẠO URDF TRỰC TIẾP TẠI ĐÂY ---
    # Chúng ta tự chạy xacro ở đây để lấy chuỗi XML, đảm bảo use_sim:=true được bật
    xacro_file = os.path.join(pkg_description, "urdf", "robot.urdf.xacro")
    
    robot_description_content = Command([
        FindExecutable(name="xacro"), " ",
        xacro_file, 
        " use_sim:=true" 
    ])

    install_dir = os.path.dirname(pkg_description) 
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=install_dir
    )

    # 2. Khởi động Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 3. Spawn Robot vào Gazebo (DÙNG -string THAY VÌ -topic)
    # Đây là chìa khóa để sửa lỗi robot bị ngã
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', robot_description_content, 
            '-name', 'robot',
            '-x', '0.0', '-y', '0.0', '-z', '1.02',
            '-allow_renaming', 'true'
        ], 
        output='screen',
    )

    # 4. Bridge
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        output='screen'
    )

    # 5. BẬT CONTROLLER (Sau khi Spawn xong)
    launch_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_control, 'launch', 'control.launch.py')
        ),
        launch_arguments={'is_sim': 'True'}.items(),
    )   
    
    start_controllers_callback = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[launch_control],
        )
    )

    return LaunchDescription([
        gz_resource_path,
        gazebo,
        spawn_entity,
        bridge,
        start_controllers_callback
    ])