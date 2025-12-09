import os
from launch import LaunchDescription
from launch.substitutions import Command, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('robot_description')
    default_model_path = os.path.join(pkg_share, 'urdf/robot.urdf.xacro')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/config.rviz')

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': ParameterValue(
                # --- SỬA Ở ĐÂY: Đổi 'cat ' thành 'xacro ' ---
                Command(['xacro ', LaunchConfiguration('model')]),
                value_type=str
            )
        }]
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', default_value=default_model_path,
            description='Absolute path to robot urdf file'
        ),
        DeclareLaunchArgument(
            'rvizconfig', default_value=default_rviz_config_path,
            description='Absolute path to rviz config file'
        ),
        joint_state_publisher_gui_node,
        robot_state_publisher_node,
        rviz_node
    ])
