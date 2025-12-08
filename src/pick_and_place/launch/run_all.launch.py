import os
import yaml
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, RegisterEventHandler, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
import xacro

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():
    pkg_robot = get_package_share_directory('robot')
    pkg_robot_moveit = get_package_share_directory('robot_moveit')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    
    urdf_file_path = os.path.join(pkg_robot, 'urdf', 'robot.urdf.xacro')
    doc = xacro.process_file(urdf_file_path, mappings={'use_sim': 'true'})
    robot_description_content = doc.toxml()

    # Load OMPL Config
    ompl_planning_yaml = load_yaml('robot_moveit', 'config/ompl_planning.yaml')
    
    moveit_config = (
        MoveItConfigsBuilder("robot", package_name="robot_moveit")
        .robot_description(file_path=urdf_file_path)
        .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    
    if ompl_planning_yaml:
        moveit_config.planning_pipelines['ompl'] = ompl_planning_yaml

    # 1. Move Group (QUAN TRỌNG: use_sim_time=True)
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"default_planning_pipeline": "ompl"},
            {'use_sim_time': True} # <--- SỬA LỖI LỆCH GIỜ TẠI ĐÂY
        ],
    )

    # 2. RViz (Cũng cần use_sim_time)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(pkg_robot_moveit, "config", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            {'use_sim_time': True} # <--- RViz cũng phải đồng bộ giờ
        ],
    )

    # 3. Gazebo & Bridge
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_robot, 'launch', 'gazebo.launch.py'))
    )
    
    spawn_entity = Node(package='ros_gz_sim', executable='create', arguments=['-topic', 'robot_description', '-name', 'robot', '-z', '1.05'], output='screen')
    
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', arguments=['/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image', '/model/robot/detachable_joint/toggle@std_msgs/msg/Empty]gz.msgs.Empty'], output='screen')
    
    camera_viewer = Node(package='rqt_image_view', executable='rqt_image_view', arguments=['/camera/image_raw'], output='screen')

    # 4. Code Gắp (Chạy với use_sim_time)
    pick_place_script = os.path.expanduser('~/robot_ws/src/pick_and_place/pick_and_place/simple_pick_place.py')
    pick_place_node = TimerAction(
        period=15.0,
        actions=[
            ExecuteProcess(
                cmd=['python3', pick_place_script, '--ros-args', '-p', 'use_sim_time:=true'], # <--- Ép code gắp dùng giờ Sim
                output='screen'
            )
        ]
    )

    node_robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen', parameters=[{'robot_description': robot_description_content, 'use_sim_time': True}]
    )
    
    joint_state_broadcaster = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"])
    arm_controller = Node(package="controller_manager", executable="spawner", arguments=["arm_controller"])
    gripper_controller = Node(package="controller_manager", executable="spawner", arguments=["gripper_controller"])

    return LaunchDescription([
        gazebo, bridge, camera_viewer, node_robot_state_publisher, spawn_entity,
        RegisterEventHandler(event_handler=OnProcessExit(target_action=spawn_entity, on_exit=[joint_state_broadcaster])),
        RegisterEventHandler(event_handler=OnProcessExit(target_action=joint_state_broadcaster, on_exit=[arm_controller, gripper_controller])),
        TimerAction(period=5.0, actions=[run_move_group_node]),
        TimerAction(period=7.0, actions=[rviz_node]),
        pick_place_node
    ])
