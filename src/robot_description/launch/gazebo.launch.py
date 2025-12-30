import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, RegisterEventHandler, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    pkg_description  = get_package_share_directory('robot_description')    
    pkg_share_parent = os.path.dirname(pkg_description)
    ros_gz_sim_pkg   = get_package_share_directory('ros_gz_sim') 
    world_file       = os.path.join(pkg_description, "worlds", "table", "arm_on_the_table.sdf")
    worlds_dir       = os.path.join(pkg_description, "worlds")
    xacro_file       = os.path.join(pkg_description, "urdf", "robot.urdf.xacro")
    
    robot_description_content = Command([
                                FindExecutable(name    = "xacro"),   " ",
                                xacro_file,  " use_sim:=true"  ])

    gz_resource_path = SetEnvironmentVariable(
        name         ='GZ_SIM_RESOURCE_PATH',
        value        =':'.join([
            pkg_share_parent,
            pkg_description,        
            worlds_dir,   
            os.environ.get('GZ_SIM_RESOURCE_PATH', '')
        ])
    )

    gz_model_path = SetEnvironmentVariable(
        name      ='GZ_SIM_MODEL_PATH',
        value     =':'.join([
            worlds_dir,       
            os.environ.get('GZ_SIM_MODEL_PATH', '')
        ])
    )

    #___________________ Thực thi ________________________________________
    # 1. Khởi động Gazebo với world file
    gazebo               = IncludeLaunchDescription(
        PythonLaunchDescriptionSource( 
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments = {'gz_args': f'-r {world_file}'}.items(), 
    )

    spawn_entity   = Node(
        package    ='ros_gz_sim',
        executable ='create',
        arguments  =[
            '-string', robot_description_content, 
            '-name', 'robot',
            '-x', '0.0', 
            '-y', '0.0', 
            '-z', '1.02',
            '-allow_renaming', 'true'
        ], 
        output     ='screen', 
    )

    spawn_wood     = Node(
        package    ='robot_description',
        executable ='random_wood_blocks.py',
        output     ='screen',
        parameters =[
            {'world': 'default'},
            {'count': 5},       
            {'seed' : 0},         
            {'x_min': 0.35},     
            {'x_max': 0.65},
            {'y_min': -0.20},    
            {'y_max': 0.20},
            {'z'    : 1.25}, 
        ]  
    )

    spawn_wood_after_robot = RegisterEventHandler(
        OnProcessExit(
            target_action  =spawn_entity,
            on_exit        =[TimerAction(period=1.0, actions=[spawn_wood])]
        )  
    )

    bridge         = Node(
        package    ='ros_gz_bridge',
        executable ='parameter_bridge',
        arguments  =[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/astra/rgb/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/astra/rgb/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/astra/depth/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/astra/depth/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
        ],
        output     ='screen' 
    )

    spawn_jsb      = Node(
        package    ="controller_manager",
        executable ="spawner",
        arguments  =["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        parameters =[{'use_sim_time': True}],
        output     ="screen", 
    )

    spawn_arm      = Node(
        package    ="controller_manager",
        executable ="spawner",
        arguments  =["arm_controller", "--controller-manager", "/controller_manager"],
        parameters =[{'use_sim_time': True}], 
        output     ="screen", 
    )

    spawn_gripper   = Node(
        package     ="controller_manager",
        executable  ="spawner",
        arguments   =["gripper_controller", "--controller-manager", "/controller_manager"],
        parameters  =[{'use_sim_time': True}],  
    )

    node_robot_state_publisher = Node(
                package        ='robot_state_publisher',
                executable     ='robot_state_publisher',
                output         ='screen',
                parameters     =[{
            'use_sim_time': True,
            'robot_description': robot_description_content 
        }], 
    )

    return LaunchDescription([
        gz_resource_path,
        gz_model_path,
        node_robot_state_publisher,
        gazebo,
        spawn_entity,
        bridge,
        spawn_jsb,
        spawn_arm,
        spawn_gripper,
        spawn_wood_after_robot,
    ])