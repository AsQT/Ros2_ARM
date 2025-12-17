from setuptools import setup
import os
from glob import glob

package_name = 'vision_tracking'

setup(
    name       =package_name,
    version    ='0.0.0',
    packages   =[package_name],
    data_files =[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), glob('resource/*')),
    ],
    
    install_requires =['setuptools'],
    zip_safe         =True,
    maintainer       ='User',
    maintainer_email ='user@todo.todo',
    description      ='ROS 2 package for combined YOLO and SIFT object tracking.',
    license          ='TODO: License declaration',
    tests_require    =['pytest'],
    
    entry_points={
        'console_scripts':[
            'sift_yolo_tracker_node = vision_tracking.sift_yolo_tracker_node:main',
            'yolo_publisher_node = vision_tracking.yolo_publisher_node:main',
            'test = vision_tracking.test:main',
        ],
    },
)