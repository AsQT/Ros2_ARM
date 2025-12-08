from setuptools import setup
import os
from glob import glob

package_name = 'pick_and_place'

setup(
    name=package_name,
    version='0.0.0',
    # Tự động tìm tất cả các gói con (quan trọng để tìm thấy thư mục pick_and_place)
    packages=[package_name], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Copy toàn bộ file launch
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='Pick and place demo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Tên lệnh = thư_mục_code.tên_file:tên_hàm_main
            'start_pick_place = pick_and_place.simple_pick_place:main',
        ],
    },
)
