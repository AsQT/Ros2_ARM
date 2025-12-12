from setuptools import find_packages, setup

package_name = 'robot_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='minhquang',
    maintainer_email='Tranminhquang617@outlook.com.vn',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Dòng quan trọng nhất để chạy lệnh 'ros2 run'
            'commander_gui = robot_gui.commander_gui:main',
        ],
    },
)
