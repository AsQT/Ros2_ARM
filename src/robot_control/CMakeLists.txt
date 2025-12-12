cmake_minimum_required(VERSION 3.8)
project(robot_control)

# Tìm dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# --- THÊM ĐOẠN NÀY ĐỂ CÀI FILE PYTHON ---
install(PROGRAMS
  robot_control/slider_control.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME slider_control
)
# ----------------------------------------

# 2. Cài đặt folder Config & Launch
install(DIRECTORY
  config
  launch
  test
  robot_control
  DESTINATION share/${PROJECT_NAME}
)

ament_package()