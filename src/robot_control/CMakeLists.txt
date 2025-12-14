cmake_minimum_required(VERSION 3.22)
project(robot_control)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(controller_manager REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

include_directories(include)

add_library(robot_hardware SHARED
  src/robot_hardware.cpp
  src/plugin_export.cpp
)

ament_target_dependencies(robot_hardware
  rclcpp
  pluginlib
  hardware_interface
  rclcpp_lifecycle
)

# Export plugin description
pluginlib_export_plugin_description_file(hardware_interface robot_control_plugins.xml)

install(
  TARGETS robot_hardware
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
