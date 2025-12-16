cmake_minimum_required(VERSION 3.16)
project(robot_hardware)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_library(robot_hardware_lib
  src/serial_port.cpp
  src/fas_protocol.cpp
  src/robot_hw_node.cpp
)

target_include_directories(robot_hardware_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(robot_hardware_lib
  rclcpp sensor_msgs std_msgs std_srvs
)
add_executable(robot_hardware_node src/main.cpp)
target_link_libraries(robot_hardware_node robot_hardware_lib)
ament_target_dependencies(robot_hardware_node
  rclcpp sensor_msgs std_msgs std_srvs
)

install(TARGETS robot_hardware_lib robot_hardware_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
