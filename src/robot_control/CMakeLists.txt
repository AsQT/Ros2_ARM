cmake_minimum_required(VERSION 3.8)
project(robot_control)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

<<<<<<< HEAD
# 1) Install python executable so `ros2 run` can find it
=======
# --- THÊM ĐOẠN NÀY ĐỂ CÀI FILE PYTHON ---
>>>>>>> 050b7c084b52d14a3fc2d916d8c7158f042e78fc
install(PROGRAMS
  robot_control/slider_control.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME slider_control
)
# ----------------------------------------

<<<<<<< HEAD
# 2) Install launch/config (runtime resources)
=======
# 2. Cài đặt folder Config & Launch
>>>>>>> 050b7c084b52d14a3fc2d916d8c7158f042e78fc
install(DIRECTORY
  config
  launch
  test
  robot_control
  DESTINATION share/${PROJECT_NAME}
)

# 3) (Optional) if you want to ship extra python files (utils, etc.)
#    Put them under robot_control/ and keep this ON.
#    If you ONLY have slider_control.py, you can delete this block.
install(DIRECTORY
  robot_control
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# 4) Install tests only if you really need them at runtime (usually not needed)
# install(DIRECTORY
#   test
#   DESTINATION share/${PROJECT_NAME}
# )

ament_package()
