cmake_minimum_required(VERSION 3.8)
project(robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(urdf_tutorial REQUIRED) 
find_package(ament_lint_auto REQUIRED)

ament_lint_auto_find_test_dependencies() 

# --- PHẦN QUAN TRỌNG ĐÃ GỘP LẠI ---
install(DIRECTORY 
  launch
  meshes
  urdf
  rviz    # Đừng quên thư mục rviz
  worlds  # Thư mục chứa file world bạn cần
  config
  DESTINATION share/${PROJECT_NAME}
)
# ----------------------------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
