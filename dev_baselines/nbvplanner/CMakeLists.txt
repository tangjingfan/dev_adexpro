cmake_minimum_required(VERSION 3.8)
project(nbvplanner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(nbvplanner_node
  src/nbvplanner_node.cpp
  src/nbvplanner.cpp
  src/rrt_tree.cpp
)

target_include_directories(nbvplanner_node
  PRIVATE
    include
    ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(nbvplanner_node
  rclcpp
  nav_msgs
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

install(TARGETS nbvplanner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
