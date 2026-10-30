cmake_minimum_required(VERSION 3.8)
project(global_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-O3 -march=native)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(prob_msgs REQUIRED)
find_package(rpv_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

include_directories(include)

set(SUPPORT_SOURCES
  src/avltree.cpp
  src/binary_set.cpp
  src/graph.cpp
  src/search_rpv.cpp
)

add_executable(global_planner_node
  src/global_planner.cpp
  src/global_planner_frontiers.cpp
  src/global_planner_reachability.cpp
  src/global_planner_planning.cpp
  src/global_planner_scheduling.cpp
  src/global_planner_node.cpp
  ${SUPPORT_SOURCES}
)

ament_target_dependencies(global_planner_node
  geometry_msgs
  nav_msgs
  prob_msgs
  rpv_msgs
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  visualization_msgs
)

install(TARGETS
  global_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
