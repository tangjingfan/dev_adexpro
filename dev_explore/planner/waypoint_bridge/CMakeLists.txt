cmake_minimum_required(VERSION 3.8)
project(waypoint_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-O3 -march=native)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(prob_msgs REQUIRED)

add_executable(waypoint_bridge src/waypoint_bridge.cpp)
ament_target_dependencies(waypoint_bridge rclcpp nav_msgs geometry_msgs prob_msgs)

install(TARGETS waypoint_bridge DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
