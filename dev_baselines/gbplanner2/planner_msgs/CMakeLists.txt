cmake_minimum_required(VERSION 3.8)
project(planner_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/Edge.msg
  msg/Graph.msg
  msg/PlanningBound.msg
  msg/PlanningMode.msg
  msg/Vertex.msg
  srv/PlannerSrv.srv
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs)
ament_package()
