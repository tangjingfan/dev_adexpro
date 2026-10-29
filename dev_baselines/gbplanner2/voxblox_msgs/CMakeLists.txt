cmake_minimum_required(VERSION 3.8)
project(voxblox_msgs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/Block.msg
  msg/Layer.msg
  msg/Mesh.msg
  msg/MeshBlock.msg
  msg/VoxelEvaluationDetails.msg
  srv/FilePath.srv
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs)
ament_package()
