cmake_minimum_required(VERSION 3.8)
project(voxblox_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)

find_package(minkindr REQUIRED)
find_package(minkindr_conversions REQUIRED)
find_package(voxblox REQUIRED)
find_package(voxblox_msgs REQUIRED)

set(SERVER_SRCS
  src/tsdf_server.cc
  src/esdf_server.cc
  src/transformer.cc
)

add_library(${PROJECT_NAME} ${SERVER_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  Eigen3::Eigen
  glog::glog
  gflags
  ${PCL_LIBRARIES}
  minkindr::minkindr
  minkindr_conversions::minkindr_conversions
  voxblox::voxblox)
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp
  builtin_interfaces
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  tf2
  tf2_ros
  tf2_eigen
  pcl_conversions
  voxblox_msgs)

# Executables.
add_executable(tsdf_server src/tsdf_server_node.cc)
target_link_libraries(tsdf_server ${PROJECT_NAME})

add_executable(esdf_server src/esdf_server_node.cc)
target_link_libraries(esdf_server ${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  TARGETS tsdf_server esdf_server
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp builtin_interfaces geometry_msgs sensor_msgs std_msgs std_srvs
  visualization_msgs tf2 tf2_ros tf2_eigen pcl_conversions
  Eigen3 glog gflags PCL minkindr minkindr_conversions voxblox voxblox_msgs)
ament_export_include_directories(include)
ament_package()
