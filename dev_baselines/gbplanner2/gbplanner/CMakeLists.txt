cmake_minimum_required(VERSION 3.8)
project(gbplanner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters surface)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

find_package(kdtree REQUIRED)
find_package(planner_msgs REQUIRED)
find_package(planner_common REQUIRED)
find_package(voxblox REQUIRED)
find_package(voxblox_msgs REQUIRED)
find_package(voxblox_ros REQUIRED)

# Match the voxblox compile flags used by planner_common.
add_compile_definitions(COL_CHECK_METHOD=0 EDGE_CHECK_METHOD=0 RAY_CAST_METHOD=0)

add_library(${PROJECT_NAME}_lib
  src/gbplanner.cpp
  src/gbplanner_rviz.cpp
  src/rrg.cpp)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_lib PUBLIC
  Eigen3::Eigen
  glog::glog
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  planner_common::planner_common
  voxblox::voxblox
  voxblox_ros::voxblox_ros
  kdtree::kdtree)
ament_target_dependencies(${PROJECT_NAME}_lib PUBLIC
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  planner_msgs
  voxblox_msgs)

add_executable(gbplanner_node src/gbplanner_node.cpp)
target_link_libraries(gbplanner_node ${PROJECT_NAME}_lib)

install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  TARGETS gbplanner_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs visualization_msgs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions
  Eigen3 PCL OpenCV glog
  kdtree planner_msgs planner_common voxblox voxblox_msgs voxblox_ros)
ament_export_include_directories(include)
ament_package()
