cmake_minimum_required(VERSION 3.8)
project(planner_common)

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
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(glog REQUIRED)
find_package(Boost REQUIRED)

find_package(kdtree REQUIRED)
find_package(planner_msgs REQUIRED)
find_package(minkindr REQUIRED)
find_package(minkindr_conversions REQUIRED)
find_package(voxblox REQUIRED)
find_package(voxblox_msgs REQUIRED)
find_package(voxblox_ros REQUIRED)

# voxblox collision/raycast/edge-check method selection. Match upstream values.
add_compile_definitions(COL_CHECK_METHOD=0 EDGE_CHECK_METHOD=0 RAY_CAST_METHOD=0)

set(${PROJECT_NAME}_SRCS
  src/graph.cpp
  src/graph_manager.cpp
  src/params.cpp
  src/random_sampler.cpp
  src/trajectory.cpp
  src/voxblox/voxblox_common_impl.cpp
  src/voxblox/voxblox_alt_impl.cpp
)

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
  ${PCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  Eigen3::Eigen
  glog::glog
  Boost::boost
  ${PCL_LIBRARIES}
  kdtree::kdtree
  minkindr::minkindr
  minkindr_conversions::minkindr_conversions
  voxblox::voxblox
  voxblox_ros::voxblox_ros)
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  planner_msgs
  voxblox_msgs)


install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs visualization_msgs
  tf2 tf2_ros tf2_geometry_msgs pcl_conversions
  Eigen3 PCL Boost glog
  kdtree planner_msgs minkindr minkindr_conversions
  voxblox voxblox_msgs voxblox_ros)
ament_export_include_directories(include)
ament_package()
