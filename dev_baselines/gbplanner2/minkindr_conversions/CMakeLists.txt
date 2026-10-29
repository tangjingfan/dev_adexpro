cmake_minimum_required(VERSION 3.8)
project(minkindr_conversions)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(minkindr REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)

# Header-only library.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} INTERFACE
  Eigen3::Eigen
  glog::glog
  minkindr::minkindr)
ament_target_dependencies(${PROJECT_NAME} INTERFACE
  geometry_msgs
  rclcpp
  tf2)

install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 glog geometry_msgs minkindr rclcpp tf2)
ament_package()
