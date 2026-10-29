cmake_minimum_required(VERSION 3.8)
project(minkindr)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

# Header-only library.
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} INTERFACE
  Eigen3::Eigen
  glog::glog)

install(
  DIRECTORY include/
  DESTINATION include)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME})

# NOTE: Upstream tests depend on eigen-checks (a ROS1 ASL package not ported
# to ROS 2). Tests are intentionally disabled here. If you need them, vendor
# eigen-checks under test/eigen-checks and re-enable below.
if(BUILD_TESTING AND MINKINDR_ENABLE_TESTS)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Threads REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_tests
    test/test-main.cc
    test/test-rotation.cc
    test/test-sim3.cc
    test/test-transformation.cc
    test/test-transformation-2d.cc)
  target_include_directories(${PROJECT_NAME}_tests PRIVATE test)
  target_link_libraries(${PROJECT_NAME}_tests
    ${PROJECT_NAME}
    Threads::Threads)
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 glog)
ament_package()
