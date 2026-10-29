cmake_minimum_required(VERSION 3.5)
project(prob_rviz_plugins)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  add_compile_options(-O3 -march=native)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Qt5 REQUIRED COMPONENTS Widgets)
set(CMAKE_AUTOMOC ON)

set(prob_rviz_plugins_headers_to_moc
  include/prob_rviz_plugins/prob_grid_display.hpp
  include/prob_rviz_plugins/prediction_panel.hpp
)

foreach(header "${prob_rviz_plugins_headers_to_moc}")
  qt5_wrap_cpp(prob_rviz_plugins_moc_files "${header}")
endforeach()

ament_auto_add_library(${PROJECT_NAME} SHARED
  ${prob_rviz_plugins_moc_files}
  src/prob_grid_display.cpp
  src/prediction_panel.cpp
)

target_link_libraries(${PROJECT_NAME}
  Qt5::Widgets
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${Qt5Widgets_INCLUDE_DIRS}
)

target_compile_definitions(${PROJECT_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

if(${rclcpp_VERSION_MAJOR} VERSION_LESS "3.0.0")
  target_compile_definitions(${PROJECT_NAME} PRIVATE "FOXY")
endif()

pluginlib_export_plugin_description_file(rviz_common plugins_description.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
