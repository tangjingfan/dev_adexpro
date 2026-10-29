cmake_minimum_required(VERSION 3.8)
project(voxblox)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(Protobuf REQUIRED)
find_package(minkindr REQUIRED)

############
# PROTOBUF #
############
# Upstream used catkin's PROTOBUF_CATKIN_GENERATE_CPP2 helper. ROS 2 builds use
# the upstream FindProtobuf module. Generated .pb.{h,cc} files are placed in
# ${CMAKE_CURRENT_BINARY_DIR}/proto/voxblox/ so downstream includes of the form
#   #include "./Block.pb.h"  (from voxblox/core/block.h)
# resolve via the include directory below.
set(PROTO_FILES
  proto/voxblox/Block.proto
  proto/voxblox/Layer.proto
)

# Tell protoc to look for imports relative to the proto/ root and to emit into
# a parallel tree under the build dir.
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# We invoke protoc manually because protobuf_generate_cpp() does not honor a
# custom --proto_path easily across CMake versions, and upstream layout
# expects generated files to live under proto/voxblox/.
set(PROTO_SRCS)
set(PROTO_HDRS)
foreach(_proto IN LISTS PROTO_FILES)
  get_filename_component(_proto_we ${_proto} NAME_WE)
  set(_out_src "${PROTO_GEN_DIR}/voxblox/${_proto_we}.pb.cc")
  set(_out_hdr "${PROTO_GEN_DIR}/voxblox/${_proto_we}.pb.h")
  add_custom_command(
    OUTPUT  ${_out_src} ${_out_hdr}
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
            --cpp_out=${PROTO_GEN_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/proto
            ${CMAKE_CURRENT_SOURCE_DIR}/${_proto}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_proto}
    COMMENT "Generating C++ protobuf for ${_proto}"
    VERBATIM
  )
  list(APPEND PROTO_SRCS ${_out_src})
  list(APPEND PROTO_HDRS ${_out_hdr})
endforeach()

add_library(${PROJECT_NAME}_proto ${PROTO_SRCS})
target_include_directories(${PROJECT_NAME}_proto PUBLIC
  "$<BUILD_INTERFACE:${PROTO_GEN_DIR}>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME}_proto PUBLIC ${Protobuf_LIBRARIES})

####################
# SET SOURCE FILES #
####################
set(${PROJECT_NAME}_SRCS
  src/alignment/icp.cc
  src/core/block.cc
  src/core/esdf_map.cc
  src/core/tsdf_map.cc
  src/integrator/esdf_integrator.cc
  src/integrator/esdf_occ_integrator.cc
  src/integrator/integrator_utils.cc
  src/integrator/intensity_integrator.cc
  src/integrator/tsdf_integrator.cc
  src/io/mesh_ply.cc
  src/io/sdf_ply.cc
  src/mesh/marching_cubes.cc
  src/simulation/objects.cc
  src/simulation/simulation_world.cc
  src/utils/camera_model.cc
  src/utils/evaluation_utils.cc
  src/utils/layer_utils.cc
  src/utils/neighbor_tools.cc
  src/utils/protobuf_utils.cc
  src/utils/timing.cc
  src/utils/voxel_utils.cc
)

#############
# LIBRARIES #
#############
add_library(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<BUILD_INTERFACE:${PROTO_GEN_DIR}>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${PROJECT_NAME}_proto
  Eigen3::Eigen
  glog::glog
  gflags
  minkindr::minkindr
  ${Protobuf_LIBRARIES})

############
# BINARIES #
############
add_executable(tsdf_to_esdf test/tsdf_to_esdf.cc)
target_link_libraries(tsdf_to_esdf ${PROJECT_NAME})

add_executable(test_load_esdf test/test_load_esdf.cc)
target_link_libraries(test_load_esdf ${PROJECT_NAME})

###########
# INSTALL #
###########
install(
  DIRECTORY include/
  DESTINATION include)

install(
  FILES ${PROTO_HDRS}
  DESTINATION include/voxblox)

install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_proto
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(
  TARGETS tsdf_to_esdf test_load_esdf
  RUNTIME DESTINATION lib/${PROJECT_NAME})

#########
# TESTS #
#########
# NOTE: Upstream tests are gated behind BUILD_TESTING and depend on the eigen-
# catkin / eigen-checks ASL test helpers. They are intentionally disabled here.
# Re-enable by vendoring those helpers and setting VOXBLOX_ENABLE_TESTS=ON.
if(BUILD_TESTING AND VOXBLOX_ENABLE_TESTS)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_approx_hash_array  test/test_approx_hash_array.cc)
  target_link_libraries(test_approx_hash_array  ${PROJECT_NAME})
  ament_add_gtest(test_tsdf_map           test/test_tsdf_map.cc)
  target_link_libraries(test_tsdf_map           ${PROJECT_NAME})
  ament_add_gtest(test_protobuf           test/test_protobuf.cc)
  target_link_libraries(test_protobuf           ${PROJECT_NAME})
  ament_add_gtest(test_tsdf_interpolator  test/test_tsdf_interpolator.cc)
  target_link_libraries(test_tsdf_interpolator  ${PROJECT_NAME})
  ament_add_gtest(test_layer              test/test_layer.cc)
  target_link_libraries(test_layer              ${PROJECT_NAME})
  ament_add_gtest(test_merge_integration  test/test_merge_integration.cc)
  target_link_libraries(test_merge_integration  ${PROJECT_NAME})
  ament_add_gtest(test_layer_utils        test/test_layer_utils.cc)
  target_link_libraries(test_layer_utils        ${PROJECT_NAME})
  ament_add_gtest(test_sdf_integrators    test/test_sdf_integrators.cc)
  target_link_libraries(test_sdf_integrators    ${PROJECT_NAME})
  ament_add_gtest(test_bucket_queue       test/test_bucket_queue.cc)
  target_link_libraries(test_bucket_queue       ${PROJECT_NAME})
  ament_add_gtest(test_clear_spheres      test/test_clear_spheres.cc)
  target_link_libraries(test_clear_spheres      ${PROJECT_NAME})
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 glog gflags Protobuf minkindr)
ament_export_include_directories(include)
ament_package()
