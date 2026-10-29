#ifndef VOXBLOX_ROS_ROS_PARAMS_H_
#define VOXBLOX_ROS_ROS_PARAMS_H_

#include <string>

#include <rclcpp/rclcpp.hpp>

#include <voxblox/alignment/icp.h>
#include <voxblox/core/esdf_map.h>
#include <voxblox/core/tsdf_map.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox/integrator/tsdf_integrator.h>
#include <voxblox/mesh/mesh_integrator.h>

namespace voxblox {

// declareOrGet: ROS 2 forbids declaring a parameter twice on the same node.
// The original ROS 1 helpers below are nested (getEsdfMapConfigFromRosParam
// invokes getTsdfMapConfigFromRosParam, which would re-declare the same
// names). This wrapper makes the declare-then-read pattern idempotent.
template <typename T>
inline T declareOrGet(rclcpp::Node* node, const std::string& name,
                      const T& default_val) {
  if (node->has_parameter(name)) {
    T v;
    node->get_parameter(name, v);
    return v;
  }
  return node->declare_parameter<T>(name, default_val);
}

inline TsdfMap::Config getTsdfMapConfigFromRosParam(rclcpp::Node* node) {
  TsdfMap::Config tsdf_config;

  double voxel_size = tsdf_config.tsdf_voxel_size;
  int voxels_per_side = tsdf_config.tsdf_voxels_per_side;
  voxel_size = declareOrGet<double>(node, "tsdf_voxel_size", voxel_size);
  voxels_per_side =
      declareOrGet<int>(node, "tsdf_voxels_per_side", voxels_per_side);
  if (!isPowerOfTwo(voxels_per_side)) {
    RCLCPP_ERROR(node->get_logger(),
                 "voxels_per_side must be a power of 2, setting to default");
    voxels_per_side = tsdf_config.tsdf_voxels_per_side;
  }

  tsdf_config.tsdf_voxel_size = static_cast<FloatingPoint>(voxel_size);
  tsdf_config.tsdf_voxels_per_side = voxels_per_side;

  return tsdf_config;
}

inline ICP::Config getICPConfigFromRosParam(rclcpp::Node* node) {
  ICP::Config icp_config;

  icp_config.min_match_ratio = declareOrGet<double>(
      node, "icp_min_match_ratio", icp_config.min_match_ratio);
  icp_config.subsample_keep_ratio = declareOrGet<double>(
      node, "icp_subsample_keep_ratio", icp_config.subsample_keep_ratio);
  icp_config.mini_batch_size = declareOrGet<int>(
      node, "icp_mini_batch_size", icp_config.mini_batch_size);
  icp_config.refine_roll_pitch = declareOrGet<bool>(
      node, "icp_refine_roll_pitch", icp_config.refine_roll_pitch);
  icp_config.inital_translation_weighting = declareOrGet<double>(
      node, "icp_inital_translation_weighting",
      icp_config.inital_translation_weighting);
  icp_config.inital_rotation_weighting = declareOrGet<double>(
      node, "icp_inital_rotation_weighting",
      icp_config.inital_rotation_weighting);

  return icp_config;
}

inline TsdfIntegratorBase::Config getTsdfIntegratorConfigFromRosParam(
    rclcpp::Node* node) {
  TsdfIntegratorBase::Config integrator_config;
  integrator_config.voxel_carving_enabled = true;

  const TsdfMap::Config tsdf_config = getTsdfMapConfigFromRosParam(node);
  integrator_config.default_truncation_distance =
      tsdf_config.tsdf_voxel_size * 4;

  double truncation_distance = integrator_config.default_truncation_distance;
  double max_weight = integrator_config.max_weight;

  integrator_config.voxel_carving_enabled = declareOrGet<bool>(
      node, "voxel_carving_enabled", integrator_config.voxel_carving_enabled);
  truncation_distance =
      declareOrGet<double>(node, "truncation_distance", truncation_distance);
  integrator_config.max_ray_length_m = declareOrGet<double>(
      node, "max_ray_length_m", integrator_config.max_ray_length_m);
  integrator_config.min_ray_length_m = declareOrGet<double>(
      node, "min_ray_length_m", integrator_config.min_ray_length_m);
  max_weight = declareOrGet<double>(node, "max_weight", max_weight);
  integrator_config.use_const_weight = declareOrGet<bool>(
      node, "use_const_weight", integrator_config.use_const_weight);
  integrator_config.use_weight_dropoff = declareOrGet<bool>(
      node, "use_weight_dropoff", integrator_config.use_weight_dropoff);
  integrator_config.allow_clear = declareOrGet<bool>(
      node, "allow_clear", integrator_config.allow_clear);
  integrator_config.start_voxel_subsampling_factor = declareOrGet<double>(
      node, "start_voxel_subsampling_factor",
      integrator_config.start_voxel_subsampling_factor);
  integrator_config.max_consecutive_ray_collisions = declareOrGet<int>(
      node, "max_consecutive_ray_collisions",
      integrator_config.max_consecutive_ray_collisions);
  integrator_config.clear_checks_every_n_frames = declareOrGet<int>(
      node, "clear_checks_every_n_frames",
      integrator_config.clear_checks_every_n_frames);
  integrator_config.max_integration_time_s = declareOrGet<double>(
      node, "max_integration_time_s",
      integrator_config.max_integration_time_s);
  integrator_config.enable_anti_grazing = declareOrGet<bool>(
      node, "anti_grazing", integrator_config.enable_anti_grazing);
  integrator_config.use_sparsity_compensation_factor = declareOrGet<bool>(
      node, "use_sparsity_compensation_factor",
      integrator_config.use_sparsity_compensation_factor);
  integrator_config.sparsity_compensation_factor = declareOrGet<double>(
      node, "sparsity_compensation_factor",
      integrator_config.sparsity_compensation_factor);
  integrator_config.integration_order_mode = declareOrGet<std::string>(
      node, "integration_order_mode",
      integrator_config.integration_order_mode);

  integrator_config.default_truncation_distance =
      static_cast<float>(truncation_distance);
  integrator_config.max_weight = static_cast<float>(max_weight);

  return integrator_config;
}

inline EsdfMap::Config getEsdfMapConfigFromRosParam(rclcpp::Node* node) {
  EsdfMap::Config esdf_config;

  const TsdfMap::Config tsdf_config = getTsdfMapConfigFromRosParam(node);
  esdf_config.esdf_voxel_size = tsdf_config.tsdf_voxel_size;
  esdf_config.esdf_voxels_per_side = tsdf_config.tsdf_voxels_per_side;

  return esdf_config;
}

inline EsdfIntegrator::Config getEsdfIntegratorConfigFromRosParam(
    rclcpp::Node* node) {
  EsdfIntegrator::Config esdf_integrator_config;

  TsdfIntegratorBase::Config tsdf_integrator_config =
      getTsdfIntegratorConfigFromRosParam(node);

  esdf_integrator_config.min_distance_m =
      tsdf_integrator_config.default_truncation_distance / 2.0;

  esdf_integrator_config.full_euclidean_distance = declareOrGet<bool>(
      node, "esdf_euclidean_distance",
      esdf_integrator_config.full_euclidean_distance);
  esdf_integrator_config.max_distance_m = declareOrGet<double>(
      node, "esdf_max_distance_m", esdf_integrator_config.max_distance_m);
  esdf_integrator_config.min_distance_m = declareOrGet<double>(
      node, "esdf_min_distance_m", esdf_integrator_config.min_distance_m);
  esdf_integrator_config.default_distance_m = declareOrGet<double>(
      node, "esdf_default_distance_m",
      esdf_integrator_config.default_distance_m);
  esdf_integrator_config.min_diff_m = declareOrGet<double>(
      node, "esdf_min_diff_m", esdf_integrator_config.min_diff_m);
  esdf_integrator_config.clear_sphere_radius = declareOrGet<double>(
      node, "clear_sphere_radius",
      esdf_integrator_config.clear_sphere_radius);
  esdf_integrator_config.occupied_sphere_radius = declareOrGet<double>(
      node, "occupied_sphere_radius",
      esdf_integrator_config.occupied_sphere_radius);
  esdf_integrator_config.add_occupied_crust = declareOrGet<bool>(
      node, "esdf_add_occupied_crust",
      esdf_integrator_config.add_occupied_crust);

  if (esdf_integrator_config.default_distance_m <
      esdf_integrator_config.max_distance_m) {
    esdf_integrator_config.default_distance_m =
        esdf_integrator_config.max_distance_m;
  }

  return esdf_integrator_config;
}

inline MeshIntegratorConfig getMeshIntegratorConfigFromRosParam(
    rclcpp::Node* node) {
  MeshIntegratorConfig mesh_integrator_config;

  mesh_integrator_config.min_weight = declareOrGet<double>(
      node, "mesh_min_weight", mesh_integrator_config.min_weight);
  mesh_integrator_config.use_color = declareOrGet<bool>(
      node, "mesh_use_color", mesh_integrator_config.use_color);

  return mesh_integrator_config;
}

}  // namespace voxblox

#endif  // VOXBLOX_ROS_ROS_PARAMS_H_
