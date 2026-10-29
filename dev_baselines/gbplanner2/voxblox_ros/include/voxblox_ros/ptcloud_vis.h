#ifndef VOXBLOX_ROS_PTCLOUD_VIS_H_
#define VOXBLOX_ROS_PTCLOUD_VIS_H_

#include <algorithm>
#include <functional>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <voxblox/core/common.h>
#include <voxblox/core/layer.h>
#include <voxblox/core/voxel.h>

#include "voxblox_ros/conversions.h"

namespace voxblox {

namespace ph = std::placeholders;

template <typename VoxelType>
using ShouldVisualizeVoxelColorFunctionType = std::function<bool(
    const VoxelType& voxel, const Point& coord, Color* color)>;

template <typename VoxelType>
using ShouldVisualizeVoxelIntensityFunctionType = std::function<bool(
    const VoxelType& voxel, const Point& coord, double* intensity)>;

template <typename VoxelType>
using ShouldVisualizeVoxelFunctionType =
    std::function<bool(const VoxelType& voxel, const Point& coord)>;  // NOLINT

template <typename VoxelType>
void createColorPointcloudFromLayer(
    const Layer<VoxelType>& layer,
    const ShouldVisualizeVoxelColorFunctionType<VoxelType>& vis_function,
    pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  pointcloud->clear();
  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);

  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  Color color;
  for (const BlockIndex& index : blocks) {
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord,
                       &color)) {
        pcl::PointXYZRGB point;
        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();
        point.r = color.r;
        point.g = color.g;
        point.b = color.b;
        pointcloud->push_back(point);
      }
    }
  }
}

template <typename VoxelType>
void createColorPointcloudFromLayer(
    const Layer<VoxelType>& layer,
    const ShouldVisualizeVoxelIntensityFunctionType<VoxelType>& vis_function,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  pointcloud->clear();
  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);

  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;

  double intensity = 0.0;
  for (const BlockIndex& index : blocks) {
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord,
                       &intensity)) {
        pcl::PointXYZI point;
        point.x = coord.x();
        point.y = coord.y();
        point.z = coord.z();
        point.intensity = intensity;
        pointcloud->push_back(point);
      }
    }
  }
}

template <typename VoxelType>
void createOccupancyBlocksFromLayer(
    const Layer<VoxelType>& layer,
    const ShouldVisualizeVoxelFunctionType<VoxelType>& vis_function,
    const std::string& frame_id,
    visualization_msgs::msg::MarkerArray* marker_array) {
  CHECK_NOTNULL(marker_array);
  size_t vps = layer.voxels_per_side();
  size_t num_voxels_per_block = vps * vps * vps;
  FloatingPoint voxel_size = layer.voxel_size();

  visualization_msgs::msg::Marker block_marker;
  block_marker.header.frame_id = frame_id;
  block_marker.ns = "occupied_voxels";
  block_marker.id = 0;
  block_marker.type = visualization_msgs::msg::Marker::CUBE_LIST;
  block_marker.scale.x = block_marker.scale.y = block_marker.scale.z =
      voxel_size;
  block_marker.action = visualization_msgs::msg::Marker::ADD;

  BlockIndexList blocks;
  layer.getAllAllocatedBlocks(&blocks);
  for (const BlockIndex& index : blocks) {
    const Block<VoxelType>& block = layer.getBlockByIndex(index);

    for (size_t linear_index = 0; linear_index < num_voxels_per_block;
         ++linear_index) {
      Point coord = block.computeCoordinatesFromLinearIndex(linear_index);
      if (vis_function(block.getVoxelByLinearIndex(linear_index), coord)) {
        geometry_msgs::msg::Point cube_center;
        cube_center.x = coord.x();
        cube_center.y = coord.y();
        cube_center.z = coord.z();
        block_marker.points.push_back(cube_center);
        std_msgs::msg::ColorRGBA color_msg;
        colorVoxbloxToMsg(rainbowColorMap((coord.z() + 2.5) / 5.0), &color_msg);
        block_marker.colors.push_back(color_msg);
      }
    }
  }
  marker_array->markers.push_back(block_marker);
}

// /Short-hand functions for visualizing different types of voxels.
inline bool visualizeNearSurfaceTsdfVoxels(const TsdfVoxel& voxel,
                                           const Point& /*coord*/,
                                           double surface_distance,
                                           Color* color) {
  CHECK_NOTNULL(color);
  constexpr float kMinWeight = 0;
  if (voxel.weight > kMinWeight &&
      std::abs(voxel.distance) < surface_distance) {
    *color = voxel.color;
    return true;
  }
  return false;
}

inline bool visualizeTsdfVoxels(const TsdfVoxel& voxel, const Point& /*coord*/,
                                Color* color) {
  CHECK_NOTNULL(color);
  constexpr float kMinWeight = 0;
  if (voxel.weight > kMinWeight) {
    *color = voxel.color;
    return true;
  }
  return false;
}

inline bool visualizeDistanceIntensityTsdfVoxels(const TsdfVoxel& voxel,
                                                 const Point& /*coord*/,
                                                 double* intensity) {
  CHECK_NOTNULL(intensity);
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight) {
    *intensity = voxel.distance;
    return true;
  }
  return false;
}

inline bool visualizeDistanceIntensityTsdfVoxelsNearSurface(
    const TsdfVoxel& voxel, const Point& /*coord*/, double surface_distance,
    double* intensity) {
  CHECK_NOTNULL(intensity);
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight &&
      std::abs(voxel.distance) < surface_distance) {
    *intensity = voxel.distance;
    return true;
  }
  return false;
}

inline bool visualizeDistanceIntensityTsdfVoxelsSlice(
    const TsdfVoxel& voxel, const Point& coord, unsigned int free_plane_index,
    FloatingPoint free_plane_val, FloatingPoint voxel_size, double* intensity) {
  CHECK_NOTNULL(intensity);
  constexpr float kMinWeight = 1e-3;

  if (std::abs(coord(free_plane_index) - free_plane_val) <=
      (voxel_size / 2.0 + kFloatEpsilon)) {
    if (voxel.weight > kMinWeight) {
      *intensity = voxel.distance;
      return true;
    }
  }
  return false;
}

inline bool visualizeDistanceIntensityEsdfVoxels(const EsdfVoxel& voxel,
                                                 const Point& /*coord*/,
                                                 double* intensity) {
  CHECK_NOTNULL(intensity);
  if (voxel.observed) {
    *intensity = voxel.distance;
    return true;
  }
  return false;
}

inline bool visualizeIntensityVoxels(const IntensityVoxel& voxel,
                                     const Point& /*coord*/,
                                     double* intensity) {
  constexpr float kMinWeight = 1e-3;

  CHECK_NOTNULL(intensity);
  if (voxel.weight > kMinWeight) {
    *intensity = voxel.intensity;
    return true;
  }
  return false;
}

inline bool visualizeDistanceIntensityEsdfVoxelsSlice(
    const EsdfVoxel& voxel, const Point& coord, unsigned int free_plane_index,
    FloatingPoint free_plane_val, FloatingPoint voxel_size, double* intensity) {
  CHECK_NOTNULL(intensity);

  if (std::abs(coord(free_plane_index) - free_plane_val) <=
      (voxel_size / 2.0 + kFloatEpsilon)) {
    if (voxel.observed) {
      *intensity = voxel.distance;
      return true;
    }
  }
  return false;
}

inline bool visualizeOccupiedTsdfVoxels(const TsdfVoxel& voxel,
                                        const Point& /*coord*/,
                                        const float min_distance = 0.0) {
  constexpr float kMinWeight = 1e-3;
  if (voxel.weight > kMinWeight && voxel.distance <= min_distance) {
    return true;
  }
  return false;
}

inline bool visualizeFreeEsdfVoxels(const EsdfVoxel& voxel,
                                    const Point& /*coord*/, float min_distance,
                                    double* intensity) {
  if (voxel.observed && voxel.distance >= min_distance) {
    *intensity = voxel.distance;
    return true;
  }
  return false;
}

inline bool visualizeOccupiedOccupancyVoxels(const OccupancyVoxel& voxel,
                                             const Point& /*coord*/) {
  const float kThresholdLogOccupancy = logOddsFromProbability(0.7);
  if (voxel.probability_log > kThresholdLogOccupancy) {
    return true;
  }
  return false;
}

inline void createSurfacePointcloudFromTsdfLayer(
    const Layer<TsdfVoxel>& layer, double surface_distance,
    pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<TsdfVoxel>(
      layer,
      std::bind(&visualizeNearSurfaceTsdfVoxels, ph::_1, ph::_2,
                surface_distance, ph::_3),
      pointcloud);
}

inline void createPointcloudFromTsdfLayer(
    const Layer<TsdfVoxel>& layer,
    pcl::PointCloud<pcl::PointXYZRGB>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<TsdfVoxel>(layer, &visualizeTsdfVoxels,
                                            pointcloud);
}

inline void createDistancePointcloudFromTsdfLayer(
    const Layer<TsdfVoxel>& layer,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<TsdfVoxel>(
      layer, &visualizeDistanceIntensityTsdfVoxels, pointcloud);
}

inline void createSurfaceDistancePointcloudFromTsdfLayer(
    const Layer<TsdfVoxel>& layer, double surface_distance,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<TsdfVoxel>(
      layer,
      std::bind(&visualizeDistanceIntensityTsdfVoxelsNearSurface, ph::_1,
                ph::_2, surface_distance, ph::_3),
      pointcloud);
}

inline void createDistancePointcloudFromEsdfLayer(
    const Layer<EsdfVoxel>& layer,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<EsdfVoxel>(
      layer, &visualizeDistanceIntensityEsdfVoxels, pointcloud);
}

inline void createFreePointcloudFromEsdfLayer(
    const Layer<EsdfVoxel>& layer, float min_distance,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<EsdfVoxel>(
      layer,
      std::bind(&visualizeFreeEsdfVoxels, ph::_1, ph::_2, min_distance, ph::_3),
      pointcloud);
}

inline void createIntensityPointcloudFromIntensityLayer(
    const Layer<IntensityVoxel>& layer,
    pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  createColorPointcloudFromLayer<IntensityVoxel>(
      layer, &visualizeIntensityVoxels, pointcloud);
}

inline void createDistancePointcloudFromTsdfLayerSlice(
    const Layer<TsdfVoxel>& layer, unsigned int free_plane_index,
    FloatingPoint free_plane_val, pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  if (std::remainder(free_plane_val, layer.voxel_size()) < kFloatEpsilon) {
    free_plane_val += layer.voxel_size() / 2.0;
  }

  createColorPointcloudFromLayer<TsdfVoxel>(
      layer,
      std::bind(&visualizeDistanceIntensityTsdfVoxelsSlice, ph::_1, ph::_2,
                free_plane_index, free_plane_val, layer.voxel_size(), ph::_3),
      pointcloud);
}

inline void createDistancePointcloudFromEsdfLayerSlice(
    const Layer<EsdfVoxel>& layer, unsigned int free_plane_index,
    FloatingPoint free_plane_val, pcl::PointCloud<pcl::PointXYZI>* pointcloud) {
  CHECK_NOTNULL(pointcloud);
  if (std::remainder(free_plane_val, layer.voxel_size()) < kFloatEpsilon) {
    free_plane_val += layer.voxel_size() / 2.0;
  }

  createColorPointcloudFromLayer<EsdfVoxel>(
      layer,
      std::bind(&visualizeDistanceIntensityEsdfVoxelsSlice, ph::_1, ph::_2,
                free_plane_index, free_plane_val, layer.voxel_size(), ph::_3),
      pointcloud);
}

inline void createOccupancyBlocksFromTsdfLayer(
    const Layer<TsdfVoxel>& layer, const std::string& frame_id,
    visualization_msgs::msg::MarkerArray* marker_array) {
  CHECK_NOTNULL(marker_array);
  createOccupancyBlocksFromLayer<TsdfVoxel>(
      layer,
      std::bind(visualizeOccupiedTsdfVoxels, ph::_1, ph::_2,
                layer.voxel_size()),
      frame_id, marker_array);
}

inline void createOccupancyBlocksFromOccupancyLayer(
    const Layer<OccupancyVoxel>& layer, const std::string& frame_id,
    visualization_msgs::msg::MarkerArray* marker_array) {
  CHECK_NOTNULL(marker_array);
  createOccupancyBlocksFromLayer<OccupancyVoxel>(
      layer, &visualizeOccupiedOccupancyVoxels, frame_id, marker_array);
}

}  // namespace voxblox

#endif  // VOXBLOX_ROS_PTCLOUD_VIS_H_
