#ifndef MAP_MANAGER_VOXBLOX_IMPL_H_
#define MAP_MANAGER_VOXBLOX_IMPL_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <voxblox/interpolator/interpolator.h>
#include <voxblox/utils/layer_utils.h>
#include <voxblox/utils/planning_utils.h>
#include <voxblox/utils/planning_utils_inl.h>
#include <voxblox_ros/esdf_server.h>
#include <voxblox_ros/ros_params.h>
#include <voxblox_ros/transformer.h>

#include "planner_common/map_manager.h"
#include "planner_common/params.h"

#define use_tsdf

#ifdef use_tsdf
typedef voxblox::TsdfServer MapManagerVoxbloxServer;
typedef voxblox::TsdfVoxel MapManagerVoxbloxVoxel;
#else
typedef voxblox::EsdfServer MapManagerVoxbloxServer;
typedef voxblox::EsdfVoxel MapManagerVoxbloxVoxel;
#endif

#ifdef _GLIBCXX_PARALLEL
#include <parallel/algorithm>
#endif

namespace voxblox {
namespace utils {

template <typename VoxelType>
void getBoxAroundPoint(const Layer<VoxelType>& layer, const Point& center,
                       const Eigen::Vector3d& bounding_box_size,
                       HierarchicalIndexMap* block_voxel_list) {
  CHECK_NOTNULL(block_voxel_list);
  float voxel_size = layer.voxel_size();
  float voxel_size_inv = 1.0 / voxel_size;
  int voxels_per_side = layer.voxels_per_side();

  const GlobalIndex center_index =
      getGridIndexFromPoint<GlobalIndex>(center, voxel_size_inv);
  const Eigen::Vector3f halfrange_in_voxels =
      (bounding_box_size / 2 / voxel_size).cast<FloatingPoint>();

  for (FloatingPoint x = -halfrange_in_voxels[0]; x <= halfrange_in_voxels[0];
       x++) {
    for (FloatingPoint y = -halfrange_in_voxels[1]; y <= halfrange_in_voxels[1];
         y++) {
      for (FloatingPoint z = -halfrange_in_voxels[2];
           z <= halfrange_in_voxels[2]; z++) {
        Point point_voxel_space(x, y, z);

        GlobalIndex voxel_offset_index(std::floor(point_voxel_space.x()),
                                       std::floor(point_voxel_space.y()),
                                       std::floor(point_voxel_space.z()));
        BlockIndex block_index;
        VoxelIndex voxel_index;

        getBlockAndVoxelIndexFromGlobalVoxelIndex(
            voxel_offset_index + center_index, voxels_per_side, &block_index,
            &voxel_index);
        (*block_voxel_list)[block_index].push_back(voxel_index);
      }
    }
  }
}

template <typename VoxelType>
void getAndAllocateBoxAroundPoint(const Point& center,
                                  const Eigen::Vector3d& bounding_box_size,
                                  Layer<VoxelType>* layer,
                                  HierarchicalIndexMap* block_voxel_list) {
  CHECK_NOTNULL(layer);
  CHECK_NOTNULL(block_voxel_list);
  getBoxAroundPoint(*layer, center, bounding_box_size, block_voxel_list);
  for (auto it = block_voxel_list->begin(); it != block_voxel_list->end();
       ++it) {
    layer->allocateBlockPtrByIndex(it->first);
  }
}

}  // namespace utils
}  // namespace voxblox

namespace std {
template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
  size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
    size_t seed = 0;
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
      Scalar elem = *(matrix.data() + i);
      seed ^=
          std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
}  // namespace std

template <typename SDFServerType, typename SDFVoxelType>
class MapManagerVoxblox : public MapManager {
 public:
  explicit MapManagerVoxblox(rclcpp::Node* node);

  double getResolution() const override;
  bool getStatus() const override;
  VoxelStatus getVoxelStatus(const Eigen::Vector3d& position) const override;

  VoxelStatus getRayStatus(const Eigen::Vector3d& view_point,
                           const Eigen::Vector3d& voxel_to_test,
                           bool stop_at_unknown_voxel) const override;
  VoxelStatus getRayStatus(const Eigen::Vector3d& view_point,
                           const Eigen::Vector3d& voxel_to_test,
                           bool stop_at_unknown_voxel,
                           Eigen::Vector3d& end_voxel,
                           double& tsdf_dist) const override;
  VoxelStatus getBoxStatus(const Eigen::Vector3d& center,
                           const Eigen::Vector3d& size,
                           bool stop_at_unknown_voxel) const override;

  float getVoxelDistance(const Eigen::Vector3d& center) const;
  double getPointDistance(const Eigen::Vector3d& point) const;

  VoxelStatus getPathStatus(const Eigen::Vector3d& start,
                            const Eigen::Vector3d& end,
                            const Eigen::Vector3d& box_size,
                            bool stop_at_unknown_voxel) const override;

  bool augmentFreeBox(const Eigen::Vector3d& position,
                      const Eigen::Vector3d& box_size) override;
  void getScanStatus(
      Eigen::Vector3d& pos, std::vector<Eigen::Vector3d>& multiray_endpoints,
      std::tuple<int, int, int>& gain_log,
      std::vector<std::pair<Eigen::Vector3d, VoxelStatus>>& voxel_log,
      SensorParamsBase& sensor_params) override;

  void getScanStatusIterative(
      Eigen::Vector3d& pos, std::vector<Eigen::Vector3d>& multiray_endpoints,
      std::tuple<int, int, int>& gain_log,
      std::vector<std::pair<Eigen::Vector3d, VoxelStatus>>& voxel_log,
      SensorParamsBase& sensor_params);

  void augmentFreeFrustum() override;

  void getFreeSpacePointCloud(std::vector<Eigen::Vector3d>, StateVec,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr);

  void extractLocalMap(const Eigen::Vector3d& center,
                       const Eigen::Vector3d& bounding_box_size,
                       std::vector<Eigen::Vector3d>& occupied_voxels,
                       std::vector<Eigen::Vector3d>& free_voxels) override;

  void extractLocalMapAlongAxis(
      const Eigen::Vector3d& center, const Eigen::Vector3d& axis,
      const Eigen::Vector3d& bounding_box_size,
      std::vector<Eigen::Vector3d>& occupied_voxels,
      std::vector<Eigen::Vector3d>& free_voxels) override;

  void getLocalPointcloud(const Eigen::Vector3d& center, const double& range,
                          const double& yaw,
                          pcl::PointCloud<pcl::PointXYZI>& pcl,
                          bool include_unknown_voxels = false);
  void getLocalPointcloud(const Eigen::Vector3d& center, const double& range,
                          const double& yaw, const Eigen::Vector2d& z_limits,
                          pcl::PointCloud<pcl::PointXYZI>& pcl,
                          bool include_unknown_voxels = false);

  double getTruncationDistance() {
    return tsdf_integrator_config_.default_truncation_distance;
  }

  void resetMap() override { sdf_server_.clear(); }

  pcl::PointXYZI eigenVec3dToPCLPoint(Eigen::Vector3d& vec) const;

  void setRaycastingParams(bool nonuniform_ray_cast,
                           double ray_cast_step_size_multiplier) {
    nonuniform_ray_cast_ = nonuniform_ray_cast;
    ray_cast_step_size_multiplier_ = ray_cast_step_size_multiplier;
  }

  void setRobotRadius(double robot_radius) { robot_radius_ = robot_radius; }

 private:
  VoxelStatus getBoxStatusInVoxels(
      const voxblox::LongIndex& bounding_box_center,
      const voxblox::AnyIndex& bounding_box_voxels,
      bool stop_at_unknown_voxel) const;

  voxblox::Layer<SDFVoxelType>* getSDFLayer();
  bool checkUnknownStatus(const SDFVoxelType*) const;
  void clearIfUnknown(SDFVoxelType& voxel);

  SDFServerType sdf_server_;
  voxblox::Layer<SDFVoxelType>* sdf_layer_;

  voxblox::Interpolator<SDFVoxelType>* interpolator_;

  float occupancy_distance_voxelsize_factor_;

  voxblox::TsdfIntegratorBase::Config tsdf_integrator_config_;
  voxblox::EsdfIntegrator::Config esdf_integrator_config_;

  bool nonuniform_ray_cast_ = true;
  double ray_cast_step_size_multiplier_ = 1.0;

  double robot_radius_;
};

// Helper.
template class MapManagerVoxblox<MapManagerVoxbloxServer,
                                 MapManagerVoxbloxVoxel>;

#endif
