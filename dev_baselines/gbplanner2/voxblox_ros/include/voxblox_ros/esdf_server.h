#ifndef VOXBLOX_ROS_ESDF_SERVER_H_
#define VOXBLOX_ROS_ESDF_SERVER_H_

#include <memory>
#include <string>

#include <voxblox/core/esdf_map.h>
#include <voxblox/integrator/esdf_integrator.h>
#include <voxblox_msgs/msg/layer.hpp>

#include "voxblox_ros/tsdf_server.h"

namespace voxblox {

class EsdfServer : public TsdfServer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit EsdfServer(rclcpp::Node* node);
  EsdfServer(rclcpp::Node* node, const EsdfMap::Config& esdf_config,
             const EsdfIntegrator::Config& esdf_integrator_config,
             const TsdfMap::Config& tsdf_config,
             const TsdfIntegratorBase::Config& tsdf_integrator_config,
             const MeshIntegratorConfig& mesh_config);
  virtual ~EsdfServer() {}

  void generateEsdfCallback(
      const std::shared_ptr<std_srvs::srv::Empty::Request> request,
      std::shared_ptr<std_srvs::srv::Empty::Response> response);

  void publishAllUpdatedEsdfVoxels();
  virtual void publishSlices();
  void publishTraversable();

  virtual void publishPointclouds();
  virtual void newPoseCallback(const Transformation& T_G_C);
  virtual void publishMap(bool reset_remote_map = false);
  virtual bool saveMap(const std::string& file_path);
  virtual bool loadMap(const std::string& file_path);

  void updateEsdfEvent();

  void updateEsdf();
  void updateEsdfBatch(bool full_euclidean = false);

  void esdfMapCallback(const voxblox_msgs::msg::Layer::SharedPtr layer_msg);

  inline std::shared_ptr<EsdfMap> getEsdfMapPtr() { return esdf_map_; }
  inline std::shared_ptr<const EsdfMap> getEsdfMapPtr() const {
    return esdf_map_;
  }

  bool getClearSphere() const { return clear_sphere_for_planning_; }
  void setClearSphere(bool clear_sphere_for_planning) {
    clear_sphere_for_planning_ = clear_sphere_for_planning;
  }
  float getEsdfMaxDistance() const;
  void setEsdfMaxDistance(float max_distance);
  float getTraversabilityRadius() const;
  void setTraversabilityRadius(float traversability_radius);

  void disableIncrementalUpdate() { incremental_update_ = false; }
  void enableIncrementalUpdate() { incremental_update_ = true; }

  virtual void clear();

 protected:
  void setupRos();

  // Publishers / subscribers / services.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      esdf_pointcloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr esdf_slice_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr traversable_pub_;
  rclcpp::Publisher<voxblox_msgs::msg::Layer>::SharedPtr esdf_map_pub_;
  rclcpp::Subscription<voxblox_msgs::msg::Layer>::SharedPtr esdf_map_sub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr generate_esdf_srv_;
  rclcpp::TimerBase::SharedPtr update_esdf_timer_;

  bool clear_sphere_for_planning_;
  bool publish_esdf_map_;
  bool publish_traversable_;
  float traversability_radius_;
  bool incremental_update_;
  int num_subscribers_esdf_map_;

  // ESDF maps.
  std::shared_ptr<EsdfMap> esdf_map_;
  std::unique_ptr<EsdfIntegrator> esdf_integrator_;
};

}  // namespace voxblox

#endif  // VOXBLOX_ROS_ESDF_SERVER_H_
