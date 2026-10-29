#include "voxblox_ros/esdf_server.h"

#include <chrono>

#include <pcl_conversions/pcl_conversions.h>

#include "voxblox_ros/conversions.h"
#include "voxblox_ros/ros_params.h"

namespace voxblox {

namespace {
template <typename PointT>
sensor_msgs::msg::PointCloud2 toPointCloud2Msg(
    const pcl::PointCloud<PointT>& cloud, const std::string& frame_id) {
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.frame_id = frame_id;
  return msg;
}
}  // namespace

EsdfServer::EsdfServer(rclcpp::Node* node)
    : EsdfServer(node, getEsdfMapConfigFromRosParam(node),
                 getEsdfIntegratorConfigFromRosParam(node),
                 getTsdfMapConfigFromRosParam(node),
                 getTsdfIntegratorConfigFromRosParam(node),
                 getMeshIntegratorConfigFromRosParam(node)) {}

EsdfServer::EsdfServer(rclcpp::Node* node, const EsdfMap::Config& esdf_config,
                       const EsdfIntegrator::Config& esdf_integrator_config,
                       const TsdfMap::Config& tsdf_config,
                       const TsdfIntegratorBase::Config& tsdf_integrator_config,
                       const MeshIntegratorConfig& mesh_config)
    : TsdfServer(node, tsdf_config, tsdf_integrator_config, mesh_config),
      clear_sphere_for_planning_(false),
      publish_esdf_map_(false),
      publish_traversable_(false),
      traversability_radius_(1.0),
      incremental_update_(true),
      num_subscribers_esdf_map_(0) {
  esdf_map_.reset(new EsdfMap(esdf_config));
  esdf_integrator_.reset(new EsdfIntegrator(esdf_integrator_config,
                                            tsdf_map_->getTsdfLayerPtr(),
                                            esdf_map_->getEsdfLayerPtr()));
  setupRos();
}

void EsdfServer::setupRos() {
  auto latched_qos = rclcpp::QoS(1).transient_local();

  esdf_pointcloud_pub_ =
      node_->create_publisher<sensor_msgs::msg::PointCloud2>("esdf_pointcloud",
                                                             latched_qos);
  esdf_slice_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
      "esdf_slice", latched_qos);
  traversable_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
      "traversable", latched_qos);

  esdf_map_pub_ = node_->create_publisher<voxblox_msgs::msg::Layer>(
      "esdf_map_out", rclcpp::QoS(1));

  esdf_map_sub_ = node_->create_subscription<voxblox_msgs::msg::Layer>(
      "esdf_map_in", 1,
      std::bind(&EsdfServer::esdfMapCallback, this, std::placeholders::_1));

  clear_sphere_for_planning_ = declareOrGet<bool>(
      node_, "clear_sphere_for_planning", clear_sphere_for_planning_);
  publish_esdf_map_ =
      declareOrGet<bool>(node_, "publish_esdf_map", publish_esdf_map_);
  publish_traversable_ =
      declareOrGet<bool>(node_, "publish_traversable", publish_traversable_);
  // traversability_radius_ is float; ROS 2 declare_parameter prefers double.
  double trav_radius =
      static_cast<double>(traversability_radius_);
  trav_radius =
      declareOrGet<double>(node_, "traversability_radius", trav_radius);
  traversability_radius_ = static_cast<float>(trav_radius);

  double update_esdf_every_n_sec =
      declareOrGet<double>(node_, "update_esdf_every_n_sec", 1.0);
  if (update_esdf_every_n_sec > 0.0) {
    update_esdf_timer_ = node_->create_wall_timer(
        std::chrono::duration<double>(update_esdf_every_n_sec),
        std::bind(&EsdfServer::updateEsdfEvent, this));
  }

  generate_esdf_srv_ = node_->create_service<std_srvs::srv::Empty>(
      "generate_esdf",
      std::bind(&EsdfServer::generateEsdfCallback, this,
                std::placeholders::_1, std::placeholders::_2));
}

void EsdfServer::publishAllUpdatedEsdfVoxels() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  createDistancePointcloudFromEsdfLayer(esdf_map_->getEsdfLayer(), &pointcloud);
  esdf_pointcloud_pub_->publish(toPointCloud2Msg(pointcloud, world_frame_));
}

void EsdfServer::publishSlices() {
  TsdfServer::publishSlices();

  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  constexpr int kZAxisIndex = 2;
  createDistancePointcloudFromEsdfLayerSlice(
      esdf_map_->getEsdfLayer(), kZAxisIndex, slice_level_, &pointcloud);
  esdf_slice_pub_->publish(toPointCloud2Msg(pointcloud, world_frame_));
}

void EsdfServer::generateEsdfCallback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/) {
  const bool clear_esdf = true;
  if (clear_esdf) {
    esdf_integrator_->updateFromTsdfLayerBatch();
  } else {
    const bool clear_updated_flag = true;
    esdf_integrator_->updateFromTsdfLayer(clear_updated_flag);
  }
  publishAllUpdatedEsdfVoxels();
  publishSlices();
}

void EsdfServer::updateEsdfEvent() { updateEsdf(); }

void EsdfServer::publishPointclouds() {
  publishAllUpdatedEsdfVoxels();
  if (publish_slices_) publishSlices();
  if (publish_traversable_) publishTraversable();
  TsdfServer::publishPointclouds();
}

void EsdfServer::publishTraversable() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  createFreePointcloudFromEsdfLayer(esdf_map_->getEsdfLayer(),
                                    traversability_radius_, &pointcloud);
  traversable_pub_->publish(toPointCloud2Msg(pointcloud, world_frame_));
}

void EsdfServer::publishMap(bool reset_remote_map) {
  if (!publish_esdf_map_) return;

  const int subscribers =
      static_cast<int>(esdf_map_pub_->get_subscription_count());
  if (subscribers > 0) {
    if (num_subscribers_esdf_map_ < subscribers) {
      reset_remote_map = true;
    }
    const bool only_updated = !reset_remote_map;
    timing::Timer publish_map_timer("map/publish_esdf");
    voxblox_msgs::msg::Layer layer_msg;
    serializeLayerAsMsg<EsdfVoxel>(esdf_map_->getEsdfLayer(), only_updated,
                                   &layer_msg);
    if (reset_remote_map) {
      layer_msg.action = static_cast<uint8_t>(MapDerializationAction::kReset);
    }
    esdf_map_pub_->publish(layer_msg);
    publish_map_timer.Stop();
  }
  num_subscribers_esdf_map_ = subscribers;
  TsdfServer::publishMap();
}

bool EsdfServer::saveMap(const std::string& file_path) {
  const bool success = TsdfServer::saveMap(file_path);
  constexpr bool kClearFile = false;
  return success &&
         io::SaveLayer(esdf_map_->getEsdfLayer(), file_path, kClearFile);
}

bool EsdfServer::loadMap(const std::string& file_path) {
  bool success = TsdfServer::loadMap(file_path);
  constexpr bool kMultipleLayerSupport = true;
  return success &&
         io::LoadBlocksFromFile(
             file_path, Layer<EsdfVoxel>::BlockMergingStrategy::kReplace,
             kMultipleLayerSupport, esdf_map_->getEsdfLayerPtr());
}

void EsdfServer::updateEsdf() {
  if (tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() > 0) {
    const bool clear_updated_flag_esdf = true;
    esdf_integrator_->updateFromTsdfLayer(clear_updated_flag_esdf);
  }
}

void EsdfServer::updateEsdfBatch(bool full_euclidean) {
  if (tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks() > 0) {
    esdf_integrator_->setFullEuclidean(full_euclidean);
    esdf_integrator_->updateFromTsdfLayerBatch();
  }
}

float EsdfServer::getEsdfMaxDistance() const {
  return esdf_integrator_->getEsdfMaxDistance();
}

void EsdfServer::setEsdfMaxDistance(float max_distance) {
  esdf_integrator_->setEsdfMaxDistance(max_distance);
}

float EsdfServer::getTraversabilityRadius() const {
  return traversability_radius_;
}

void EsdfServer::setTraversabilityRadius(float traversability_radius) {
  traversability_radius_ = traversability_radius;
}

void EsdfServer::newPoseCallback(const Transformation& T_G_C) {
  if (clear_sphere_for_planning_) {
    esdf_integrator_->addNewRobotPosition(T_G_C.getPosition());
  }

  timing::Timer block_remove_timer("remove_distant_blocks");
  esdf_map_->getEsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
  block_remove_timer.Stop();
}

void EsdfServer::esdfMapCallback(
    const voxblox_msgs::msg::Layer::SharedPtr layer_msg) {
  timing::Timer receive_map_timer("map/receive_esdf");

  bool success = deserializeMsgToLayer<EsdfVoxel>(
      *layer_msg, esdf_map_->getEsdfLayerPtr());

  if (!success) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 10000,
                          "Got an invalid ESDF map message!");
  } else {
    RCLCPP_INFO_ONCE(node_->get_logger(), "Got an ESDF map from ROS topic!");
    if (publish_pointclouds_) publishPointclouds();
  }
}

void EsdfServer::clear() {
  esdf_map_->getEsdfLayerPtr()->removeAllBlocks();
  esdf_integrator_->clear();
  CHECK_EQ(esdf_map_->getEsdfLayerPtr()->getNumberOfAllocatedBlocks(), 0u);

  TsdfServer::clear();

  constexpr bool kResetRemoteMap = true;
  publishMap(kResetRemoteMap);
}

}  // namespace voxblox
