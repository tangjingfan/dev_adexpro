#include "voxblox_ros/tsdf_server.h"

#include <chrono>

#include <minkindr_conversions/kindr_msg.h>
#include <minkindr_conversions/kindr_tf.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "voxblox_ros/conversions.h"
#include "voxblox_ros/ros_params.h"

namespace voxblox {

namespace {
// Convert pcl::PointCloud<T> to sensor_msgs::msg::PointCloud2. pcl_ros provides
// this in ROS 1 but the ROS 2 helper requires pcl::toROSMsg from pcl_conversions.
template <typename PointT>
sensor_msgs::msg::PointCloud2 toPointCloud2Msg(
    const pcl::PointCloud<PointT>& cloud, const std::string& frame_id) {
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  msg.header.frame_id = frame_id;
  return msg;
}
}  // namespace

TsdfServer::TsdfServer(rclcpp::Node* node)
    : TsdfServer(node, getTsdfMapConfigFromRosParam(node),
                 getTsdfIntegratorConfigFromRosParam(node),
                 getMeshIntegratorConfigFromRosParam(node)) {}

TsdfServer::TsdfServer(rclcpp::Node* node, const TsdfMap::Config& config,
                       const TsdfIntegratorBase::Config& integrator_config,
                       const MeshIntegratorConfig& mesh_config)
    : node_(node),
      verbose_(true),
      world_frame_("world"),
      icp_corrected_frame_("icp_corrected"),
      pose_corrected_frame_("pose_corrected"),
      max_block_distance_from_body_(std::numeric_limits<FloatingPoint>::max()),
      slice_level_(0.5),
      use_freespace_pointcloud_(false),
      color_map_(new RainbowColorMap()),
      min_time_between_msgs_(rclcpp::Duration(0, 0)),
      publish_pointclouds_on_update_(false),
      publish_slices_(false),
      publish_pointclouds_(false),
      publish_tsdf_map_(false),
      cache_mesh_(false),
      enable_icp_(false),
      accumulate_icp_corrections_(true),
      pointcloud_queue_size_(1),
      num_subscribers_tsdf_map_(0),
      transformer_(node) {
  getServerConfigFromRosParam();

  // Latched topics in ROS 2 use TRANSIENT_LOCAL + KEEP_LAST(1) QoS.
  auto latched_qos = rclcpp::QoS(1).transient_local();

  surface_pointcloud_pub_ =
      node_->create_publisher<sensor_msgs::msg::PointCloud2>(
          "surface_pointcloud", latched_qos);
  tsdf_pointcloud_pub_ =
      node_->create_publisher<sensor_msgs::msg::PointCloud2>("tsdf_pointcloud",
                                                             latched_qos);
  occupancy_marker_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>(
          "occupied_nodes", latched_qos);
  tsdf_slice_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
      "tsdf_slice", latched_qos);

  pointcloud_queue_size_ =
      declareOrGet<int>(node_, "pointcloud_queue_size", pointcloud_queue_size_);
  pointcloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      "pointcloud", pointcloud_queue_size_,
      std::bind(&TsdfServer::insertPointcloud, this, std::placeholders::_1));

  mesh_pub_ = node_->create_publisher<voxblox_msgs::msg::Mesh>("mesh",
                                                               latched_qos);

  // Publishing/subscribing to a layer from another node (when using this as
  // a library, for example within a planner).
  tsdf_map_pub_ = node_->create_publisher<voxblox_msgs::msg::Layer>(
      "tsdf_map_out", rclcpp::QoS(1));
  tsdf_map_sub_ = node_->create_subscription<voxblox_msgs::msg::Layer>(
      "tsdf_map_in", 1,
      std::bind(&TsdfServer::tsdfMapCallback, this, std::placeholders::_1));
  publish_tsdf_map_ =
      declareOrGet<bool>(node_, "publish_tsdf_map", publish_tsdf_map_);

  if (use_freespace_pointcloud_) {
    freespace_pointcloud_sub_ =
        node_->create_subscription<sensor_msgs::msg::PointCloud2>(
            "freespace_pointcloud", pointcloud_queue_size_,
            std::bind(&TsdfServer::insertFreespacePointcloud, this,
                      std::placeholders::_1));
  }

  if (enable_icp_) {
    icp_transform_pub_ =
        node_->create_publisher<geometry_msgs::msg::TransformStamped>(
            "icp_transform", latched_qos);
    icp_corrected_frame_ = declareOrGet<std::string>(
        node_, "icp_corrected_frame", icp_corrected_frame_);
    pose_corrected_frame_ = declareOrGet<std::string>(
        node_, "pose_corrected_frame", pose_corrected_frame_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node_);
  }

  // Initialize TSDF map and integrator.
  tsdf_map_.reset(new TsdfMap(config));

  std::string method =
      declareOrGet<std::string>(node_, "method", std::string("merged"));
  if (method == "simple") {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method == "merged") {
    tsdf_integrator_.reset(new MergedTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else if (method == "fast") {
    tsdf_integrator_.reset(new FastTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  } else {
    tsdf_integrator_.reset(new SimpleTsdfIntegrator(
        integrator_config, tsdf_map_->getTsdfLayerPtr()));
  }

  mesh_layer_.reset(new MeshLayer(tsdf_map_->block_size()));
  mesh_integrator_.reset(new MeshIntegrator<TsdfVoxel>(
      mesh_config, tsdf_map_->getTsdfLayerPtr(), mesh_layer_.get()));

  icp_.reset(new ICP(getICPConfigFromRosParam(node_)));

  generate_mesh_srv_ = node_->create_service<std_srvs::srv::Empty>(
      "generate_mesh",
      std::bind(&TsdfServer::generateMeshCallback, this,
                std::placeholders::_1, std::placeholders::_2));
  clear_map_srv_ = node_->create_service<std_srvs::srv::Empty>(
      "clear_map",
      std::bind(&TsdfServer::clearMapCallback, this, std::placeholders::_1,
                std::placeholders::_2));
  save_map_srv_ = node_->create_service<voxblox_msgs::srv::FilePath>(
      "save_map",
      std::bind(&TsdfServer::saveMapCallback, this, std::placeholders::_1,
                std::placeholders::_2));
  load_map_srv_ = node_->create_service<voxblox_msgs::srv::FilePath>(
      "load_map",
      std::bind(&TsdfServer::loadMapCallback, this, std::placeholders::_1,
                std::placeholders::_2));
  publish_pointclouds_srv_ = node_->create_service<std_srvs::srv::Empty>(
      "publish_pointclouds",
      std::bind(&TsdfServer::publishPointcloudsCallback, this,
                std::placeholders::_1, std::placeholders::_2));
  publish_tsdf_map_srv_ = node_->create_service<std_srvs::srv::Empty>(
      "publish_map",
      std::bind(&TsdfServer::publishTsdfMapCallback, this,
                std::placeholders::_1, std::placeholders::_2));

  double update_mesh_every_n_sec =
      declareOrGet<double>(node_, "update_mesh_every_n_sec", 1.0);
  if (update_mesh_every_n_sec > 0.0) {
    update_mesh_timer_ = node_->create_wall_timer(
        std::chrono::duration<double>(update_mesh_every_n_sec),
        std::bind(&TsdfServer::updateMeshEvent, this));
  }

  double publish_map_every_n_sec =
      declareOrGet<double>(node_, "publish_map_every_n_sec", 1.0);
  if (publish_map_every_n_sec > 0.0) {
    publish_map_timer_ = node_->create_wall_timer(
        std::chrono::duration<double>(publish_map_every_n_sec),
        std::bind(&TsdfServer::publishMapEvent, this));
  }
}

void TsdfServer::getServerConfigFromRosParam() {
  double min_time_between_msgs_sec =
      declareOrGet<double>(node_, "min_time_between_msgs_sec", 0.0);
  min_time_between_msgs_ =
      rclcpp::Duration::from_seconds(min_time_between_msgs_sec);

  max_block_distance_from_body_ = declareOrGet<double>(
      node_, "max_block_distance_from_body", max_block_distance_from_body_);
  slice_level_ = declareOrGet<double>(node_, "slice_level", slice_level_);
  world_frame_ = declareOrGet<std::string>(node_, "world_frame", world_frame_);
  publish_pointclouds_on_update_ = declareOrGet<bool>(
      node_, "publish_pointclouds_on_update", publish_pointclouds_on_update_);
  publish_slices_ =
      declareOrGet<bool>(node_, "publish_slices", publish_slices_);
  publish_pointclouds_ =
      declareOrGet<bool>(node_, "publish_pointclouds", publish_pointclouds_);

  use_freespace_pointcloud_ = declareOrGet<bool>(
      node_, "use_freespace_pointcloud", use_freespace_pointcloud_);
  pointcloud_queue_size_ =
      declareOrGet<int>(node_, "pointcloud_queue_size", pointcloud_queue_size_);
  enable_icp_ = declareOrGet<bool>(node_, "enable_icp", enable_icp_);
  accumulate_icp_corrections_ = declareOrGet<bool>(
      node_, "accumulate_icp_corrections", accumulate_icp_corrections_);

  verbose_ = declareOrGet<bool>(node_, "verbose", verbose_);

  mesh_filename_ =
      declareOrGet<std::string>(node_, "mesh_filename", mesh_filename_);
  std::string color_mode =
      declareOrGet<std::string>(node_, "color_mode", std::string(""));
  color_mode_ = getColorModeFromString(color_mode);

  std::string intensity_colormap = declareOrGet<std::string>(
      node_, "intensity_colormap", std::string("rainbow"));
  double intensity_max_value = declareOrGet<double>(
      node_, "intensity_max_value", static_cast<double>(kDefaultMaxIntensity));

  if (intensity_colormap == "rainbow") {
    color_map_.reset(new RainbowColorMap());
  } else if (intensity_colormap == "inverse_rainbow") {
    color_map_.reset(new InverseRainbowColorMap());
  } else if (intensity_colormap == "grayscale") {
    color_map_.reset(new GrayscaleColorMap());
  } else if (intensity_colormap == "inverse_grayscale") {
    color_map_.reset(new InverseGrayscaleColorMap());
  } else if (intensity_colormap == "ironbow") {
    color_map_.reset(new IronbowColorMap());
  } else {
    RCLCPP_ERROR_STREAM(node_->get_logger(),
                        "Invalid color map: " << intensity_colormap);
  }
  color_map_->setMaxValue(intensity_max_value);
}

void TsdfServer::processPointCloudMessageAndInsert(
    const sensor_msgs::msg::PointCloud2::SharedPtr& pointcloud_msg,
    const Transformation& T_G_C, const bool is_freespace_pointcloud) {
  // Detect color / intensity fields. ROS 2 PointField datatype constants live
  // in sensor_msgs::msg::PointField.
  bool color_pointcloud = false;
  bool has_intensity = false;
  for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
    if (pointcloud_msg->fields[d].name == std::string("rgb")) {
      pointcloud_msg->fields[d].datatype =
          sensor_msgs::msg::PointField::FLOAT32;
      color_pointcloud = true;
    } else if (pointcloud_msg->fields[d].name == std::string("intensity")) {
      has_intensity = true;
    }
  }

  Pointcloud points_C;
  Colors colors;
  timing::Timer ptcloud_timer("ptcloud_preprocess");

  if (color_pointcloud) {
    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  } else if (has_intensity) {
    pcl::PointCloud<pcl::PointXYZI> pointcloud_pcl;
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  } else {
    pcl::PointCloud<pcl::PointXYZ> pointcloud_pcl;
    pcl::fromROSMsg(*pointcloud_msg, pointcloud_pcl);
    convertPointcloud(pointcloud_pcl, color_map_, &points_C, &colors);
  }
  ptcloud_timer.Stop();

  Transformation T_G_C_refined = T_G_C;
  if (enable_icp_) {
    timing::Timer icp_timer("icp");
    if (!accumulate_icp_corrections_) {
      icp_corrected_transform_.setIdentity();
    }
    const size_t num_icp_updates =
        icp_->runICP(tsdf_map_->getTsdfLayer(), points_C,
                     icp_corrected_transform_ * T_G_C, &T_G_C_refined);
    if (verbose_) {
      RCLCPP_INFO(node_->get_logger(),
                  "ICP refinement performed %zu successful update steps",
                  num_icp_updates);
    }
    icp_corrected_transform_ = T_G_C_refined * T_G_C.inverse();

    if (!icp_->refiningRollPitch()) {
      Transformation::Vector6 T_vec = icp_corrected_transform_.log();
      T_vec[3] = 0.0;
      T_vec[4] = 0.0;
      icp_corrected_transform_ = Transformation::exp(T_vec);
    }

    // Publish transforms as both TF and message.
    geometry_msgs::msg::TransformStamped icp_tf_stamped;
    icp_tf_stamped.header.stamp = pointcloud_msg->header.stamp;
    icp_tf_stamped.header.frame_id = world_frame_;
    icp_tf_stamped.child_frame_id = icp_corrected_frame_;
    tf2::transformKindrToMsg(icp_corrected_transform_.cast<double>(),
                             &icp_tf_stamped.transform);

    geometry_msgs::msg::TransformStamped pose_tf_stamped;
    pose_tf_stamped.header.stamp = pointcloud_msg->header.stamp;
    pose_tf_stamped.header.frame_id = icp_corrected_frame_;
    pose_tf_stamped.child_frame_id = pose_corrected_frame_;
    tf2::transformKindrToMsg(T_G_C.cast<double>(), &pose_tf_stamped.transform);

    if (tf_broadcaster_) {
      tf_broadcaster_->sendTransform(icp_tf_stamped);
      tf_broadcaster_->sendTransform(pose_tf_stamped);
    }

    icp_transform_pub_->publish(icp_tf_stamped);
    icp_timer.Stop();
  }

  if (verbose_) {
    RCLCPP_INFO(node_->get_logger(),
                "Integrating a pointcloud with %lu points.", points_C.size());
  }

  auto start = std::chrono::steady_clock::now();
  integratePointcloud(T_G_C_refined, points_C, colors, is_freespace_pointcloud);
  auto end = std::chrono::steady_clock::now();
  if (verbose_) {
    RCLCPP_INFO(
        node_->get_logger(),
        "Finished integrating in %f seconds, have %lu blocks.",
        std::chrono::duration<double>(end - start).count(),
        tsdf_map_->getTsdfLayer().getNumberOfAllocatedBlocks());
  }

  timing::Timer block_remove_timer("remove_distant_blocks");
  tsdf_map_->getTsdfLayerPtr()->removeDistantBlocks(
      T_G_C.getPosition(), max_block_distance_from_body_);
  mesh_layer_->clearDistantMesh(T_G_C.getPosition(),
                                max_block_distance_from_body_);
  block_remove_timer.Stop();

  newPoseCallback(T_G_C);
}

bool TsdfServer::getNextPointcloudFromQueue(
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr>* queue,
    sensor_msgs::msg::PointCloud2::SharedPtr* pointcloud_msg,
    Transformation* T_G_C) {
  const size_t kMaxQueueSize = 10;
  if (queue->empty()) {
    return false;
  }
  *pointcloud_msg = queue->front();
  if (transformer_.lookupTransform((*pointcloud_msg)->header.frame_id,
                                   world_frame_,
                                   (*pointcloud_msg)->header.stamp, T_G_C)) {
    queue->pop();
    return true;
  } else {
    if (queue->size() >= kMaxQueueSize) {
      RCLCPP_ERROR_THROTTLE(
          node_->get_logger(), *node_->get_clock(), 60000,
          "Input pointcloud queue getting too long! Dropping some pointclouds. "
          "Either unable to look up transform timestamps or the processing is "
          "taking too long.");
      while (queue->size() >= kMaxQueueSize) {
        queue->pop();
      }
    }
  }
  return false;
}

void TsdfServer::insertPointcloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg_in) {
  const rclcpp::Time msg_stamp(pointcloud_msg_in->header.stamp);
  if (msg_stamp - last_msg_time_ptcloud_ > min_time_between_msgs_) {
    last_msg_time_ptcloud_ = msg_stamp;
    pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg;
  bool processed_any = false;
  while (getNextPointcloudFromQueue(&pointcloud_queue_, &pointcloud_msg,
                                    &T_G_C)) {
    constexpr bool is_freespace_pointcloud = false;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
    processed_any = true;
  }

  if (!processed_any) return;

  if (publish_pointclouds_on_update_) publishPointclouds();

  if (verbose_) {
    RCLCPP_INFO_STREAM(node_->get_logger(),
                       "Timings: " << std::endl << timing::Timing::Print());
    RCLCPP_INFO_STREAM(
        node_->get_logger(),
        "Layer memory: " << tsdf_map_->getTsdfLayer().getMemorySize());
  }
}

void TsdfServer::insertFreespacePointcloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg_in) {
  const rclcpp::Time msg_stamp(pointcloud_msg_in->header.stamp);
  if (msg_stamp - last_msg_time_freespace_ptcloud_ > min_time_between_msgs_) {
    last_msg_time_freespace_ptcloud_ = msg_stamp;
    freespace_pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::msg::PointCloud2::SharedPtr pointcloud_msg;
  while (getNextPointcloudFromQueue(&freespace_pointcloud_queue_,
                                    &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = true;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
  }
}

void TsdfServer::integratePointcloud(const Transformation& T_G_C,
                                     const Pointcloud& ptcloud_C,
                                     const Colors& colors,
                                     const bool is_freespace_pointcloud) {
  CHECK_EQ(ptcloud_C.size(), colors.size());
  tsdf_integrator_->integratePointCloud(T_G_C, ptcloud_C, colors,
                                        is_freespace_pointcloud);
}

void TsdfServer::publishAllUpdatedTsdfVoxels() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  createDistancePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(), &pointcloud);
  tsdf_pointcloud_pub_->publish(toPointCloud2Msg(pointcloud, world_frame_));
}

void TsdfServer::publishTsdfSurfacePoints() {
  pcl::PointCloud<pcl::PointXYZRGB> pointcloud;
  const float surface_distance_thresh =
      tsdf_map_->getTsdfLayer().voxel_size() * 0.75;
  createSurfacePointcloudFromTsdfLayer(tsdf_map_->getTsdfLayer(),
                                       surface_distance_thresh, &pointcloud);
  surface_pointcloud_pub_->publish(toPointCloud2Msg(pointcloud, world_frame_));
}

void TsdfServer::publishTsdfOccupiedNodes() {
  visualization_msgs::msg::MarkerArray marker_array;
  createOccupancyBlocksFromTsdfLayer(tsdf_map_->getTsdfLayer(), world_frame_,
                                     &marker_array);
  occupancy_marker_pub_->publish(marker_array);
}

void TsdfServer::publishSlices() {
  pcl::PointCloud<pcl::PointXYZI> pointcloud;
  createDistancePointcloudFromTsdfLayerSlice(tsdf_map_->getTsdfLayer(), 2,
                                             slice_level_, &pointcloud);
  tsdf_slice_pub_->publish(toPointCloud2Msg(pointcloud, world_frame_));
}

void TsdfServer::publishMap(bool reset_remote_map) {
  if (!publish_tsdf_map_) return;

  // In ROS 2, get_subscription_count() exists on rclcpp::Publisher.
  const int subscribers =
      static_cast<int>(tsdf_map_pub_->get_subscription_count());
  if (subscribers > 0) {
    if (num_subscribers_tsdf_map_ < subscribers) {
      reset_remote_map = true;
    }
    const bool only_updated = !reset_remote_map;
    timing::Timer publish_map_timer("map/publish_tsdf");
    voxblox_msgs::msg::Layer layer_msg;
    serializeLayerAsMsg<TsdfVoxel>(tsdf_map_->getTsdfLayer(), only_updated,
                                   &layer_msg);
    if (reset_remote_map) {
      layer_msg.action = static_cast<uint8_t>(MapDerializationAction::kReset);
    }
    tsdf_map_pub_->publish(layer_msg);
    publish_map_timer.Stop();
  }
  num_subscribers_tsdf_map_ = subscribers;
}

void TsdfServer::publishPointclouds() {
  publishAllUpdatedTsdfVoxels();
  publishTsdfSurfacePoints();
  publishTsdfOccupiedNodes();
  if (publish_slices_) {
    publishSlices();
  }
}

void TsdfServer::updateMesh() {
  if (verbose_) RCLCPP_INFO(node_->get_logger(), "Updating mesh.");

  timing::Timer generate_mesh_timer("mesh/update");
  constexpr bool only_mesh_updated_blocks = true;
  constexpr bool clear_updated_flag = true;
  mesh_integrator_->generateMesh(only_mesh_updated_blocks, clear_updated_flag);
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::msg::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg, node_->now());
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_->publish(mesh_msg);

  if (cache_mesh_) cached_mesh_msg_ = mesh_msg;
  publish_mesh_timer.Stop();

  if (publish_pointclouds_ && !publish_pointclouds_on_update_) {
    publishPointclouds();
  }
}

bool TsdfServer::generateMesh() {
  timing::Timer generate_mesh_timer("mesh/generate");
  const bool clear_mesh = true;
  if (clear_mesh) {
    constexpr bool only_mesh_updated_blocks = false;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  } else {
    constexpr bool only_mesh_updated_blocks = true;
    constexpr bool clear_updated_flag = true;
    mesh_integrator_->generateMesh(only_mesh_updated_blocks,
                                   clear_updated_flag);
  }
  generate_mesh_timer.Stop();

  timing::Timer publish_mesh_timer("mesh/publish");
  voxblox_msgs::msg::Mesh mesh_msg;
  generateVoxbloxMeshMsg(mesh_layer_, color_mode_, &mesh_msg, node_->now());
  mesh_msg.header.frame_id = world_frame_;
  mesh_pub_->publish(mesh_msg);
  publish_mesh_timer.Stop();

  if (!mesh_filename_.empty()) {
    timing::Timer output_mesh_timer("mesh/output");
    const bool success = outputMeshLayerAsPly(mesh_filename_, *mesh_layer_);
    output_mesh_timer.Stop();
    if (success) {
      RCLCPP_INFO(node_->get_logger(), "Output file as PLY: %s",
                  mesh_filename_.c_str());
    } else {
      RCLCPP_INFO(node_->get_logger(), "Failed to output mesh as PLY: %s",
                  mesh_filename_.c_str());
    }
  }

  RCLCPP_INFO_STREAM(node_->get_logger(),
                     "Mesh Timings: " << std::endl << timing::Timing::Print());
  return true;
}

bool TsdfServer::saveMap(const std::string& file_path) {
  return io::SaveLayer(tsdf_map_->getTsdfLayer(), file_path);
}

bool TsdfServer::loadMap(const std::string& file_path) {
  constexpr bool kMulitpleLayerSupport = true;
  bool success = io::LoadBlocksFromFile(
      file_path, Layer<TsdfVoxel>::BlockMergingStrategy::kReplace,
      kMulitpleLayerSupport, tsdf_map_->getTsdfLayerPtr());
  if (success) LOG(INFO) << "Successfully loaded TSDF layer.";
  return success;
}

void TsdfServer::clearMapCallback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/) {
  clear();
}

void TsdfServer::generateMeshCallback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/) {
  generateMesh();
}

void TsdfServer::saveMapCallback(
    const std::shared_ptr<voxblox_msgs::srv::FilePath::Request> request,
    std::shared_ptr<voxblox_msgs::srv::FilePath::Response> /*response*/) {
  saveMap(request->file_path);
}

void TsdfServer::loadMapCallback(
    const std::shared_ptr<voxblox_msgs::srv::FilePath::Request> request,
    std::shared_ptr<voxblox_msgs::srv::FilePath::Response> /*response*/) {
  loadMap(request->file_path);
}

void TsdfServer::publishPointcloudsCallback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/) {
  publishPointclouds();
}

void TsdfServer::publishTsdfMapCallback(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/) {
  publishMap();
}

void TsdfServer::updateMeshEvent() { updateMesh(); }

void TsdfServer::publishMapEvent() { publishMap(); }

void TsdfServer::clear() {
  tsdf_map_->getTsdfLayerPtr()->removeAllBlocks();
  mesh_layer_->clear();

  if (publish_tsdf_map_) {
    constexpr bool kResetRemoteMap = true;
    publishMap(kResetRemoteMap);
  }
}

void TsdfServer::tsdfMapCallback(
    const voxblox_msgs::msg::Layer::SharedPtr layer_msg) {
  timing::Timer receive_map_timer("map/receive_tsdf");

  bool success = deserializeMsgToLayer<TsdfVoxel>(
      *layer_msg, tsdf_map_->getTsdfLayerPtr());

  if (!success) {
    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 10000,
                          "Got an invalid TSDF map message!");
  } else {
    RCLCPP_INFO_ONCE(node_->get_logger(), "Got an TSDF map from ROS topic!");
    if (publish_pointclouds_on_update_) publishPointclouds();
  }
}

}  // namespace voxblox
