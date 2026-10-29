// Node lifecycle (ctor/dtor), parameter declaration/loading, ROS sub/pub/timer
// init, raw topic callbacks, and tiny geometry helpers. The rest of the
// implementation is split by responsibility into:
//   - global_planner_frontiers.cpp     frontier extraction, filter, downsample
//   - global_planner_reachability.cpp  BFS, A*, distance matrix, traversability
//   - global_planner_planning.cpp      vertex prep, RPV solve, publishPath
//   - global_planner_scheduling.cpp    timer, dispatch, background thread
// All method definitions still belong to the single GlobalPlanner class in
// include/global_planner.hpp; the split is purely for maintainability.

#include "../include/global_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

GlobalPlanner::GlobalPlanner() : Node("global_planner") {
  declareParameters();
  loadParameters();
  initializeSubscribers();
  initializePublishers();
  initializeTimers();
  planningThread_ = std::thread(&GlobalPlanner::backgroundPlanningLoop, this);
}

GlobalPlanner::~GlobalPlanner() {
  {
    std::lock_guard<std::mutex> lock(planningMutex_);
    killPlanningThread_ = true;
    planRequested_ = true;
  }
  planningCv_.notify_all();
  if (planningThread_.joinable()) {
    planningThread_.join();
  }
}

void GlobalPlanner::declareParameters() {
  declare_parameter("checker_cycle", 0.5);
  declare_parameter("max_range_of_local_frontiers", 20.0);
  declare_parameter("replan_path_length_threshold", 1.0);
  declare_parameter("replan_time_threshold", 2.0);
  declare_parameter("frontier_detection_range", -1.0);
  declare_parameter("global_planning_failure_threshold", 1);
  declare_parameter("local_cluster_size", 8);
  declare_parameter("global_cluster_size", 30);
  declare_parameter("local_min_marginal_gain", 2);
  declare_parameter("global_min_marginal_gain", 2);
  declare_parameter("predicted_confidence_multiplier", 1.0);
  declare_parameter("turn_cost_factor", 2.0);
  declare_parameter("range_threshold_global_frontier", 6.0);
  declare_parameter("max_frontiers_to_plan", 20);
  declare_parameter("planning_time_limit", 1.0);
  declare_parameter("exploration_enabled_topic", "/exploration_enabled");
  declare_parameter("exploration_enabled_initial", true);
  declare_parameter("distance_matrix_worker_count", 0);
  declare_parameter("distance_matrix_parallel_min_vertices", 8);
  declare_parameter("frontier_unknown_neighbor_threshold", 2);
  declare_parameter("neighbor_safety_radius", 1);
  declare_parameter("min_clearance_cells", 0.0);
  declare_parameter("clearance_tight_cell_budget", 5);
  declare_parameter("occupied_cell_threshold", 50);
  declare_parameter("predicted_traversable_threshold", 65);
  declare_parameter("predicted_occupied_penalty", 10.0);
  declare_parameter("cardinal_move_cost", 1.0);
  declare_parameter("diagonal_move_cost", 1.414);
  declare_parameter("confidence_threshold", 0.5);
  declare_parameter("local_hysteresis_factor", 1.0);
  declare_parameter("local_hysteresis_match_radius", 1.5);
  declare_parameter("global_guide_enabled", true);
  declare_parameter("global_guide_period", 3);
  declare_parameter("predicted_region_grouping", false);
  declare_parameter("predicted_region_min_cells", 20);
  declare_parameter("cross_source_min_separation_cells", 5.0);
  declare_parameter("predicted_edge_inflation_alpha", 0.0);
}

void GlobalPlanner::loadParameters() {
  checkerCycle_                      = get_parameter("checker_cycle").as_double();
  maxRangeOfLocalFrontiers_          = get_parameter("max_range_of_local_frontiers").as_double();
  replanPathLengthThreshold_         = get_parameter("replan_path_length_threshold").as_double();
  replanTimeThreshold_               = get_parameter("replan_time_threshold").as_double();
  frontierDetectionRange_            = get_parameter("frontier_detection_range").as_double();
  globalPlanningFailureThreshold_    = get_parameter("global_planning_failure_threshold").as_int();
  localClusterSize_                  = get_parameter("local_cluster_size").as_int();
  globalClusterSize_                 = get_parameter("global_cluster_size").as_int();
  localMinMarginalGain_              = get_parameter("local_min_marginal_gain").as_int();
  globalMinMarginalGain_             = get_parameter("global_min_marginal_gain").as_int();
  predictedConfidenceMultiplier_     = get_parameter("predicted_confidence_multiplier").as_double();
  turnCostFactor_                    = get_parameter("turn_cost_factor").as_double();
  rangeThresholdGlobalFrontier_      = get_parameter("range_threshold_global_frontier").as_double();
  maxFrontiersToPlan_                = get_parameter("max_frontiers_to_plan").as_int();
  planningTimeLimit_                 = get_parameter("planning_time_limit").as_double();
  explorationEnabledTopic_           = get_parameter("exploration_enabled_topic").as_string();
  explorationEnabledInitial_         = get_parameter("exploration_enabled_initial").as_bool();
  explorationEnabled_                = explorationEnabledInitial_;
  distanceMatrixWorkerCount_         = get_parameter("distance_matrix_worker_count").as_int();
  distanceMatrixParallelMinVertices_ = get_parameter("distance_matrix_parallel_min_vertices").as_int();
  frontierUnknownNeighborThreshold_  = get_parameter("frontier_unknown_neighbor_threshold").as_int();
  neighborSafetyRadius_              = get_parameter("neighbor_safety_radius").as_int();
  minClearanceCells_                 = get_parameter("min_clearance_cells").as_double();
  clearanceTightCellBudget_          = get_parameter("clearance_tight_cell_budget").as_int();
  occupiedCellThreshold_             = get_parameter("occupied_cell_threshold").as_int();
  predictedTraversableThreshold_     = get_parameter("predicted_traversable_threshold").as_int();
  predictedOccupiedPenalty_          = get_parameter("predicted_occupied_penalty").as_double();
  cardinalMoveCost_                  = get_parameter("cardinal_move_cost").as_double();
  diagonalMoveCost_                  = get_parameter("diagonal_move_cost").as_double();
  confidenceThreshold_               = get_parameter("confidence_threshold").as_double();
  localHysteresisFactor_             = get_parameter("local_hysteresis_factor").as_double();
  localHysteresisMatchRadius_        = get_parameter("local_hysteresis_match_radius").as_double();
  globalGuideEnabled_                = get_parameter("global_guide_enabled").as_bool();
  globalGuidePeriod_                 = std::max(1, static_cast<int>(get_parameter("global_guide_period").as_int()));
  predictedRegionGrouping_           = get_parameter("predicted_region_grouping").as_bool();
  predictedRegionMinCells_           = static_cast<int>(get_parameter("predicted_region_min_cells").as_int());
  crossSourceMinSeparationCells_     = get_parameter("cross_source_min_separation_cells").as_double();
  predictedEdgeInflationAlpha_       = get_parameter("predicted_edge_inflation_alpha").as_double();
}

void GlobalPlanner::initializeSubscribers() {
  visibleSub_    = create_subscription<nav_msgs::msg::OccupancyGrid>("/visible_map", 10, std::bind(&GlobalPlanner::visibleCallback, this, std::placeholders::_1));
  predictedSub_  = create_subscription<nav_msgs::msg::OccupancyGrid>("/predicted_map", 10, std::bind(&GlobalPlanner::predictedCallback, this, std::placeholders::_1));
  confidenceSub_ = create_subscription<prob_msgs::msg::ProbGrid>("/confidence_map", 10, std::bind(&GlobalPlanner::confidenceCallback, this, std::placeholders::_1));
  odomSub_       = create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 10, std::bind(&GlobalPlanner::odomCallback, this, std::placeholders::_1));
  replanSub_     = create_subscription<std_msgs::msg::Bool>("/replan_request", 10, std::bind(&GlobalPlanner::replanCallback, this, std::placeholders::_1));
  rclcpp::QoS explorationControlQos(1);
  explorationControlQos.transient_local();
  explorationControlQos.reliable();
  explorationEnabledSub_ = create_subscription<std_msgs::msg::Bool>(
    explorationEnabledTopic_, explorationControlQos, std::bind(&GlobalPlanner::explorationEnabledCallback, this, std::placeholders::_1)
  );
}

void GlobalPlanner::initializePublishers() {
  localFrontierPub_       = create_publisher<sensor_msgs::msg::PointCloud2>("/local_frontiers", 10);
  globalFrontierPub_      = create_publisher<sensor_msgs::msg::PointCloud2>("/global_frontiers", 10);
  mapEdgeFrontierPub_     = create_publisher<sensor_msgs::msg::PointCloud2>("/map_edge_frontiers", 10);
  predictedFrontierPub_   = create_publisher<sensor_msgs::msg::PointCloud2>("/predicted_frontiers", 10);
  paddedMapPub_           = create_publisher<nav_msgs::msg::OccupancyGrid>("/padded_visible_map", 10);
  globalFrontiersProbPub_ = create_publisher<prob_msgs::msg::ProbPointVec>("/global_frontiers_prob", 10);
  planningStatsPub_       = create_publisher<rpv_msgs::msg::PlanningStats>("/global_planner_stats", rclcpp::QoS(1));
  solutionPathPublisher_  = create_publisher<nav_msgs::msg::Path>("/solution_path", 10);
  plannerRuntimePub_      = create_publisher<std_msgs::msg::Float32>("/runtime", 10);
  clearanceDroppedPub_    = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/clearance_filtered_vertices", 10);
  globalGuidePub_         = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/global_guide", 10);
  rclcpp::QoS explorationCompleteQos(1);
  explorationCompleteQos.transient_local();
  explorationCompleteQos.reliable();
  explorationCompletePub_ = create_publisher<std_msgs::msg::Bool>(
      "/exploration_complete", explorationCompleteQos);
}

void GlobalPlanner::initializeTimers() {
  const auto checkerPeriod = std::chrono::milliseconds(std::max(1, static_cast<int>(checkerCycle_ * 1000.0)));
  checkerTimer_ = create_wall_timer(checkerPeriod, std::bind(&GlobalPlanner::checkerCallback, this));
}

void GlobalPlanner::visibleCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg
) {
  if (!msg) {
    return;
  }
  visibleReceived_ = true;
  visible_ = *msg;
}

void GlobalPlanner::predictedCallback(
    const nav_msgs::msg::OccupancyGrid::SharedPtr msg
) {
  if (!msg) {
    return;
  }
  predictedReceived_ = true;
  predicted_ = *msg;
}

void GlobalPlanner::confidenceCallback(const prob_msgs::msg::ProbGrid::SharedPtr msg
) {
  if (!msg) {
    return;
  }
  confidenceReceived_ = true;
  confidence_ = *msg;
}

void GlobalPlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  if (!msg) {
    return;
  }

  odomReceived_ = true;
  currentPose_ = msg->pose.pose;

  if (!visibleReceived_) {
    return;
  }

  robotMapPos_.x = static_cast<int>(
      (currentPose_.position.x - visible_.info.origin.position.x) /
      visible_.info.resolution
  );
  robotMapPos_.y = static_cast<int>(
      (currentPose_.position.y - visible_.info.origin.position.y) /
      visible_.info.resolution
  );
}

void GlobalPlanner::replanCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  if (!msg || !msg->data) {
    return;
  }

  replanNeeded_ = true;
  RCLCPP_INFO(get_logger(), "Replanning requested by path follower");
}

void GlobalPlanner::explorationEnabledCallback(
    const std_msgs::msg::Bool::SharedPtr msg
) {
  if (!msg) {
    return;
  }

  const bool wasEnabled = explorationEnabled_;
  explorationEnabled_ = msg->data;
  if (!wasEnabled && explorationEnabled_) {
    replanNeeded_ = true;
  }
}

bool GlobalPlanner::mapsReady() const {
  return visibleReceived_ && predictedReceived_ && confidenceReceived_ &&
         odomReceived_;
}

// ─── Small geometry helpers used across the split TUs ────────────────────────

std::pair<int, int> GlobalPlanner::worldToGrid(
    double x, double y, const nav_msgs::msg::OccupancyGrid& map
) const {
  const int gridX =
      static_cast<int>((x - map.info.origin.position.x) / map.info.resolution);
  const int gridY =
      static_cast<int>((y - map.info.origin.position.y) / map.info.resolution);
  return {gridX, gridY};
}

std::pair<int, int> GlobalPlanner::worldToGrid(
    const Vertex& vertex, const nav_msgs::msg::OccupancyGrid& map
) const {
  return worldToGrid(vertex.x, vertex.y, map);
}

nav_msgs::msg::OccupancyGrid GlobalPlanner::padVisibleToPredicted(
    const nav_msgs::msg::OccupancyGrid& visible,
    const nav_msgs::msg::OccupancyGrid& predicted
) const {
  const int predWidth  = static_cast<int>(predicted.info.width);
  const int predHeight = static_cast<int>(predicted.info.height);
  const int visWidth   = static_cast<int>(visible.info.width);
  const int visHeight  = static_cast<int>(visible.info.height);

  const int expX = static_cast<int>(std::round(
      (visible.info.origin.position.x - predicted.info.origin.position.x) /
      visible.info.resolution));
  const int expY = static_cast<int>(std::round(
      (visible.info.origin.position.y - predicted.info.origin.position.y) /
      visible.info.resolution));

  nav_msgs::msg::OccupancyGrid padded;
  padded.header   = visible.header;
  padded.info     = predicted.info;
  padded.data.assign(predWidth * predHeight, -1);

  // Precompute the x-axis copy range once; skip per-cell bounds checks inside the loop.
  const int vxStart   = std::max(0, -expX);
  const int vxEnd     = std::min(visWidth, predWidth - expX);
  const int copyCount = vxEnd - vxStart;
  if (copyCount > 0) {
    const int pxStart = vxStart + expX;
    for (int vy = 0; vy < visHeight; ++vy) {
      const int py = vy + expY;
      if (py < 0 || py >= predHeight) {
        continue;
      }
      std::memcpy(
          padded.data.data() + py * predWidth + pxStart,
          visible.data.data() + vy * visWidth + vxStart,
          static_cast<std::size_t>(copyCount)
      );
    }
  }

  return padded;
}

double GlobalPlanner::getYawFromQuaternion(const geometry_msgs::msg::Quaternion& quat
) const {
  const double sinyCosp = 2.0 * (quat.w * quat.z + quat.x * quat.y);
  const double cosyCosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z);
  return std::atan2(sinyCosp, cosyCosp);
}
