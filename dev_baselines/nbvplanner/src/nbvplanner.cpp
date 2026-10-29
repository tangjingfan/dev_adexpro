#include "nbvplanner/nbvplanner.hpp"

#include "geometry_msgs/msg/point_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>

NBVPlanner::NBVPlanner() : Node("nbvplanner") {
  declareParameters();
  loadParameters();
  initializeSubscribers();
  initializePublishers();
  initializeTimers();
  RCLCPP_INFO(get_logger(), "NbvPlanner node initialized");
}

void NBVPlanner::declareParameters() {
  declare_parameter("vehicle_name", "av1");
  declare_parameter("planning_cycle", 2.0);
  declare_parameter("goal_reach_threshold", 1.0);
  declare_parameter("waypoint_topic", std::string("/way_point"));
  declare_parameter("exploration_enabled_topic", "/exploration_enabled");
  declare_parameter("exploration_enabled_initial", false);

  // Bounding box is derived from the map at runtime — no static params needed.

  // Vehicle
  declare_parameter("system.vehicle_radius",   0.5);

  // RRT tree
  declare_parameter("nbvp.tree.extension_range",    1.5);
  declare_parameter("nbvp.tree.initial_iterations",  15);
  declare_parameter("nbvp.tree.cutoff_iterations",  500);
  declare_parameter("nbvp.tree.allow_unknown_traversal", false);

  // Gain
  declare_parameter("nbvp.gain.range",             3.0);
  declare_parameter("nbvp.gain.unmapped",          1.0);
  declare_parameter("nbvp.gain.free",              0.0);
  declare_parameter("nbvp.gain.degressive_coeff",  0.25);
  declare_parameter("nbvp.gain.zero",              0.05);

  // Map
  declare_parameter("occupied_threshold", 50);

}

void NBVPlanner::loadParameters() {
  vehicleName_             = get_parameter("vehicle_name").as_string();
  planningCycle_           = get_parameter("planning_cycle").as_double();
  goalReachThreshold_      = get_parameter("goal_reach_threshold").as_double();
  waypointTopic_           = get_parameter("waypoint_topic").as_string();
  explorationEnabledTopic_ = get_parameter("exploration_enabled_topic").as_string();
  explorationEnabled_      = get_parameter("exploration_enabled_initial").as_bool();

  // bbox is set dynamically in runPlanning() from the map — skip here.

  rrtParams_.vehicleRadius         = get_parameter("system.vehicle_radius").as_double();
  rrtParams_.extensionRange        = get_parameter("nbvp.tree.extension_range").as_double();
  rrtParams_.initIterations        = get_parameter("nbvp.tree.initial_iterations").as_int();
  rrtParams_.cutoffIterations      = get_parameter("nbvp.tree.cutoff_iterations").as_int();
  rrtParams_.allowUnknownTraversal = get_parameter("nbvp.tree.allow_unknown_traversal").as_bool();
  rrtParams_.gainRange             = get_parameter("nbvp.gain.range").as_double();
  rrtParams_.igUnmapped            = get_parameter("nbvp.gain.unmapped").as_double();
  rrtParams_.igFree                = get_parameter("nbvp.gain.free").as_double();
  rrtParams_.degressiveCoeff       = get_parameter("nbvp.gain.degressive_coeff").as_double();
  rrtParams_.zeroGain              = get_parameter("nbvp.gain.zero").as_double();
  rrtParams_.occupiedThreshold     = get_parameter("occupied_threshold").as_int();

  tree_.setParams(rrtParams_);
}

void NBVPlanner::initializeSubscribers() {
  visibleSub_ = create_subscription<nav_msgs::msg::OccupancyGrid>("/visible_map", 10, std::bind(&NBVPlanner::visibleCallback, this, std::placeholders::_1));

  odomSub_ = create_subscription<nav_msgs::msg::Odometry>("/state_estimation", 10, std::bind(&NBVPlanner::odomCallback, this, std::placeholders::_1));

  rclcpp::QoS explorationQos(1);
  explorationQos.transient_local();
  explorationQos.reliable();
  explorationEnabledSub_ = create_subscription<std_msgs::msg::Bool>(explorationEnabledTopic_, explorationQos, std::bind(&NBVPlanner::explorationEnabledCallback, this, std::placeholders::_1));
}

void NBVPlanner::initializePublishers() {
  waypointPub_ = create_publisher<geometry_msgs::msg::PointStamped>(waypointTopic_, 10);
  treePub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/nbvp_tree", 10);
}

void NBVPlanner::initializeTimers() {
  const auto period = std::chrono::milliseconds(std::max(1, static_cast<int>(planningCycle_ * 1000.0)));
  timer_ = create_wall_timer(period, std::bind(&NBVPlanner::timerCallback, this));

  // Check goal-reach at ~5 Hz so replanning triggers immediately on arrival.
  goalCheckTimer_ = create_wall_timer(std::chrono::milliseconds(200), std::bind(&NBVPlanner::goalCheckCallback, this));
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void NBVPlanner::explorationEnabledCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  const bool wasEnabled = explorationEnabled_;
  explorationEnabled_ = msg->data;
  if (!wasEnabled && explorationEnabled_) {
    replanFlag_ = true;
    RCLCPP_INFO(get_logger(), "Exploration enabled — planning immediately");
  }
}

void NBVPlanner::visibleCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(mapMutex_);
  visible_         = *msg;
  visibleReceived_ = true;
}

void NBVPlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  // Extract (x, y, yaw) from the odometry message.
  const auto& pos = msg->pose.pose.position;
  const auto& q   = msg->pose.pose.orientation;

  // Convert quaternion to yaw (rotation around Z)
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  const double yaw       = std::atan2(siny_cosp, cosy_cosp);

  std::lock_guard<std::mutex> lock(mapMutex_);
  currentState_ = Eigen::Vector3d(pos.x, pos.y, yaw);
  odomReceived_ = true;
}

void NBVPlanner::timerCallback() {
  if (!explorationEnabled_) return;

  std::lock_guard<std::mutex> lock(mapMutex_);
  if (!visibleReceived_ || !odomReceived_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for map and odometry...");
    return;
  }

  runPlanning();
}

void NBVPlanner::goalCheckCallback() {
  if (!explorationEnabled_) return;

  std::lock_guard<std::mutex> lock(mapMutex_);
  if (!visibleReceived_ || !odomReceived_ || !hasGoal_) return;

  const double dist = std::hypot(currentState_[0] - currentGoal_[0], currentState_[1] - currentGoal_[1]);
  if (dist < goalReachThreshold_) {
    hasGoal_ = false;
    RCLCPP_DEBUG(get_logger(), "NBV: goal reached (dist=%.2f), replanning", dist);
    // Reset the period timer so it doesn't fire again shortly after.
    timer_->reset();
    runPlanning();
  }
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

void NBVPlanner::runPlanning() {
  // Derive the sampling bbox directly from the current visible map extent.
  // This ensures RRT always samples within the known map area.
  const auto& info = visible_.info;
  rrtParams_.minX = info.origin.position.x;
  rrtParams_.minY = info.origin.position.y;
  rrtParams_.maxX = rrtParams_.minX + info.width  * info.resolution;
  rrtParams_.maxY = rrtParams_.minY + info.height * info.resolution;

  tree_.setParams(rrtParams_);
  tree_.setMap(visible_);

  // If the robot's current position falls in an unknown or occupied cell,
  // snap the root to the nearest known-free cell before building the tree.
  auto root = currentState_;
  if (tree_.snapToFreeSpace(root)) {
    RCLCPP_DEBUG(get_logger(), "NBV: root snapped (%.2f,%.2f) → (%.2f,%.2f)", currentState_[0], currentState_[1], root[0], root[1]);
  }
  tree_.setRoot(root);
  tree_.clear();
  tree_.initialize();  // re-inserts memorized best branch

  // Count only nodes added by fresh sampling in this planning cycle.
  // Re-inserted memory nodes should not satisfy iteration/cutoff criteria.
  const int baseCounter = tree_.getCounter();

  int loopCount = 0;
  while (rclcpp::ok()) {
    const int newNodeCount = tree_.getCounter() - baseCounter;

    if (newNodeCount > rrtParams_.cutoffIterations) {
      RCLCPP_WARN(get_logger(), "NBV: cutoff reached (%d iterations), no gain found", newNodeCount);
      return;
    }
    // Guard against infinite loop when sampling keeps failing
    if (loopCount > 1000 * (newNodeCount + 1)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "NBV: too many failed sample attempts — skipping cycle");
      break;
    }

    tree_.iterate();
    loopCount++;

    // Stop when we have at least one promising node AND enough iterations.
    const int freshNodeCount = tree_.getCounter() - baseCounter;
    if (tree_.gainFound() && freshNodeCount >= rrtParams_.initIterations) break;
  }

  auto path = tree_.getBestPath();

  if (path.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "NBV: no path found this cycle");
    return;
  }

  RCLCPP_DEBUG(get_logger(), "NBV: planned %zu poses after %d tree nodes",
               path.size(), tree_.getCounter());

  auto bestEdge = tree_.getBestEdge();
  tree_.memorizeBestBranch();

  publishWaypoint(bestEdge);
  publishVisualization(path);
}

// ---------------------------------------------------------------------------
// Publishing
// ---------------------------------------------------------------------------

void NBVPlanner::publishWaypoint(
    const nbvplanner::RrtTree::State& state) {
  geometry_msgs::msg::PointStamped msg;
  msg.header.stamp    = now();
  msg.header.frame_id = "map";
  msg.point.x = state[0];
  msg.point.y = state[1];
  msg.point.z = 0.0;
  waypointPub_->publish(msg);

  currentGoal_ = Eigen::Vector2d(state[0], state[1]);
  hasGoal_ = true;
}

void NBVPlanner::publishVisualization(
    const std::vector<nbvplanner::RrtTree::State>& bestPath) const {
  visualization_msgs::msg::MarkerArray arr;

  // Delete all previous markers before redrawing
  visualization_msgs::msg::Marker del;
  del.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(del);

  const auto stamp = now();

  // --- 1. Full RRT tree: one LINE_STRIP edge per node (grey-blue) ---
  int id = 0;
  for (const auto& node : tree_.getNodes()) {
    if (!node->parent) continue;

    visualization_msgs::msg::Marker edge;
    edge.header.frame_id = "map";
    edge.header.stamp    = stamp;
    edge.ns              = "nbvp_tree";
    edge.id              = id++;
    edge.type            = visualization_msgs::msg::Marker::LINE_STRIP;
    edge.action          = visualization_msgs::msg::Marker::ADD;
    edge.scale.x         = 0.15;
    edge.color.r         = 0.4f;
    edge.color.g         = 0.6f;
    edge.color.b         = 1.0f;
    edge.color.a         = 0.8f;
    geometry_msgs::msg::Point p0, p1;
    p0.x = node->parent->state[0];
    p0.y = node->parent->state[1];
    p1.x = node->state[0];
    p1.y = node->state[1];
    edge.points = {p0, p1};
    arr.markers.push_back(edge);
  }

  // --- 2. Chosen path: thick orange LINE_STRIP ---
  if (!bestPath.empty()) {
    visualization_msgs::msg::Marker path;
    path.header.frame_id = "map";
    path.header.stamp    = stamp;
    path.ns              = "nbvp_best_path";
    path.id              = 0;
    path.type            = visualization_msgs::msg::Marker::LINE_STRIP;
    path.action          = visualization_msgs::msg::Marker::ADD;
    path.scale.x         = 0.2;
    path.color.r         = 1.0f;
    path.color.g         = 0.55f;
    path.color.b         = 0.0f;
    path.color.a         = 1.0f;
    for (const auto& s : bestPath) {
      geometry_msgs::msg::Point p;
      p.x = s[0];
      p.y = s[1];
      path.points.push_back(p);
    }
    arr.markers.push_back(path);

    // --- 3. Goal sphere at the end of the chosen path ---
    // visualization_msgs::msg::Marker goal;
    // goal.header.frame_id = "map";
    // goal.header.stamp    = stamp;
    // goal.ns              = "nbvp_best_path";
    // goal.id              = 1;
    // goal.type            = visualization_msgs::msg::Marker::SPHERE;
    // goal.action          = visualization_msgs::msg::Marker::ADD;
    // goal.pose.position.x = bestPath.back()[0];
    // goal.pose.position.y = bestPath.back()[1];
    // goal.pose.orientation.w = 1.0;
    // goal.scale.x = goal.scale.y = goal.scale.z = 0.3;
    // goal.color.r = 1.0f;
    // goal.color.g = 0.3f;
    // goal.color.b = 0.0f;
    // goal.color.a = 1.0f;
    // goal.lifetime = rclcpp::Duration::from_seconds(planningCycle_ * 1.5);
    // arr.markers.push_back(goal);
  }

  treePub_->publish(arr);
}
