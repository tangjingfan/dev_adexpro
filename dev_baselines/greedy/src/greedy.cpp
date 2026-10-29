#include "../include/greedy.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

Greedy::Greedy() : Node("greedy") {
  declareParameters();
  loadParameters();
  initializeSubscribers();
  initializePublishers();
  initializeTimer();
  RCLCPP_INFO(get_logger(), "Greedy node initialized");
}

void Greedy::declareParameters() {
  declare_parameter("vehicle_name", "av1");
  declare_parameter("planning_cycle", 1.0);
  declare_parameter("cluster_size", 6);
  declare_parameter("occupied_threshold", 50);
  declare_parameter("distance_threshold", 3.0);
  declare_parameter("travel_distance_threshold", 3.0);
  declare_parameter("exploration_enabled_topic", "/exploration_enabled");
  declare_parameter("exploration_enabled_initial", false);
  declare_parameter("replan_cycle", 10.0);
}

void Greedy::loadParameters() {
  vehicleName_             = get_parameter("vehicle_name").as_string();
  planningCycle_           = get_parameter("planning_cycle").as_double();
  clusterSize_             = get_parameter("cluster_size").as_int();
  occupiedThreshold_       = get_parameter("occupied_threshold").as_int();
  distanceThreshold_       = get_parameter("distance_threshold").as_double();
  travelDistanceThreshold_ = get_parameter("travel_distance_threshold").as_double();
  explorationEnabledTopic_ = get_parameter("exploration_enabled_topic").as_string();
  explorationEnabled_      = get_parameter("exploration_enabled_initial").as_bool();
  replanCycle_             = get_parameter("replan_cycle").as_double();
}

void Greedy::initializeSubscribers() {
  visibleSub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/visible_map", 10,
      std::bind(&Greedy::visibleCallback, this, std::placeholders::_1));

  odomSub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/state_estimation", 10,
      std::bind(&Greedy::odomCallback, this, std::placeholders::_1));

  rclcpp::QoS explorationControlQos(1);
  explorationControlQos.transient_local();
  explorationControlQos.reliable();
  explorationEnabledSub_ = create_subscription<std_msgs::msg::Bool>(
      explorationEnabledTopic_, explorationControlQos,
      std::bind(&Greedy::explorationEnabledCallback, this, std::placeholders::_1));
}

void Greedy::initializePublishers() {
  frontierPub_    = create_publisher<sensor_msgs::msg::PointCloud2>("/greedy_frontiers", 10);
  rawFrontierPub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/greedy_raw_frontiers", 10);
  pathPub_        = create_publisher<nav_msgs::msg::Path>("/solution_path", 10);
}

void Greedy::initializeTimer() {
  const auto period = std::chrono::milliseconds(
      std::max(1, static_cast<int>(planningCycle_ * 1000.0)));
  timer_ = create_wall_timer(period, std::bind(&Greedy::timerCallback, this));

  const auto replanPeriod = std::chrono::milliseconds(
      std::max(1, static_cast<int>(replanCycle_ * 1000.0)));
  replanTimer_ = create_wall_timer(replanPeriod, [this]() {
    if (explorationEnabled_) {
      replanFlag_ = true;
      RCLCPP_INFO(get_logger(), "Periodic replan triggered");
    }
  });
}

void Greedy::explorationEnabledCallback(const std_msgs::msg::Bool::SharedPtr msg) {
  if (!msg) return;
  const bool wasEnabled = explorationEnabled_;
  explorationEnabled_ = msg->data;
  if (!wasEnabled && explorationEnabled_) {
    replanFlag_ = true;
  }
}

void Greedy::timerCallback() {
  if (!explorationEnabled_) return;

  std::lock_guard<std::mutex> lock(mapMutex_);
  if (!visibleReceived_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for map data...");
    return;
  }

  rawFrontiers_ = findFrontiers();
  frontiers_    = clusterFrontiers(rawFrontiers_, clusterSize_);
  filterUnreachableFrontiers();

  publishFrontierCloud();
  publishRawFrontierCloud();

  if (replanFlag_) {
    assignNearestFrontier();
  }
}

void Greedy::visibleCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  if (!msg) return;
  const int expected = static_cast<int>(msg->info.width * msg->info.height);
  if (static_cast<int>(msg->data.size()) != expected) {
    RCLCPP_ERROR(get_logger(), "Map size mismatch: %zu vs %d", msg->data.size(), expected);
    return;
  }

  std::lock_guard<std::mutex> lock(mapMutex_);
  visible_         = *msg;
  width_           = static_cast<int>(msg->info.width);
  height_          = static_cast<int>(msg->info.height);
  resolution_      = static_cast<float>(msg->info.resolution);
  originX_         = static_cast<float>(msg->info.origin.position.x);
  originY_         = static_cast<float>(msg->info.origin.position.y);
  visibleReceived_ = true;
}

void Greedy::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  if (!msg) return;

  std::lock_guard<std::mutex> lock(mapMutex_);
  currentPose_ = msg->pose.pose;

  if (!explorationEnabled_ || !visibleReceived_ || replanFlag_) return;

  if (checkArriveTarget()) {
    visitedFrontiers_.insert(targetFrontier_);
    targetFrontier_ = {0, 0};
    replanFlag_ = true;
    RCLCPP_INFO(get_logger(), "Arrived at frontier — replanning");
  } else if (checkTravelFarEnough()) {
    replanFlag_ = true;
    RCLCPP_INFO(get_logger(), "Traveled far enough — replanning");
  }
}

std::vector<Point2D> Greedy::findFrontiers() const {
  std::vector<Point2D> frontiers;
  for (int y = 1; y < height_ - 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      if (isFrontier(x, y)) {
        frontiers.push_back({x, y});
      }
    }
  }
  return frontiers;
}

std::vector<Point2D> Greedy::clusterFrontiers(
    const std::vector<Point2D>& rawFrontiers, int clusterSize) const
{
  if (clusterSize <= 0) return rawFrontiers;

  std::unordered_map<int64_t, std::vector<Point2D>> clusters;
  clusters.reserve(rawFrontiers.size() / 4 + 1);
  for (const auto& pt : rawFrontiers) {
    const int64_t key =
        (static_cast<int64_t>(pt.x / clusterSize) << 32) |
        static_cast<uint32_t>(pt.y / clusterSize);
    clusters[key].push_back(pt);
  }

  std::vector<Point2D> clustered;
  for (const auto& [key, points] : clusters) {
    (void)key;
    if (points.empty()) continue;

    double sumX = 0.0, sumY = 0.0;
    for (const auto& pt : points) { sumX += pt.x; sumY += pt.y; }
    const double cx = sumX / points.size();
    const double cy = sumY / points.size();

    double best = std::numeric_limits<double>::max();
    Point2D bestPt = points.front();
    for (const auto& pt : points) {
      const double d = (pt.x - cx) * (pt.x - cx) + (pt.y - cy) * (pt.y - cy);
      if (d < best) { best = d; bestPt = pt; }
    }
    clustered.push_back(bestPt);
  }
  return clustered;
}

void Greedy::filterUnreachableFrontiers() {
  constexpr std::array<std::pair<int, int>, 8> dirs = {{
    {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
  }};

  auto inBounds = [&](int x, int y) {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
  };
  auto isBlocked = [&](int x, int y) {
    const auto v = visible_.data[y * width_ + x];
    return v < 0 || v >= static_cast<int8_t>(occupiedThreshold_);
  };

  int robotX, robotY;
  worldToGrid(currentPose_, robotX, robotY);
  int startX = robotX;
  int startY = robotY;
  if (!inBounds(startX, startY) || isBlocked(startX, startY)) {
    if (!findNearestFree(robotX, robotY, startX, startY)) {
      frontiers_.clear();
      return;
    }
  }

  std::vector<uint8_t> reachable(static_cast<size_t>(width_ * height_), 0);
  std::queue<Point2D> q;
  q.push({startX, startY});
  reachable[startY * width_ + startX] = 1;

  while (!q.empty()) {
    const auto cur = q.front();
    q.pop();
    for (const auto& [dx, dy] : dirs) {
      const int nx = cur.x + dx;
      const int ny = cur.y + dy;
      if (!inBounds(nx, ny) || isBlocked(nx, ny)) continue;
      if (dx != 0 && dy != 0) {
        if (isBlocked(cur.x + dx, cur.y) || isBlocked(cur.x, cur.y + dy)) continue;
      }

      const int index = ny * width_ + nx;
      if (reachable[index]) continue;
      reachable[index] = 1;
      q.push({nx, ny});
    }
  }

  std::vector<Point2D> result;
  result.reserve(frontiers_.size());
  for (const auto& frontier : frontiers_) {
    if (!inBounds(frontier.x, frontier.y)) continue;
    if (reachable[frontier.y * width_ + frontier.x]) {
      result.push_back(frontier);
    }
  }
  frontiers_ = std::move(result);
}

// Returns A* path cost in grid steps, or -1 if unreachable.
// Uses flat g-score vector + min-heap; no shared_ptr node chains.
float Greedy::aStarCost(int startX, int startY, int goalX, int goalY) const {
  constexpr std::array<std::pair<int, int>, 8> dirs = {{
    {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
  }};

  auto inBounds  = [&](int x, int y) {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
  };
  auto isBlocked = [&](int x, int y) {
    const auto v = visible_.data[y * width_ + x];
    return v < 0 || v >= static_cast<int8_t>(occupiedThreshold_);
  };

  if (!inBounds(startX, startY) || !inBounds(goalX, goalY)) return -1.0f;
  if (isBlocked(goalX, goalY)) return -1.0f;

  const int mapSize  = width_ * height_;
  const int startIdx = startY * width_ + startX;
  const int goalIdx  = goalY  * width_ + goalX;

  std::vector<float> g(mapSize, std::numeric_limits<float>::max());
  g[startIdx] = 0.0f;

  auto h = [&](int x, int y) {
    return std::hypot(static_cast<float>(x - goalX), static_cast<float>(y - goalY));
  };

  using Entry = std::pair<float, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  open.push({h(startX, startY), startIdx});

  while (!open.empty()) {
    auto [f, curIdx] = open.top(); open.pop();
    if (curIdx == goalIdx) return g[goalIdx];

    const int cx = curIdx % width_;
    const int cy = curIdx / width_;
    for (auto [dx, dy] : dirs) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (!inBounds(nx, ny) || isBlocked(nx, ny)) continue;
      if (dx != 0 && dy != 0) {
        if (isBlocked(cx + dx, cy) || isBlocked(cx, cy + dy)) continue;
      }
      const float cost = (dx != 0 && dy != 0) ? 1.414f : 1.0f;
      const float ng   = g[curIdx] + cost;
      const int   ni   = ny * width_ + nx;
      if (ng < g[ni]) {
        g[ni] = ng;
        open.push({ng + h(nx, ny), ni});
      }
    }
  }
  return -1.0f;
}

void Greedy::assignNearestFrontier() {
  if (frontiers_.empty()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "No frontiers available");
    return;
  }

  int robotX, robotY;
  worldToGrid(currentPose_, robotX, robotY);

  int startX = robotX, startY = robotY;
  {
    const int idx = robotY * width_ + robotX;
    if (idx >= 0 && idx < static_cast<int>(visible_.data.size()) &&
        visible_.data[idx] >= occupiedThreshold_) {
      int freeX, freeY;
      if (findNearestFree(robotX, robotY, freeX, freeY)) {
        startX = freeX;
        startY = freeY;
      }
    }
  }

  float   bestCost = std::numeric_limits<float>::max();
  Point2D bestFrontier{0, 0};
  bool    found = false;

  for (const auto& frontier : frontiers_) {
    if (visitedFrontiers_.count(frontier)) continue;
    const float cost = aStarCost(startX, startY, frontier.x, frontier.y);
    if (cost > 0.0f && cost < bestCost) {
      bestCost     = cost;
      bestFrontier = frontier;
      found        = true;
    }
  }

  if (!found) {
    RCLCPP_WARN(get_logger(), "No reachable frontier found");
    return;
  }

  targetFrontier_ = bestFrontier;
  startPoint_     = {startX, startY};
  replanFlag_     = false;

  publishPath(startX, startY, targetFrontier_.x, targetFrontier_.y);
  RCLCPP_INFO(get_logger(), "Assigned frontier (%d,%d), A* cost=%.1f",
              targetFrontier_.x, targetFrontier_.y, bestCost);
}

bool Greedy::isFrontier(int x, int y) const {
  if (x <= 0 || x >= width_ - 1 || y <= 0 || y >= height_ - 1) return false;
  if (visible_.data[y * width_ + x] != 0) return false;

  int unknownCount = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      if (visible_.data[(y + dy) * width_ + (x + dx)] == -1) ++unknownCount;
    }
  }
  return unknownCount >= 2;
}

bool Greedy::checkArriveTarget() const {
  if (targetFrontier_.x == 0 && targetFrontier_.y == 0) return false;
  int rx, ry;
  worldToGrid(currentPose_, rx, ry);
  return std::hypot(rx - targetFrontier_.x, ry - targetFrontier_.y) <= distanceThreshold_;
}

bool Greedy::checkTravelFarEnough() const {
  if (targetFrontier_.x == 0 && targetFrontier_.y == 0) return false;
  int rx, ry;
  worldToGrid(currentPose_, rx, ry);
  return std::hypot(rx - startPoint_.x, ry - startPoint_.y) >= travelDistanceThreshold_;
}

void Greedy::worldToGrid(
    const geometry_msgs::msg::Pose& pose, int& gridX, int& gridY) const
{
  gridX = static_cast<int>(std::floor((pose.position.x - originX_) / resolution_));
  gridY = static_cast<int>(std::floor((pose.position.y - originY_) / resolution_));
}

bool Greedy::findNearestFree(int x, int y, int& freeX, int& freeY) const {
  constexpr std::array<std::pair<int, int>, 8> dirs = {{
    {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
  }};

  std::queue<Point2D> q;
  q.push({x, y});
  std::unordered_set<int64_t> visited;
  visited.insert(static_cast<int64_t>(y) * width_ + x);

  while (!q.empty()) {
    const auto cur = q.front(); q.pop();
    for (const auto& [dx, dy] : dirs) {
      const int nx = cur.x + dx;
      const int ny = cur.y + dy;
      if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
      const int64_t key = static_cast<int64_t>(ny) * width_ + nx;
      if (!visited.insert(key).second) continue;
      const auto v = visible_.data[ny * width_ + nx];
      if (v == 0) { freeX = nx; freeY = ny; return true; }
      if (v >= 0)  q.push({nx, ny});
    }
  }
  return false;
}

static void fillPointCloud(
    sensor_msgs::msg::PointCloud2& cloud,
    const std::vector<Point2D>& pts,
    float res, float ox, float oy, float z)
{
  sensor_msgs::PointCloud2Modifier mod(cloud);
  mod.setPointCloud2Fields(
      3,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32);
  mod.resize(pts.size());

  sensor_msgs::PointCloud2Iterator<float> ix(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iy(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iz(cloud, "z");
  for (const auto& pt : pts) {
    *ix = ox + (pt.x + 0.5f) * res; ++ix;
    *iy = oy + (pt.y + 0.5f) * res; ++iy;
    *iz = z;                         ++iz;
  }
}

void Greedy::publishFrontierCloud() const {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = visible_.header;
  cloud.header.stamp = now();
  cloud.height = 1;
  cloud.width  = static_cast<uint32_t>(frontiers_.size());
  fillPointCloud(cloud, frontiers_, resolution_, originX_, originY_,
                 static_cast<float>(currentPose_.position.z));
  frontierPub_->publish(cloud);
}

void Greedy::publishRawFrontierCloud() const {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header = visible_.header;
  cloud.header.stamp = now();
  cloud.height = 1;
  cloud.width  = static_cast<uint32_t>(rawFrontiers_.size());
  fillPointCloud(cloud, rawFrontiers_, resolution_, originX_, originY_,
                 static_cast<float>(currentPose_.position.z));
  rawFrontierPub_->publish(cloud);
}

void Greedy::publishPath(int startX, int startY, int goalX, int goalY) const {
  nav_msgs::msg::Path path;
  path.header.frame_id = visible_.header.frame_id;
  path.header.stamp    = now();
  const float z = static_cast<float>(currentPose_.position.z);

  auto makePose = [&](int gx, int gy) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header            = path.header;
    ps.pose.position.x   = originX_ + (gx + 0.5f) * resolution_;
    ps.pose.position.y   = originY_ + (gy + 0.5f) * resolution_;
    ps.pose.position.z   = z;
    ps.pose.orientation.w = 1.0;
    return ps;
  };

  path.poses.push_back(makePose(startX, startY));
  path.poses.push_back(makePose(goalX,  goalY));
  pathPub_->publish(path);
}
