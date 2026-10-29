#include "occupancy_mapping/occupancy_mapping.hpp"

#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace {
inline float probToLogOdds(double p) {
  p = std::min(std::max(p, 1e-3), 1.0 - 1e-3);
  return static_cast<float>(std::log(p / (1.0 - p)));
}
}  // namespace

OccupancyMapping::OccupancyMapping() : Node("occupancy_mapping_node") {
  declareParameters();
  loadParameters();
  initializeSubscribers();
  initializePublishers();
  initializeTimers();
  RCLCPP_INFO(get_logger(),
              "occupancy_mapping_node initialized (res=%.2f, max_range=%.1f, "
              "obstacle_z=[%.2f, %.2f])",
              resolution_, maxRange_, obstacleMinZ_, obstacleMaxZ_);
}

void OccupancyMapping::declareParameters() {
  declare_parameter("frame_id", "map");
  declare_parameter("scan_topic", "/registered_scan");
  declare_parameter("odom_topic", "/state_estimation");
  declare_parameter("map_topic", "/projected_map");
  declare_parameter("resolution", 0.5);
  declare_parameter("max_range", 20.0);
  declare_parameter("obstacle_min_z", 0.2);
  declare_parameter("obstacle_max_z", 2.5);
  declare_parameter("publish_period", 0.5);
  declare_parameter("grow_margin", 5.0);
  declare_parameter("wall_seal_depth", 0.5);
  declare_parameter("wall_seal_max_deg", 20.0);
  declare_parameter("prob_hit", 0.7);
  declare_parameter("prob_miss", 0.4);
  declare_parameter("clamp_min", 0.12);
  declare_parameter("clamp_max", 0.97);
  declare_parameter("occupancy_prob_thresh", 0.5);
}

void OccupancyMapping::loadParameters() {
  frameId_       = get_parameter("frame_id").as_string();
  scanTopic_     = get_parameter("scan_topic").as_string();
  odomTopic_     = get_parameter("odom_topic").as_string();
  mapTopic_      = get_parameter("map_topic").as_string();
  resolution_    = get_parameter("resolution").as_double();
  maxRange_      = get_parameter("max_range").as_double();
  obstacleMinZ_  = get_parameter("obstacle_min_z").as_double();
  obstacleMaxZ_  = get_parameter("obstacle_max_z").as_double();
  publishPeriod_   = get_parameter("publish_period").as_double();
  growMargin_      = get_parameter("grow_margin").as_double();
  wallSealDepth_   = get_parameter("wall_seal_depth").as_double();
  wallSealMaxDeg_  = get_parameter("wall_seal_max_deg").as_double();

  lHit_       = probToLogOdds(get_parameter("prob_hit").as_double());
  lMiss_      = probToLogOdds(get_parameter("prob_miss").as_double());
  lMin_       = probToLogOdds(get_parameter("clamp_min").as_double());
  lMax_       = probToLogOdds(get_parameter("clamp_max").as_double());
  lOccThres_  = probToLogOdds(get_parameter("occupancy_prob_thresh").as_double());
}

void OccupancyMapping::initializeSubscribers() {
  scanSub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      scanTopic_, rclcpp::SensorDataQoS(),
      std::bind(&OccupancyMapping::scanCallback, this, std::placeholders::_1));

  odomSub_ = create_subscription<nav_msgs::msg::Odometry>(
      odomTopic_, 10,
      std::bind(&OccupancyMapping::odomCallback, this, std::placeholders::_1));
}

void OccupancyMapping::initializePublishers() {
  mapPub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(mapTopic_, 1);
}

void OccupancyMapping::initializeTimers() {
  const auto period = std::chrono::milliseconds(
      std::max(1, static_cast<int>(publishPeriod_ * 1000.0)));
  publishTimer_ =
      create_wall_timer(period, std::bind(&OccupancyMapping::publishTimerCallback, this));
}

void OccupancyMapping::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mapMutex_);
  robotX_   = msg->pose.pose.position.x;
  robotY_   = msg->pose.pose.position.y;
  haveOdom_ = true;
}

void OccupancyMapping::ensureBounds(double minX, double minY, double maxX,
                                    double maxY) {
  minX -= growMargin_;
  minY -= growMargin_;
  maxX += growMargin_;
  maxY += growMargin_;

  if (!gridInit_) {
    originX_ = std::floor(minX / resolution_) * resolution_;
    originY_ = std::floor(minY / resolution_) * resolution_;
    width_   = std::max(1, static_cast<int>(std::ceil((maxX - originX_) / resolution_)));
    height_  = std::max(1, static_cast<int>(std::ceil((maxY - originY_) / resolution_)));
    logOdds_.assign(static_cast<size_t>(width_) * height_, 0.0f);
    observed_.assign(static_cast<size_t>(width_) * height_, 0);
    gridInit_ = true;
    return;
  }

  const double curMaxX = originX_ + width_ * resolution_;
  const double curMaxY = originY_ + height_ * resolution_;
  if (minX >= originX_ && minY >= originY_ && maxX <= curMaxX && maxY <= curMaxY) {
    return;  // already covered
  }

  const double newOriginX = std::min(originX_, std::floor(minX / resolution_) * resolution_);
  const double newOriginY = std::min(originY_, std::floor(minY / resolution_) * resolution_);
  const double newMaxX    = std::max(curMaxX, std::ceil(maxX / resolution_) * resolution_);
  const double newMaxY    = std::max(curMaxY, std::ceil(maxY / resolution_) * resolution_);
  const int newW = static_cast<int>(std::lround((newMaxX - newOriginX) / resolution_));
  const int newH = static_cast<int>(std::lround((newMaxY - newOriginY) / resolution_));

  std::vector<float>   newLog(static_cast<size_t>(newW) * newH, 0.0f);
  std::vector<uint8_t> newObs(static_cast<size_t>(newW) * newH, 0);
  const int offX = static_cast<int>(std::lround((originX_ - newOriginX) / resolution_));
  const int offY = static_cast<int>(std::lround((originY_ - newOriginY) / resolution_));
  for (int y = 0; y < height_; ++y) {
    const size_t srcRow = static_cast<size_t>(y) * width_;
    const size_t dstRow = static_cast<size_t>(y + offY) * newW + offX;
    for (int x = 0; x < width_; ++x) {
      newLog[dstRow + x] = logOdds_[srcRow + x];
      newObs[dstRow + x] = observed_[srcRow + x];
    }
  }
  logOdds_.swap(newLog);
  observed_.swap(newObs);
  width_   = newW;
  height_  = newH;
  originX_ = newOriginX;
  originY_ = newOriginY;
}

void OccupancyMapping::castFreeRay(int x0, int y0, int x1, int y1,
                                   bool stopBeforeEnd) {
  const int dx = std::abs(x1 - x0);
  const int dy = std::abs(y1 - y0);
  const int sx = x0 < x1 ? 1 : -1;
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx - dy;
  int x = x0, y = y0;
  while (true) {
    const bool atEnd = (x == x1 && y == y1);
    if (atEnd && stopBeforeEnd) break;
    if (x >= 0 && x < width_ && y >= 0 && y < height_) {
      const int idx = y * width_ + x;
      if (logOdds_[idx] <= lOccThres_) integrateAt(idx, lMiss_);  // occupied is sticky
    }
    if (atEnd) break;
    const int e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 <  dx) { err += dx; y += sy; }
  }
}

void OccupancyMapping::scanCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mapMutex_);
  if (!haveOdom_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Waiting for odometry before mapping...");
    return;
  }

  const double rx = robotX_;
  const double ry = robotY_;
  const double maxRangeSq = maxRange_ * maxRange_;

  // One free ray per azimuth bin; arc stays below half a cell at max range.
  const int nBins = std::max(
      720, static_cast<int>(std::ceil(4.0 * M_PI * maxRange_ / resolution_)));
  const double binScale = nBins / (2.0 * M_PI);
  const float kInf = std::numeric_limits<float>::infinity();
  std::vector<float>  binRange(nBins, kInf);
  std::vector<double> binX(nBins, 0.0);
  std::vector<double> binY(nBins, 0.0);

  std::vector<std::pair<double, double>> hitPts;
  sensor_msgs::PointCloud2ConstIterator<float> ix(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iy(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iz(*msg, "z");
  for (; ix != ix.end(); ++ix, ++iy, ++iz) {
    const double px = *ix, py = *iy, pz = *iz;
    if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;
    if (pz < obstacleMinZ_ || pz > obstacleMaxZ_) continue;  // ground / ceiling
    const double dx = px - rx;
    const double dy = py - ry;
    const double rangeSq = dx * dx + dy * dy;
    if (rangeSq > maxRangeSq || rangeSq < 1e-9) continue;

    int b = static_cast<int>((std::atan2(dy, dx) + M_PI) * binScale);
    if (b < 0) b = 0;
    if (b >= nBins) b = nBins - 1;
    const double range = std::sqrt(rangeSq);
    if (range < binRange[b]) {
      binRange[b] = static_cast<float>(range);
      binX[b] = px;
      binY[b] = py;
    }
    hitPts.emplace_back(px, py);
  }

  ensureBounds(rx - maxRange_, ry - maxRange_, rx + maxRange_, ry + maxRange_);

  int rcx, rcy;
  worldToCell(rx, ry, rcx, rcy);

  // Per-bin stop point. Bin with a return stops at it. Empty bin in a narrow
  // same-wall gap is interpolated so free stops at the wall line (no leak);
  // empty bin across a real opening fills to max range (swept-empty => free).
  std::vector<double>  stopX(nBins), stopY(nBins);
  std::vector<uint8_t> stopBefore(nBins, 0);
  auto maxEnd = [&](int b, double& ex, double& ey) {
    const double a = (b + 0.5) / binScale - M_PI;
    ex = rx + maxRange_ * std::cos(a);
    ey = ry + maxRange_ * std::sin(a);
  };

  std::vector<int> occ;
  occ.reserve(nBins);
  for (int b = 0; b < nBins; ++b) {
    if (std::isfinite(binRange[b])) occ.push_back(b);
  }

  if (occ.size() < 2) {
    for (int b = 0; b < nBins; ++b) {
      if (std::isfinite(binRange[b])) {
        stopX[b] = binX[b];
        stopY[b] = binY[b];
        stopBefore[b] = 1;
      } else {
        maxEnd(b, stopX[b], stopY[b]);
      }
    }
  } else {
    for (int b : occ) {
      stopX[b] = binX[b];
      stopY[b] = binY[b];
      stopBefore[b] = 1;
    }
    const double sealBins = wallSealMaxDeg_ * M_PI / 180.0 * binScale;
    for (size_t i = 0; i < occ.size(); ++i) {
      const int cur = occ[i];
      const int nxt = occ[(i + 1) % occ.size()];
      int gapLen = nxt - cur - 1;
      if (gapLen < 0) gapLen += nBins;  // wrap-around
      if (gapLen <= 0) continue;
      const bool sealable =
          std::fabs(binRange[cur] - binRange[nxt]) <= wallSealDepth_ &&
          gapLen <= sealBins;
      for (int k = 1; k <= gapLen; ++k) {
        const int b = (cur + k) % nBins;
        if (sealable) {
          const double t = static_cast<double>(k) / (gapLen + 1);
          stopX[b] = binX[cur] + t * (binX[nxt] - binX[cur]);
          stopY[b] = binY[cur] + t * (binY[nxt] - binY[cur]);
          stopBefore[b] = 1;
        } else {
          maxEnd(b, stopX[b], stopY[b]);
          stopBefore[b] = 0;
        }
      }
    }
  }

  for (int b = 0; b < nBins; ++b) {
    int ecx, ecy;
    worldToCell(stopX[b], stopY[b], ecx, ecy);
    castFreeRay(rcx, rcy, ecx, ecy, stopBefore[b] != 0);
  }

  for (const auto& [px, py] : hitPts) {
    int cx, cy;
    worldToCell(px, py, cx, cy);
    if (cx >= 0 && cx < width_ && cy >= 0 && cy < height_) {
      integrateAt(cy * width_ + cx, lHit_);
    }
  }
}

void OccupancyMapping::publishTimerCallback() {
  std::lock_guard<std::mutex> lock(mapMutex_);
  if (!gridInit_) return;

  nav_msgs::msg::OccupancyGrid grid;
  grid.header.frame_id = frameId_;
  grid.header.stamp = now();
  grid.info.resolution = static_cast<float>(resolution_);
  grid.info.width  = static_cast<uint32_t>(width_);
  grid.info.height = static_cast<uint32_t>(height_);
  grid.info.origin.position.x = originX_;
  grid.info.origin.position.y = originY_;
  grid.info.origin.orientation.w = 1.0;

  const size_t n = static_cast<size_t>(width_) * height_;
  grid.data.resize(n);
  for (size_t i = 0; i < n; ++i) {
    if (!observed_[i]) {
      grid.data[i] = -1;
    } else {
      grid.data[i] = logOdds_[i] > lOccThres_ ? 100 : 0;
    }
  }
  mapPub_->publish(grid);
}
