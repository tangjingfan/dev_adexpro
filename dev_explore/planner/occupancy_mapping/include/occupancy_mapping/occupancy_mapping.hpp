#ifndef OCCUPANCY_MAPPING_OCCUPANCY_MAPPING_HPP
#define OCCUPANCY_MAPPING_OCCUPANCY_MAPPING_HPP

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// octomap_server-style 2D occupancy mapping, but free space is filled by
// angular-sector ray casting (swept-empty => free) instead of only along rays
// that hit a return. Occupied is sticky. Drop-in /projected_map publisher.
class OccupancyMapping : public rclcpp::Node {
public:
  OccupancyMapping();

private:
  void declareParameters();
  void loadParameters();
  void initializeSubscribers();
  void initializePublishers();
  void initializeTimers();

  void scanCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void publishTimerCallback();

  // Grow the global grid so [minX,maxX] x [minY,maxY] (world coords) fits,
  // preserving existing cells. Lazily allocates on first call.
  void ensureBounds(double minX, double minY, double maxX, double maxY);

  inline void worldToCell(double wx, double wy, int& cx, int& cy) const {
    cx = static_cast<int>(std::floor((wx - originX_) / resolution_));
    cy = static_cast<int>(std::floor((wy - originY_) / resolution_));
  }

  inline void integrateAt(int idx, float delta) {
    observed_[idx] = 1;
    float v = logOdds_[idx] + delta;
    logOdds_[idx] = v < lMin_ ? lMin_ : (v > lMax_ ? lMax_ : v);
  }

  void castFreeRay(int x0, int y0, int x1, int y1, bool stopBeforeEnd);

  // Parameters
  std::string frameId_;
  std::string scanTopic_;
  std::string odomTopic_;
  std::string mapTopic_;
  double resolution_{0.5};
  double maxRange_{20.0};
  double obstacleMinZ_{0.2};
  double obstacleMaxZ_{2.5};
  double publishPeriod_{0.5};
  double growMargin_{5.0};
  double wallSealDepth_{0.5};    // max range jump across a gap still treated as one wall
  double wallSealMaxDeg_{20.0};  // max angular span of a gap still sealed as wall

  float lHit_{0.85f};
  float lMiss_{-0.4f};
  float lMin_{-2.0f};
  float lMax_{3.5f};
  float lOccThres_{0.0f};  // log-odds above this -> occupied in the output

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr scanSub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr       odomSub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr     mapPub_;
  rclcpp::TimerBase::SharedPtr                                   publishTimer_;

  // Global grid state (row-major, idx = y * width_ + x)
  std::vector<float>   logOdds_;
  std::vector<uint8_t> observed_;
  int    width_{0};
  int    height_{0};
  double originX_{0.0};  // world coord of cell (0,0) lower-left corner
  double originY_{0.0};
  bool   gridInit_{false};

  // Latest robot pose (= ray origin)
  double robotX_{0.0};
  double robotY_{0.0};
  bool   haveOdom_{false};

  std::mutex mapMutex_;
};

#endif  // OCCUPANCY_MAPPING_OCCUPANCY_MAPPING_HPP
