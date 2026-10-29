#ifndef GREEDY_H
#define GREEDY_H

#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/bool.hpp"

#include <mutex>
#include <set>
#include <string>
#include <vector>

struct Point2D {
  int x{0};
  int y{0};

  bool operator<(const Point2D& other) const {
    return (x != other.x) ? (x < other.x) : (y < other.y);
  }
  bool operator==(const Point2D& other) const {
    return x == other.x && y == other.y;
  }
};

class Greedy : public rclcpp::Node {
public:
  Greedy();

private:
  void declareParameters();
  void loadParameters();
  void initializeSubscribers();
  void initializePublishers();
  void initializeTimer();

  void timerCallback();
  void visibleCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void explorationEnabledCallback(const std_msgs::msg::Bool::SharedPtr msg);

  std::vector<Point2D> findFrontiers() const;
  std::vector<Point2D> clusterFrontiers(
      const std::vector<Point2D>& rawFrontiers, int clusterSize) const;
  void filterUnreachableFrontiers();

  // Returns A* path cost in grid steps, or -1 if unreachable.
  float aStarCost(int startX, int startY, int goalX, int goalY) const;
  void assignNearestFrontier();

  bool isFrontier(int x, int y) const;
  bool findNearestFree(int x, int y, int& freeX, int& freeY) const;
  bool checkArriveTarget() const;
  bool checkTravelFarEnough() const;
  void worldToGrid(
      const geometry_msgs::msg::Pose& pose, int& gridX, int& gridY) const;
  void publishFrontierCloud() const;
  void publishRawFrontierCloud() const;
  void publishPath(int startX, int startY, int goalX, int goalY) const;

  std::string vehicleName_;
  std::string explorationEnabledTopic_;
  double planningCycle_{1.0};
  double replanCycle_{1.0};
  int    clusterSize_{6};
  int    occupiedThreshold_{50};
  double distanceThreshold_{3.0};
  double travelDistanceThreshold_{3.0};

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibleSub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odomSub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr          explorationEnabledSub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   frontierPub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr   rawFrontierPub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr             pathPub_;
  rclcpp::TimerBase::SharedPtr                                  timer_;
  rclcpp::TimerBase::SharedPtr                                  replanTimer_;

  nav_msgs::msg::OccupancyGrid visible_{};
  float resolution_{0.0f};
  float originX_{0.0f};
  float originY_{0.0f};
  int   width_{0};
  int   height_{0};
  bool  visibleReceived_{false};
  mutable std::mutex mapMutex_;

  geometry_msgs::msg::Pose currentPose_{};
  Point2D targetFrontier_{0, 0};
  Point2D startPoint_{0, 0};
  bool    replanFlag_{true};
  bool    explorationEnabled_{false};

  std::vector<Point2D> frontiers_{};
  std::vector<Point2D> rawFrontiers_{};
  std::set<Point2D>    visitedFrontiers_{};
};

#endif  // GREEDY_H
