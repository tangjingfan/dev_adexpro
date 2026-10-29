#ifndef NBVPLANNER_NBVPLANNER_HPP
#define NBVPLANNER_NBVPLANNER_HPP

#include "nbvplanner/rrt_tree.hpp"

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <Eigen/Dense>
#include <mutex>
#include <string>

class NBVPlanner : public rclcpp::Node {
public:
  NBVPlanner();

private:
  void declareParameters();
  void loadParameters();
  void initializeSubscribers();
  void initializePublishers();
  void initializeTimers();

  void timerCallback();
  void goalCheckCallback();
  void visibleCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void explorationEnabledCallback(const std_msgs::msg::Bool::SharedPtr msg);

  // Build the RRT tree and publish the best waypoint.
  void runPlanning();

  void publishWaypoint(const nbvplanner::RrtTree::State& state);
  // Publish full RRT tree (grey-blue) + chosen path (orange) as a single MarkerArray.
  void publishVisualization(const std::vector<nbvplanner::RrtTree::State>& bestPath) const;

  // ----- Parameters (ROS) -----
  std::string vehicleName_;
  std::string explorationEnabledTopic_;
  std::string waypointTopic_{"/way_point"};
  double      planningCycle_{2.0};
  double      goalReachThreshold_{1.0};

  // Forwarded to RrtTree::Params
  nbvplanner::Params rrtParams_;

  // ----- Runtime state -----
  bool explorationEnabled_{false};
  bool replanFlag_{false};

  nav_msgs::msg::OccupancyGrid visible_;
  bool visibleReceived_{false};

  Eigen::Vector3d currentState_{0.0, 0.0, 0.0};  // (x, y, yaw)
  bool            odomReceived_{false};

  Eigen::Vector2d currentGoal_{0.0, 0.0};
  bool            hasGoal_{false};

  nbvplanner::RrtTree tree_;

  mutable std::mutex mapMutex_;

  // ----- ROS interfaces -----
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibleSub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odomSub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr          explorationEnabledSub_;

  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr       waypointPub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr   treePub_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr goalCheckTimer_;
};

#endif  // NBVPLANNER_NBVPLANNER_HPP
