#ifndef GBPLANNER_H_
#define GBPLANNER_H_

#include <memory>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include "gbplanner/rrg.h"
#include "planner_common/graph.h"
#include "planner_common/graph_base.h"
#include "planner_common/graph_manager.h"
#include "planner_common/params.h"

#include <planner_msgs/srv/planner_srv.hpp>

namespace explorer {

// ROS 2 port (slim):
// - Dropped homing / set-homing-pos / search / geofence / passing-gate /
//   set-global-bound / set-dynamic-global-bound / clear-untraversable-zones /
//   load-graph / save-graph / go-to-waypoint / set-planning-trigger-mode
//   service callbacks. Only the canonical "plan now" service is kept, mostly
//   for debugging; the autonomy_stack does not call it.
// - Dropped untraversable-polygon subscriber and robot-status subscriber.
// - Map save service client removed.
class Gbplanner {
 public:
  enum PlannerStatus { NOT_READY = 0, READY };

  explicit Gbplanner(rclcpp::Node* node);
  Gbplanner(rclcpp::Node* node,
            MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>*
                map_manager);

  void initializeAttributes();

  // Single retained service: trigger a one-shot plan. The autonomy_stack does
  // not need this; it's exposed only for manual debugging.
  void plannerServiceCallback(
      const std::shared_ptr<planner_msgs::srv::PlannerSrv::Request> request,
      std::shared_ptr<planner_msgs::srv::PlannerSrv::Response> response);

  void setSharedParams(const RobotParams& robot_params,
                       const BoundedSpaceParams& global_space_params);
  void setSharedParams(const RobotParams& robot_params,
                       const BoundedSpaceParams& global_space_params,
                       const BoundedSpaceParams& local_space_params);

  // Direct accessors so a thin "node" entry can drive the planner from
  // topic callbacks (the typical pattern in this slimmed port).
  void processPose(const geometry_msgs::msg::Pose& pose);
  void poseCallback(
      const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose);
  void poseStampedCallback(
      const geometry_msgs::msg::PoseStamped::SharedPtr pose);
  void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr odo);

  PlannerStatus getPlannerStatus();
  Rrg* getRrg() { return rrg_; }

  Rrg* rrg_;

 private:
  rclcpp::Node* node_;
  rclcpp::Service<planner_msgs::srv::PlannerSrv>::SharedPtr planner_service_;

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      pose_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      pose_stamped_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_subscriber_;

  PlannerStatus planner_status_;
};

}  // namespace explorer
#endif
