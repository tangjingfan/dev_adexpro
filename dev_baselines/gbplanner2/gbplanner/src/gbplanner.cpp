#include "gbplanner/gbplanner.h"

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace explorer {

Gbplanner::Gbplanner(rclcpp::Node* node) : node_(node) {
  planner_status_ = Gbplanner::PlannerStatus::NOT_READY;

  rrg_ = new Rrg(node_);
  if (!(rrg_->loadParams(false))) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Could not load all required parameters. Shutdown ROS node.");
    rclcpp::shutdown();
    return;
  }

  initializeAttributes();
}

Gbplanner::Gbplanner(
    rclcpp::Node* node,
    MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>*
        map_manager)
    : node_(node) {
  planner_status_ = Gbplanner::PlannerStatus::NOT_READY;
  rrg_ = new Rrg(node_, map_manager);

  if (!(rrg_->loadParams(true))) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Could not load all required parameters. Shutdown ROS node.");
    rclcpp::shutdown();
    return;
  }

  initializeAttributes();
}

void Gbplanner::initializeAttributes() {
  planner_service_ = node_->create_service<planner_msgs::srv::PlannerSrv>(
      "gbplanner",
      std::bind(&Gbplanner::plannerServiceCallback, this,
                std::placeholders::_1, std::placeholders::_2));

  pose_subscriber_ =
      node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
          "pose", 100,
          std::bind(&Gbplanner::poseCallback, this, std::placeholders::_1));
  pose_stamped_subscriber_ =
      node_->create_subscription<geometry_msgs::msg::PoseStamped>(
          "pose_stamped", 100,
          std::bind(&Gbplanner::poseStampedCallback, this,
                    std::placeholders::_1));
  odometry_subscriber_ = node_->create_subscription<nav_msgs::msg::Odometry>(
      "odometry", 100,
      std::bind(&Gbplanner::odometryCallback, this, std::placeholders::_1));
}

void Gbplanner::setSharedParams(const RobotParams& robot_params,
                                const BoundedSpaceParams& global_space_params) {
  rrg_->setSharedParams(robot_params, global_space_params);
}

void Gbplanner::setSharedParams(const RobotParams& robot_params,
                                const BoundedSpaceParams& global_space_params,
                                const BoundedSpaceParams& local_space_params) {
  rrg_->setSharedParams(robot_params, global_space_params, local_space_params);
}

void Gbplanner::plannerServiceCallback(
    const std::shared_ptr<planner_msgs::srv::PlannerSrv::Request> request,
    std::shared_ptr<planner_msgs::srv::PlannerSrv::Response> response) {
  rrg_->setGlobalFrame(request->header.frame_id);
  rrg_->setBoundMode(static_cast<BoundModeType>(request->bound_mode));
  rrg_->setRootStateForPlanning(request->root_pose);

  response->path.clear();
  if (getPlannerStatus() == Gbplanner::PlannerStatus::NOT_READY) {
    RCLCPP_WARN(node_->get_logger(), "The planner is not ready.");
    return;
  }

  rrg_->reset();
  Rrg::GraphStatus status = rrg_->buildGraph();
  switch (status) {
    case Rrg::GraphStatus::OK:
      break;
    case Rrg::GraphStatus::ERR_KDTREE:
      RCLCPP_WARN(node_->get_logger(),
                  "[PLANNER_ERROR] An issue occurred with kdtree data.");
      break;
    case Rrg::GraphStatus::ERR_NO_FEASIBLE_PATH:
      RCLCPP_WARN(node_->get_logger(),
                  "[PLANNER_ERROR] No feasible path was found.");
      break;
    case Rrg::GraphStatus::NOT_OK:
      RCLCPP_WARN(node_->get_logger(), "[GBPLANNER] Resending global path");
      response->path = rrg_->reRunGlobalPlanner();
      break;
    default:
      RCLCPP_WARN(node_->get_logger(),
                  "[PLANNER_ERROR] Error occurred in building graph.");
      break;
  }

  if (status == Rrg::GraphStatus::OK) {
    status = rrg_->evaluateGraph();
    switch (status) {
      case Rrg::GraphStatus::OK:
        break;
      case Rrg::GraphStatus::NO_GAIN:
        RCLCPP_WARN(node_->get_logger(),
                    "[PLANNER_ERROR] No positive gain was found.");
        break;
      case Rrg::GraphStatus::NOT_OK:
        RCLCPP_WARN(node_->get_logger(),
                    "[GBPLANNER] Very low local gain. Triggering global planner");
        response->path = rrg_->runGlobalPlanner(0, false, false);
        response->status =
            planner_msgs::srv::PlannerSrv::Response::REPOSITIONING;
        break;
      default:
        RCLCPP_WARN(node_->get_logger(),
                    "[PLANNER_ERROR] Error occurred in gain calculation.");
        break;
    }
  }

  if (status == Rrg::GraphStatus::OK) {
    response->path = rrg_->getBestPath(request->header.frame_id,
                                       response->status);
  }
}

void Gbplanner::poseCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose) {
  processPose(pose->pose.pose);
}

void Gbplanner::poseStampedCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr pose) {
  processPose(pose->pose);
}

void Gbplanner::processPose(const geometry_msgs::msg::Pose& pose) {
  StateVec state;
  state[0] = pose.position.x;
  state[1] = pose.position.y;
  state[2] = pose.position.z;
  state[3] = tf2::getYaw(pose.orientation);
  rrg_->setState(state);
}

void Gbplanner::odometryCallback(
    const nav_msgs::msg::Odometry::SharedPtr odo) {
  StateVec state;
  state[0] = odo->pose.pose.position.x;
  state[1] = odo->pose.pose.position.y;
  state[2] = odo->pose.pose.position.z;
  state[3] = tf2::getYaw(odo->pose.pose.orientation);
  rrg_->setState(state);
}

Gbplanner::PlannerStatus Gbplanner::getPlannerStatus() {
  // Should have a list of checking conditions to set the planner as ready.
  // For now, return READY unconditionally (matches the ROS 1 behaviour after
  // the constructor returns).
  return Gbplanner::PlannerStatus::READY;
}

}  // namespace explorer
