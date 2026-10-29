// gbplanner_node.cpp
//
// ROS 2 entry point for the slim gbplanner2 port. The autonomy_stack feeds
// this node with:
//   /state_estimation     (nav_msgs/Odometry)        -- robot pose
//   /registered_scan      (sensor_msgs/PointCloud2)  -- registered LiDAR scan
//
// The planner publishes its best frontier path as:
//   /solution_path        (nav_msgs/Path)
//
// waypoint_bridge consumes /solution_path together with /padded_visible_map
// (remapped from /projected_map) and emits the per-step /way_point that the
// autonomy_stack local_planner expects. That keeps replanning slow (~2 s) while
// the published waypoint advances at ~5 Hz, so the robot doesn't stall between
// plan ticks.

#include <memory>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "gbplanner/gbplanner.h"

namespace explorer {

class GbplannerNode : public rclcpp::Node {
 public:
  GbplannerNode() : rclcpp::Node("gbplanner") {
    plan_period_sec_ = declare_parameter<double>("plan_period_sec", 2.0);
    world_frame_id_ =
        declare_parameter<std::string>("world_frame_id", "map");
  }

  void initialize() {
    gb_ = std::make_unique<Gbplanner>(this);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/state_estimation", rclcpp::QoS(10),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          current_pose_ = msg->pose.pose;
          have_pose_ = true;
          gb_->odometryCallback(msg);
        });

    solution_path_pub_ =
        create_publisher<nav_msgs::msg::Path>("/solution_path", 5);

    plan_timer_ = create_wall_timer(
        std::chrono::duration<double>(plan_period_sec_),
        std::bind(&GbplannerNode::planTick, this));

    RCLCPP_INFO(get_logger(),
                "gbplanner2 node ready (plan_period=%.2fs, publishes "
                "/solution_path -> waypoint_bridge)",
                plan_period_sec_);
  }

 private:
  void planTick() {
    if (!have_pose_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Waiting for /state_estimation...");
      return;
    }
    Rrg* rrg = gb_->getRrg();
    if (rrg == nullptr) return;

    rrg->setGlobalFrame(world_frame_id_);
    rrg->setBoundMode(BoundModeType::kExtendedBound);

    // Mirror the original gbplanner planner_control_interface sequencing (a
    // faithful copy of Gbplanner::plannerServiceCallback). The slim port kept
    // the global-planner machinery in rrg but never drove it from this loop, so
    // once the ~10 m local box was explored the robot just wandered in place.
    // Now we sequence local <-> global like the original:
    //   buildGraph    == NOT_OK -> still repositioning to a distant global
    //                              frontier; coast on the already-published
    //                              global path (see note below).
    //   evaluateGraph == OK     -> normal local exploration path.
    //   evaluateGraph == NOT_OK -> local gain exhausted; dispatch to the best
    //                              far frontier through the global graph.
    std::vector<geometry_msgs::msg::Pose> path;
    int path_status = 0;

    rrg->reset();
    auto status = rrg->buildGraph();
    if (status == Rrg::GraphStatus::NOT_OK) {
      // global_exploration_ongoing_ is set (and survives reset()): the robot is
      // travelling toward a distant global frontier but is not yet within range
      // (buildGraph clears the flag once <5 m away and local planning resumes).
      // The original/service path calls reRunGlobalPlanner() here, but that
      // routes to current_global_vertex_id_ != 0 and runGlobalPlanner() then
      // sleeps 3 s every call -- unacceptable in this ~2 s timer loop. Instead
      // we just skip: waypoint_bridge keeps following the global path published
      // at dispatch, so the robot keeps advancing toward the frontier.
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
                           "Repositioning to global frontier; coasting on "
                           "published global path");
      return;
    } else if (status != Rrg::GraphStatus::OK) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "buildGraph status = %d, skipping",
                           static_cast<int>(status));
      return;
    }

    status = rrg->evaluateGraph();
    if (status == Rrg::GraphStatus::OK) {
      // getBestPath is only ever called on OK, so best_vertex_ is fresh -- no
      // stale-pointer deref (the earlier crash guard's concern is satisfied by
      // construction here).
      path = rrg->getBestPath(world_frame_id_, path_status);
    } else if (status == Rrg::GraphStatus::NOT_OK) {
      // Very low local gain: jump to the best distant frontier via the global
      // graph. runGlobalPlanner(0, ...) auto-picks the best frontier, sets
      // global_exploration_ongoing_, and (vertex_id == 0) does NOT sleep. This
      // restores the global exploration the slim port had dropped.
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
                           "Low local gain -> triggering global planner");
      path = rrg->runGlobalPlanner(0, false, false);
    } else {
      // NO_GAIN or an error: nothing to publish this tick.
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "evaluateGraph status = %d, skipping",
                           static_cast<int>(status));
      return;
    }

    if (path.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "Planner returned an empty path.");
      return;
    }

    // waypoint_bridge contract: poses[0] is the robot pose, poses[1..] are
    // forward frontier vertices. getBestPath() already starts with the current
    // robot vertex, so we forward the path as-is.
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = now();
    path_msg.header.frame_id = world_frame_id_;
    path_msg.poses.reserve(path.size());
    for (const auto& pose : path) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path_msg.header;
      ps.pose = pose;
      path_msg.poses.push_back(ps);
    }
    solution_path_pub_->publish(path_msg);
  }

  std::unique_ptr<Gbplanner> gb_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr solution_path_pub_;
  rclcpp::TimerBase::SharedPtr plan_timer_;

  geometry_msgs::msg::Pose current_pose_;
  bool have_pose_ = false;

  double plan_period_sec_;
  std::string world_frame_id_;
};

}  // namespace explorer

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<explorer::GbplannerNode>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
