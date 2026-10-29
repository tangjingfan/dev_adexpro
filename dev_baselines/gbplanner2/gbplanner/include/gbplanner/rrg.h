#ifndef RRG_H_
#define RRG_H_

#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <kdtree/kdtree.h>
#include <pcl/common/distances.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/pcl_macros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "gbplanner/gbplanner_rviz.h"
#include "planner_common/graph.h"
#include "planner_common/graph_base.h"
#include "planner_common/graph_manager.h"
#include "planner_common/map_manager.h"
#include "planner_common/map_manager_voxblox_impl.h"
#include "planner_common/params.h"
#include "planner_common/random_sampler.h"
#include "planner_common/trajectory.h"

#include <planner_msgs/msg/planning_bound.hpp>
#include <planner_msgs/msg/planning_mode.hpp>

// ROS 2 port (slim):
// - adaptive_obb is no longer pulled in.
// - DarpaGateParams, geofence, semantic, save/load_graph are removed.
// - search()/getHomingPath()/setHomingPos()/passing-gate functions are dropped.
// - Full visualization is still wired up.
#define FULL_PLANNER_VIZ 1

namespace explorer {

static constexpr double max_difference_waypoint_to_graph = 15.0;

// Keep tracking state of the robot every T seconds.
class RobotStateHistory {
 public:
  RobotStateHistory();
  void addState(StateVec* s);
  bool getNearestState(const StateVec* state, StateVec** s_res);
  bool getNearestStateInRange(const StateVec* state, double range,
                              StateVec** s_res);
  bool getNearestStates(const StateVec* state, double range,
                        std::vector<StateVec*>* s_res);
  void reset();
  std::vector<StateVec*> state_hist_;

 private:
  kdtree* kd_tree_;
};

class Rrg {
 public:
  enum GraphStatus {
    OK = 0,
    ERR_KDTREE,
    ERR_NO_FEASIBLE_PATH,
    NO_GAIN,
    NOT_OK,
  };

  explicit Rrg(rclcpp::Node* node);
  Rrg(rclcpp::Node* node,
      MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>*
          map_manager);

  void reset();
  void clear();

  GraphStatus buildGraph();
  GraphStatus buildGridGraph(StateVec state, Eigen::Vector3d robot_size,
                             Eigen::Vector3d grid_min, Eigen::Vector3d grid_max,
                             Eigen::Vector3d grid_res, double heading);

  void expandGraph(std::shared_ptr<GraphManager> graph_manager,
                   StateVec& new_state, ExpandGraphReport& rep,
                   bool allow_short_edge = false);
  void expandGraph(std::shared_ptr<GraphManager> graph_manager,
                   Vertex& new_vertex, ExpandGraphReport& rep,
                   bool allow_short_edge = false);

  void expandGraphEdges(std::shared_ptr<GraphManager> graph_manager,
                        Vertex* new_vertex, ExpandGraphReport& rep);

  void expandGraphEdgesBlindly(std::shared_ptr<GraphManager> graph_manager,
                               Vertex* new_vertex, double radius,
                               ExpandGraphReport& rep);

  void computeExplorationGain(bool only_leaf_vertices = false);
  void computeExplorationGain(bool only_leaf_vertices, bool clustering);

  MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>*
  getMapManager() {
    return map_manager_;
  }

  GraphStatus evaluateGraph();

  std::vector<geometry_msgs::msg::Pose> runGlobalPlanner(int vertex_id,
                                                         bool not_check_frontier,
                                                         bool ignore_time);
  std::vector<geometry_msgs::msg::Pose> reRunGlobalPlanner();

  void setState(StateVec& state);
  void setBoundMode(BoundModeType bmode);
  void setRootMode(bool plan_ahead);
  void setGlobalFrame(std::string frame_id);
  bool loadParams(bool);
  void initializeParams();
  void initializeAttributes();
  void resetMissionTimer() {
    if (!landing_engaged_) rostime_start_ = node_->now();
  }

  bool modifyPath(pcl::PointCloud<pcl::PointXYZ>* obstacle_pcl,
                  Eigen::Vector3d& p0, Eigen::Vector3d& p1,
                  Eigen::Vector3d& p1_mod);

  bool improveFreePath(const std::vector<geometry_msgs::msg::Pose>& path_orig,
                       std::vector<geometry_msgs::msg::Pose>& path_mod,
                       bool relaxed);

  std::vector<geometry_msgs::msg::Pose> getBestPath(std::string tgt_frame,
                                                    int& status);

  std::vector<geometry_msgs::msg::Pose> getGlobalPath(
      geometry_msgs::msg::PoseStamped& waypoint);

  void setRootStateForPlanning(const geometry_msgs::msg::Pose& root_pose);

  void setTimeRemaining(double t) { current_battery_time_remaining_ = t; }

  bool isPathCollisionFree(
      const std::vector<geometry_msgs::msg::Pose>& path,
      const Eigen::Vector3d& robot_size);

  bool setGlobalBound(planner_msgs::msg::PlanningBound& bound,
                      bool reset_to_default = false);
  void getGlobalBound(planner_msgs::msg::PlanningBound& bound);

  void setSharedParams(const RobotParams& robot_params,
                       const BoundedSpaceParams& global_space_params);
  void setSharedParams(const RobotParams& robot_params,
                       const BoundedSpaceParams& global_space_params,
                       const BoundedSpaceParams& local_space_params);

  void setPlannerTriggerMode(PlannerTriggerModeType& trig_mode) {
    planner_trigger_mode_ = trig_mode;
  }

 private:
  bool sampleRandomState(StateVec& state);
  bool sampleVertex(Vertex& vertex);
  bool sampleVertex(RandomSampler& random_sampler, StateVec& root_state,
                    Vertex& vertex);
  double projectSample(Eigen::Vector3d& sample,
                       MapManager::VoxelStatus& voxel_status);
  ProjectedEdgeStatus getProjectedEdgeStatus(
      const Eigen::Vector3d& start, const Eigen::Vector3d& end,
      const Eigen::Vector3d& box_size, bool stop_at_unknown_voxel,
      std::vector<Eigen::Vector3d>& projected_edge, bool);
  void correctYaw();
  void computeVolumetricGain(StateVec& state, VolumetricGain& vgain,
                             bool vis_en = false);
  void computeVolumetricGainRayModel(StateVec& state, VolumetricGain& vgain,
                                     bool vis_en = false,
                                     bool iterative = false);
  void computeVolumetricGainRayModelNoBound(StateVec& state,
                                            VolumetricGain& vgain);

  void evaluateShortestPaths();

  void addFrontiers(int best_vertex_id);

  std::string world_frame_ = "world";

  rclcpp::Node* node_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr free_cloud_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr time_log_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pci_reset_pub_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_srv_subscriber_;

  void stopMsgCallback(const std_msgs::msg::Bool::SharedPtr msg);

  // Graphs.
  std::shared_ptr<GraphManager> local_graph_;
  std::shared_ptr<GraphManager> projected_graph_;
  ShortestPathsReport local_graph_rep_;
  std::shared_ptr<GraphManager> global_graph_;
  ShortestPathsReport global_graph_rep_;
  std::vector<std::vector<double>> edge_inclinations_;

  bool addRefPathToGraph(const std::shared_ptr<GraphManager> graph_manager,
                         const std::vector<Vertex*>& vertices);
  bool addRefPathToGraph(const std::shared_ptr<GraphManager> graph_manager,
                         const std::vector<geometry_msgs::msg::Pose>& path);
  bool connectStateToGraph(std::shared_ptr<GraphManager> graph,
                           StateVec& cur_state, Vertex*& v_added,
                           double dist_ignore_collision_check);
  double getTimeElapsed();
  double getTimeRemained();
  bool isRemainingTimeSufficient(const double& time_cost, double& time_spare);

  PlannerTriggerModeType planner_trigger_mode_;

  double exploring_direction_;
  const double kTimerPeriod = 0.25;
  rclcpp::TimerBase::SharedPtr periodic_timer_;
  void timerCallback();

  std::queue<StateVec> robot_state_queue_;

  const double kFreePointCloudUpdatePeriod = 0.5;
  rclcpp::TimerBase::SharedPtr free_pointcloud_update_timer_;
  void freePointCloudtimerCallback();

  const double kGlobalGraphUpdateTimerPeriod = 0.5;
  const double kGlobalGraphFrontierAdditionTimerPeriod = 1.0;
  const double kGlobalGraphUpdateTimeBudget = 0.1;
  rclcpp::TimerBase::SharedPtr global_graph_update_timer_;
  void expandGlobalGraphTimerCallback();
  rclcpp::TimerBase::SharedPtr global_graph_frontier_addition_timer_;
  void expandGlobalGraphFrontierAdditionTimerCallback();

  const int backtracking_queue_max_size = 500;
  std::queue<StateVec> robot_backtracking_queue_;
  Vertex* robot_backtracking_prev_;

  bool compareAngles(double dir_angle_a, double dir_angle_b, double thres);
  bool comparePathWithDirectionApprioximately(
      const std::vector<geometry_msgs::msg::Pose>& path, double yaw);

  bool reconnectPathBlindly(std::vector<geometry_msgs::msg::Pose>& ref_path,
                            std::vector<geometry_msgs::msg::Pose>& mod_path);

  std::vector<int> performShortestPathsClustering(
      const std::shared_ptr<GraphManager> graph_manager,
      const ShortestPathsReport& graph_rep, std::vector<Vertex*>& vertices,
      double dist_threshold = 1.0, double principle_path_min_length = 1.0,
      bool refinement_enable = true);

  void printShortestPath(int id);

  inline void truncateYaw(double& x) {
    if (x > M_PI)
      x -= 2 * M_PI;
    else if (x < -M_PI)
      x += 2 * M_PI;
  }

  inline void offsetZAxis(StateVec& state, bool down = false) {
    if (robot_params_.type == RobotType::kGroundRobot) {
      if (down)
        state[2] -= random_sampler_.getZOffset();
      else
        state[2] += random_sampler_.getZOffset();
    }
  }

  void convertStateToPoseMsg(const StateVec& state,
                             geometry_msgs::msg::Pose& pose) {
    pose.position.x = state[0];
    pose.position.y = state[1];
    pose.position.z = state[2];
    double yawhalf = state[3] * 0.5;
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = sin(yawhalf);
    pose.orientation.w = cos(yawhalf);
  }

  void convertPoseMsgToState(const geometry_msgs::msg::Pose& pose,
                             StateVec& state);

  void convertPointToEigen(const geometry_msgs::msg::Point& point,
                           Eigen::Vector3d& vec) {
    vec(0) = point.x;
    vec(1) = point.y;
    vec(2) = point.z;
  }

  void publishTimings(std::shared_ptr<SampleStatistic> stat) {
    std_msgs::msg::Float32MultiArray time_log;
    time_log.data.push_back(stat->build_graph_time);
    time_log.data.push_back(stat->compute_exp_gain_time);
    time_log.data.push_back(stat->shortest_path_time);
    time_log.data.push_back(stat->evaluate_graph_time);
    time_log_pub_->publish(time_log);
  }

  // Visualization.
  Visualization* visualization_;

  // Planner params.
  SensorParams sensor_params_;
  SensorParams free_frustum_params_;
  RobotParams robot_params_;
  BoundedSpaceParams local_space_params_;
  BoundedSpaceParams global_space_params_;
  std::vector<BoundedSpaceParams> no_gain_zones_;
  bool use_no_gain_space_ = true;
  PlanningParams planning_params_;
  RandomSampler random_sampler_;
  RandomSampler random_sampler_to_search_;
  BoundedSpaceParams local_search_params_;
  RandomSampler random_sampler_adaptive_;
  BoundedSpaceParams local_adaptive_params_;
  Eigen::Vector3d adaptive_orig_min_val_, adaptive_orig_max_val_;
  RobotDynamicsParams robot_dynamics_params_;
  BoundingBoxType global_bound_;

  MapManagerVoxblox<MapManagerVoxbloxServer, MapManagerVoxbloxVoxel>*
      map_manager_;

  // Mission time tracking.
  rclcpp::Time rostime_start_;
  double current_battery_time_remaining_;

  Vertex* root_vertex_;
  Vertex* best_vertex_;

  // Current state of the robot, updated from odometry.
  StateVec current_state_;
  StateVec state_for_planning_;

  Eigen::Vector3d robot_box_size_;
  int planning_num_vertices_max_;
  int planning_num_edges_max_;

  std::shared_ptr<SampleStatistic> stat_, stat_chrono_;

  int planner_trigger_count_;
  TIMER ttime;

  bool odometry_ready;

  int num_low_gain_iters_;
  bool auto_global_planner_trig_;

  bool global_exploration_ongoing_;
  int current_global_vertex_id_;
  bool local_exploration_ongoing_;

  bool homing_engaged_ = false;
  bool landing_engaged_ = false;

  bool add_frontiers_to_global_graph_;

  StateVec last_state_marker_;
  StateVec last_state_marker_global_;

  std::shared_ptr<RobotStateHistory> robot_state_hist_;

  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> obs_pcl_;
  std::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> feasible_corridor_pcl_;
};

}  // namespace explorer

#endif
