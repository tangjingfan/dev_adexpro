#ifndef GBPLANNER_RVIZ_H_
#define GBPLANNER_RVIZ_H_

#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "planner_common/graph.h"
#include "planner_common/graph_base.h"
#include "planner_common/graph_manager.h"
#include "planner_common/map_manager.h"
#include "planner_common/params.h"
#include "planner_common/random_sampler.h"
#include "planner_common/trajectory.h"

namespace explorer {

class Visualization {
 public:
  explicit Visualization(rclcpp::Node* node);

  void visualizeWorkspace(StateVec& state, BoundedSpaceParams& global_ws,
                          BoundedSpaceParams& local_ws);
  void visualizeNoGainZones(std::vector<BoundedSpaceParams>& no_gain_zones);
  void visualizeGraph(const std::shared_ptr<GraphManager> graph_manager);
  void visualizeProjectedGraph(
      const std::shared_ptr<GraphManager> graph_manager);
  void visualizeFailedEdges(std::shared_ptr<SampleStatistic> ss);
  void visualizeRobotState(StateVec& state, RobotParams& robot_params);
  void visualizeSensorFOV(StateVec& state, SensorParams& sensor_params);
  void visualizeShortestPaths(const std::shared_ptr<GraphManager> graph_manager,
                              const ShortestPathsReport& graph_rep);
  void visualizeNegativePaths(const std::vector<int>& ids,
                              const std::shared_ptr<GraphManager> graph_manager,
                              const ShortestPathsReport& graph_rep);
  void visualizeNegativePaths(const std::vector<Eigen::Vector3d>& edge_vertices,
                              const std::shared_ptr<GraphManager> graph_manager,
                              const ShortestPathsReport& graph_rep);

  void visualizeBestPaths(const std::shared_ptr<GraphManager> graph_manager,
                          const ShortestPathsReport& graph_rep, int n,
                          int best_vertex_id);
  void visualizeRefPath(const std::vector<geometry_msgs::msg::Pose>& path);
  void visualizeRefPath(const std::vector<geometry_msgs::msg::Pose>& path,
                        int color);

  void visualizeVolumetricGain(
      Eigen::Vector3d& bound_min, Eigen::Vector3d& bound_max,
      std::vector<std::pair<Eigen::Vector3d, MapManager::VoxelStatus>>& voxels,
      double voxel_size);
  void visualizeSampler(RandomSampler& random_sampler);
  void visualizeRays(const StateVec state,
                     const std::vector<Eigen::Vector3d> ray_endpoints);
  void visualizeGlobalGraph(const std::shared_ptr<GraphManager> graph_manager);
  void visualizeHomingPath(const std::shared_ptr<GraphManager> graph_manager,
                           const ShortestPathsReport& graph_rep,
                           int current_id);

  void visualizeClusteredPaths(
      const std::shared_ptr<GraphManager> graph_manager,
      const ShortestPathsReport& graph_rep,
      const std::vector<Vertex*>& vertices,
      const std::vector<int>& cluster_ids);

  void visualizeRobotStateHistory(const std::vector<StateVec*> state_hist);
  void visualizeGlobalPaths(const std::shared_ptr<GraphManager> graph_manager,
                            std::vector<int>& to_frontier_ids,
                            std::vector<int>& to_home_ids);

  void visualizePCL(const pcl::PointCloud<pcl::PointXYZ>* pcl);

  void visualizePath(const std::shared_ptr<GraphManager> graph_manager,
                     const ShortestPathsReport& graph_rep, int vertex_id);
  void visualizeHyperplanes(Eigen::Vector3d& center,
                            std::vector<Eigen::Vector3d>& hyperplane_list,
                            std::vector<Eigen::Vector3d>& tangent_point_list);
  void visualizeModPath(const std::vector<geometry_msgs::msg::Pose>& path);
  void visualizeBlindModPath(const std::vector<geometry_msgs::msg::Pose>& path);

  void setGlobalFrame(std::string frame_id) { world_frame_id = frame_id; }

 private:
  rclcpp::Node* node_;

  // All publishers are MarkerArray except pcl_pub_ (PointCloud2). We type each
  // explicitly so SharedPtr::publish() works.
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_workspace_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      no_gain_zone_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      robot_state_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_graph_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_projected_graph_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_failed_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      sampler_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      geofence_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      sensor_fov_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      volumetric_gain_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr rays_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      negative_edges_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      shortest_paths_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      best_planning_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      ref_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      ref_path_color_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      mod_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      blind_mod_path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_global_graph_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_homing_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planning_global_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      hyperplanes_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      clustered_paths_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      state_history_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pcl_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_pub_;

  std::string world_frame_id = "world";
  const double ws_lifetime = 0;
  const double graph_lifetime = 0.0;
  const double shortest_paths_lifetime = 0.0;
  const double robot_lifetime = 0;
  const double sampler_lifetime = 0;
  const double ray_lifetime = 0;

  int best_path_id_;
  bool getHeatMapColor(float value, float& red, float& green, float& blue);
};

}  // namespace explorer

#endif
