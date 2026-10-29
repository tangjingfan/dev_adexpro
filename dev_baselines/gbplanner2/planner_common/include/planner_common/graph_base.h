#pragma once

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>

#include "planner_common/params.h"

// ROS 2 port note (slim):
// - Dropped planner_semantic_msgs/SemanticClass dependency (Vertex.semantic_class removed).
// - Dropped ros::serialization Serializer<> template specializations: graph
//   save/load via ROS message serialization is no longer supported.
// - printTime now writes to std::cout instead of ROS_INFO_COND.

struct SampleStatistic {
  SampleStatistic()
      : num_vertices_fail(0),
        num_edges_fail(0),
        build_graph_time(0),
        compute_exp_gain_time(0),
        shortest_path_time(0),
        evaluate_graph_time(0),
        total_time(0) {}
  StateVec current_state;
  int num_vertices_fail;
  int num_edges_fail;
  std::vector<std::vector<double>> edges_fail;
  double build_graph_time;
  double compute_exp_gain_time;
  double shortest_path_time;
  double evaluate_graph_time;
  double total_time;
  void init(StateVec& state) { current_state = state; }

  void printTime() {
    total_time = build_graph_time + compute_exp_gain_time + shortest_path_time +
                 evaluate_graph_time;
    if (global_verbosity >= Verbosity::INFO) {
      std::printf(
          "Time statistics:\n  Build graph    : %3.3f (s)\n  "
          "Compute gain   : %3.3f (s)\n  Dijkstra       : %3.3f (s)\n  "
          "Evaluate graph : %3.3f (s)\n  Total          : %3.3f (s)\n",
          build_graph_time, compute_exp_gain_time, shortest_path_time,
          evaluate_graph_time, total_time);
    }
  }
  void printTime(std::string title) {
    total_time = build_graph_time + compute_exp_gain_time + shortest_path_time +
                 evaluate_graph_time;
    if (global_verbosity >= Verbosity::INFO) {
      std::printf(
          "Time statistics (%s):\n  Build graph    : %3.3f (ms)\n  "
          "Compute gain   : %3.3f (ms)\n  Dijkstra       : %3.3f (ms)\n  "
          "Evaluate graph : %3.3f (ms)\n  Total          : %3.3f (ms)\n",
          title.c_str(), build_graph_time, compute_exp_gain_time,
          shortest_path_time, evaluate_graph_time, total_time);
    }
  }
};

struct VolumetricGain {
  VolumetricGain()
      : gain(0),
        accumulative_gain(0),
        num_unknown_voxels(0),
        num_free_voxels(0),
        num_occupied_voxels(0),
        num_unknown_surf_voxels(0),
        is_frontier(false) {}

  void reset() {
    gain = 0;
    accumulative_gain = 0;
    num_unknown_voxels = 0;
    num_free_voxels = 0;
    num_occupied_voxels = 0;
    num_unknown_surf_voxels = 0;
    is_frontier = false;
    unseen_voxel_hash_keys.clear();
  }

  double gain;
  double accumulative_gain;
  int num_unknown_voxels;
  int num_free_voxels;
  int num_occupied_voxels;
  int num_unknown_surf_voxels;
  std::vector<std::size_t> unseen_voxel_hash_keys;

  bool is_frontier;

  void printGain() {
    std::cout << "Gains: " << gain << ", " << num_unknown_voxels << ", "
              << num_occupied_voxels << ", " << num_free_voxels << std::endl;
  }
};

struct BoundingBoxType {
  void reset() {
    min_val = min_val_default;
    max_val = max_val_default;
  }

  void setDefault(Eigen::Vector3d& v_min, Eigen::Vector3d& v_max) {
    min_val_default = v_min;
    max_val_default = v_max;

    min_val = v_min;
    max_val = v_max;
  }

  void set(Eigen::Vector3d& v_min, Eigen::Vector3d& v_max) {
    min_val = v_min;
    max_val = v_max;
  }

  void set(geometry_msgs::msg::Point& p_min, geometry_msgs::msg::Point& p_max,
           bool set_z_val = false) {
    min_val.x() = p_min.x;
    min_val.y() = p_min.y;
    if (set_z_val) min_val.z() = p_min.z;

    max_val.x() = p_max.x;
    max_val.y() = p_max.y;
    if (set_z_val) max_val.z() = p_max.z;
  }

  void get(geometry_msgs::msg::Point& p_min,
           geometry_msgs::msg::Point& p_max) {
    p_min.x = min_val.x();
    p_min.y = min_val.y();
    p_min.z = min_val.z();
    p_max.x = max_val.x();
    p_max.y = max_val.y();
    p_max.z = max_val.z();
  }

  void get(Eigen::Vector3d& v_min, Eigen::Vector3d& v_max) {
    v_min = min_val;
    v_max = max_val;
  }

 private:
  Eigen::Vector3d min_val;
  Eigen::Vector3d max_val;

  Eigen::Vector3d min_val_default;
  Eigen::Vector3d max_val_default;
};

// Mark the property of the vertex.
enum struct VertexType {
  kUnvisited = 0,
  kVisited = 1,
  kFrontier = 2
};

struct Vertex {
  Vertex(int v_id, StateVec v_state) {
    id = v_id;
    state << v_state[0], v_state[1], v_state[2], v_state[3];
    vol_gain.reset();
    parent = nullptr;
    distance = 0;
    is_leaf_vertex = true;
    is_hanging = false;
    type = VertexType::kUnvisited;
    cluster_id = 0;
    dm = 0;
  }

  int id;
  StateVec state;
  VolumetricGain vol_gain;
  Vertex* parent;
  std::vector<Vertex*> children;
  double distance;
  bool is_leaf_vertex;
  bool is_hanging;
  VertexType type;
  int cluster_id;
  double dm;
};

struct SerializeVertex {
  SerializeVertex() {}

  int id;
  StateVec state;
  VolumetricGain vol_gain;
  int parent_id;
  // kFrontier=0, kUnvisited=1, kVisited=2
  int type;

  void printit() {
    std::cout << "x: " << this->state[0] << " y: " << this->state[1]
              << " z: " << this->state[2] << std::endl;
    std::cout << "ID: " << id << " Paren: " << parent_id << std::endl;
  }
};

enum class ExpandGraphStatus {
  kSuccess = 0,
  kErrorKdTree,
  kErrorCollisionEdge,
  kErrorShortEdge,
  kErrorGeofenceViolated,
  kNull
};

struct ExpandGraphReport {
  ExpandGraphReport()
      : status(ExpandGraphStatus::kNull),
        num_vertices_added(0),
        num_edges_added(0),
        vertex_added(nullptr) {}
  ExpandGraphStatus status;
  int num_vertices_added;
  int num_edges_added;
  Vertex* vertex_added;
};

enum class ConnectStatus {
  kSuccess = 0,
  kErrorCollisionAtSource,
  kErrorNoFeasiblePath
};

struct RandomSamplingParams {
  RandomSamplingParams() {
    num_vertices_max = 500;
    num_edges_max = 10000;
    num_loops_cutoff = 2000;
    num_loops_max = 100000;
    reached_target_radius = 2.0;
    check_collision_at_source = true;
    num_paths_to_target_max = 5;
  }
  int num_vertices_max;
  int num_edges_max;
  double reached_target_radius;
  bool check_collision_at_source;
  int num_paths_to_target_max;
  double num_loops_cutoff;
  double num_loops_max;
};
