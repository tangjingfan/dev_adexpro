#ifndef GRAPH_MANAGER_H_
#define GRAPH_MANAGER_H_

#include <fstream>
#include <iostream>
#include <unordered_map>

#include <eigen3/Eigen/Dense>
#include <kdtree/kdtree.h>

#include <planner_msgs/msg/edge.hpp>
#include <planner_msgs/msg/graph.hpp>
#include <planner_msgs/msg/vertex.hpp>

#include "planner_common/graph.h"
#include "planner_common/graph_base.h"
#include "planner_common/params.h"

class GraphManager {
 public:
  GraphManager();

  void reset();

  int generateSubgraphIndex();
  int generateVertexID();

  void addVertex(Vertex* v);
  void addEdge(Vertex* v, Vertex* u, double weight);
  void removeEdge(Vertex* v, Vertex* u);

  int getNumVertices() { return graph_->getNumVertices(); }
  int getNumEdges() { return graph_->getNumEdges(); }

  Vertex* getVertex(int id) { return vertices_map_[id]; }
  void getLeafVertices(std::vector<Vertex*>& leaf_vertices);
  void findLeafVertices(const ShortestPathsReport& rep);

  bool findShortestPaths(ShortestPathsReport& rep);
  bool findShortestPaths(int source_id, ShortestPathsReport& rep);

  void getShortestPath(int target_id, const ShortestPathsReport& rep,
                       bool source_to_target_order, std::vector<int>& path);
  void getShortestPath(int target_id, const ShortestPathsReport& rep,
                       bool source_to_target_order, std::vector<Vertex*>& path);
  void getShortestPath(int target_id, const ShortestPathsReport& rep,
                       bool source_to_target_order,
                       std::vector<Eigen::Vector3d>& path);
  void getShortestPath(int target_id, const ShortestPathsReport& rep,
                       bool source_to_target_order,
                       std::vector<StateVec>& path);
  double getShortestDistance(int target_id, const ShortestPathsReport& rep);
  int getParentIDFromShortestPath(int target_id,
                                  const ShortestPathsReport& rep);

  bool getNearestVertex(const StateVec* state, Vertex** v_res);
  bool getNearestVertexInRange(const StateVec* state, double range,
                               Vertex** v_res);
  bool getNearestVertices(const StateVec* state, double range,
                          std::vector<Vertex*>* v_res);
  bool existVertexInRange(const StateVec* state, double range);

  void updateVertexTypeInRange(StateVec& state, double range);

  void convertGraphToMsg(planner_msgs::msg::Graph& graph_msg);
  void convertMsgToGraph(const planner_msgs::msg::Graph& graph_msg);

  // ROS 2 port: save/load via ROS 1 serialization removed. These overloads
  // are stubs that log a warning. Persist your graph differently if needed.
  void saveGraph(const std::string& path);
  void loadGraph(const std::string& path);

  std::shared_ptr<Graph> graph_;
  std::unordered_map<int, Vertex*> vertices_map_;
  std::map<int, std::vector<std::pair<int, double>>> edge_map_;

 private:
  kdtree* kd_tree_;
  int subgraph_ind_;
  int id_count_;

  std::unordered_map<int, std::pair<int, int>> local_id_map_;
};

#endif
