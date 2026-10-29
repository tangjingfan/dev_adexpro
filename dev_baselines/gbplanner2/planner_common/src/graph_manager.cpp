#include "planner_common/graph_manager.h"

#include <algorithm>
#include <iostream>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

GraphManager::GraphManager() {
  kd_tree_ = nullptr;
  reset();
}

void GraphManager::reset() {
  if (kd_tree_) kd_free(kd_tree_);
  kd_tree_ = kd_create(3);

  graph_.reset(new Graph());

  if (!vertices_map_.empty()) {
    for (auto& kv : vertices_map_) {
      delete kv.second;
    }
  }
  vertices_map_.clear();
  edge_map_.clear();

  subgraph_ind_ = -1;
  id_count_ = -1;
}

void GraphManager::addVertex(Vertex* v) {
  kd_insert3(kd_tree_, v->state.x(), v->state.y(), v->state.z(), v);
  if (v->id == 0)
    graph_->addSourceVertex(0);
  else
    graph_->addVertex(v->id);
  vertices_map_[v->id] = v;
}

void GraphManager::addEdge(Vertex* v, Vertex* u, double weight) {
  graph_->addEdge(v->id, u->id, weight);
  edge_map_[v->id].push_back(std::make_pair(u->id, weight));
  edge_map_[u->id].push_back(std::make_pair(v->id, weight));
}

void GraphManager::removeEdge(Vertex* v, Vertex* u) {
  graph_->removeEdge(v->id, u->id);
}

bool GraphManager::getNearestVertex(const StateVec* state, Vertex** v_res) {
  if (getNumVertices() <= 0) return false;
  kdres* nearest = kd_nearest3(kd_tree_, state->x(), state->y(), state->z());
  if (kd_res_size(nearest) <= 0) {
    kd_res_free(nearest);
    return false;
  }
  *v_res = (Vertex*)kd_res_item_data(nearest);
  kd_res_free(nearest);
  return true;
}

bool GraphManager::getNearestVertexInRange(const StateVec* state, double range,
                                           Vertex** v_res) {
  if (getNumVertices() <= 0) return false;
  kdres* nearest = kd_nearest3(kd_tree_, state->x(), state->y(), state->z());
  if (kd_res_size(nearest) <= 0) {
    kd_res_free(nearest);
    return false;
  }
  *v_res = (Vertex*)kd_res_item_data(nearest);
  Eigen::Vector3d dist;
  dist << state->x() - (*v_res)->state.x(), state->y() - (*v_res)->state.y(),
      state->z() - (*v_res)->state.z();
  kd_res_free(nearest);
  if (dist.norm() > range) return false;
  return true;
}

bool GraphManager::getNearestVertices(const StateVec* state, double range,
                                      std::vector<Vertex*>* v_res) {
  kdres* neighbors =
      kd_nearest_range3(kd_tree_, state->x(), state->y(), state->z(), range);
  int neighbors_size = kd_res_size(neighbors);
  if (neighbors_size <= 0) return false;
  v_res->clear();
  for (int i = 0; i < neighbors_size; ++i) {
    Vertex* new_neighbor = (Vertex*)kd_res_item_data(neighbors);
    v_res->push_back(new_neighbor);
    if (kd_res_next(neighbors) <= 0) break;
  }
  kd_res_free(neighbors);
  return true;
}

bool GraphManager::findShortestPaths(ShortestPathsReport& rep) {
  return graph_->findDijkstraShortestPaths(0, rep);
}

bool GraphManager::findShortestPaths(int source_id, ShortestPathsReport& rep) {
  return graph_->findDijkstraShortestPaths(source_id, rep);
}

void GraphManager::getShortestPath(int target_id,
                                   const ShortestPathsReport& rep,
                                   bool source_to_target_order,
                                   std::vector<Vertex*>& path) {
  std::vector<int> path_id;
  getShortestPath(target_id, rep, source_to_target_order, path_id);
  for (auto p = path_id.begin(); p != path_id.end(); ++p) {
    path.push_back(vertices_map_[*p]);
  }
}

void GraphManager::getShortestPath(int target_id,
                                   const ShortestPathsReport& rep,
                                   bool source_to_target_order,
                                   std::vector<Eigen::Vector3d>& path) {
  std::vector<int> path_id;
  getShortestPath(target_id, rep, source_to_target_order, path_id);
  for (auto p = path_id.begin(); p != path_id.end(); ++p) {
    path.emplace_back(Eigen::Vector3d(vertices_map_[*p]->state.x(),
                                      vertices_map_[*p]->state.y(),
                                      vertices_map_[*p]->state.z()));
  }
}

void GraphManager::getShortestPath(int target_id,
                                   const ShortestPathsReport& rep,
                                   bool source_to_target_order,
                                   std::vector<StateVec>& path) {
  std::vector<int> path_id;
  getShortestPath(target_id, rep, source_to_target_order, path_id);
  for (auto p = path_id.begin(); p != path_id.end(); ++p) {
    path.emplace_back(vertices_map_[*p]->state);
  }
}

void GraphManager::getShortestPath(int target_id,
                                   const ShortestPathsReport& rep,
                                   bool source_to_target_order,
                                   std::vector<int>& path) {
  path.clear();
  if (!rep.status) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Shortest paths report is not valid" << std::endl;
    }
    return;
  }

  if (static_cast<int>(rep.parent_id_map.size()) <= target_id) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Vertex with ID [" << target_id << "] doesn't exist in the graph"
                << std::endl;
    }
    return;
  }

  if (target_id == rep.source_id) {
    path.push_back(target_id);
    return;
  }

  int parent_id = rep.parent_id_map.at(target_id);
  if (parent_id == target_id) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Vertex with ID [" << target_id << "] is isolated from the graph"
                << std::endl;
    }
    return;
  }

  path.push_back(target_id);
  path.push_back(parent_id);
  while (parent_id != rep.source_id) {
    parent_id = rep.parent_id_map.at(parent_id);
    path.push_back(parent_id);
  }

  if (source_to_target_order) {
    std::reverse(path.begin(), path.end());
  }
}

double GraphManager::getShortestDistance(int target_id,
                                         const ShortestPathsReport& rep) {
  double dist = std::numeric_limits<double>::max();

  if (!rep.status) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Shortest paths report is not valid" << std::endl;
    }
    return dist;
  }
  if (static_cast<int>(rep.parent_id_map.size()) <= target_id) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Vertex with ID [" << target_id << "] doesn't exist in the graph"
                << std::endl;
    }
    return dist;
  }
  dist = rep.distance_map.at(target_id);
  return dist;
}

int GraphManager::getParentIDFromShortestPath(int target_id,
                                              const ShortestPathsReport& rep) {
  if (!rep.status) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Shortest paths report is not valid" << std::endl;
    }
    return target_id;
  }

  if (static_cast<int>(rep.parent_id_map.size()) <= target_id) {
    if (global_verbosity >= Verbosity::WARN) {
      std::cout << "Vertex with ID [" << target_id << "] doesn't exist in the graph"
                << std::endl;
    }
    return target_id;
  }

  return rep.parent_id_map.at(target_id);
}

void GraphManager::getLeafVertices(std::vector<Vertex*>& leaf_vertices) {
  leaf_vertices.clear();
  for (int id = 0; id < getNumVertices(); ++id) {
    Vertex* v = getVertex(id);
    if (v && v->is_leaf_vertex) leaf_vertices.push_back(v);
  }
}

void GraphManager::findLeafVertices(const ShortestPathsReport& rep) {
  int num_vertices = getNumVertices();
  for (int id = 0; id < num_vertices; ++id) {
    int pid = getParentIDFromShortestPath(id, rep);
    Vertex* parent_v = getVertex(pid);
    if (parent_v) parent_v->is_leaf_vertex = false;
  }
}

int GraphManager::generateSubgraphIndex() { return ++subgraph_ind_; }

int GraphManager::generateVertexID() { return ++id_count_; }

void GraphManager::updateVertexTypeInRange(StateVec& state, double range) {
  std::vector<Vertex*> nearest_vertices;
  getNearestVertices(&state, range, &nearest_vertices);
  for (auto& v : nearest_vertices) {
    v->type = VertexType::kVisited;
  }
}

void GraphManager::convertGraphToMsg(planner_msgs::msg::Graph& graph_msg) {
  for (auto& v : vertices_map_) {
    planner_msgs::msg::Vertex vertex;
    vertex.id = v.second->id;

    vertex.pose.position.x = v.second->state[0];
    vertex.pose.position.y = v.second->state[1];
    vertex.pose.position.z = v.second->state[2];
    double yawhalf = v.second->state[3] * 0.5;
    vertex.pose.orientation.x = 0.0;
    vertex.pose.orientation.y = 0.0;
    vertex.pose.orientation.z = sin(yawhalf);
    vertex.pose.orientation.w = cos(yawhalf);

    vertex.num_unknown_voxels = v.second->vol_gain.num_unknown_voxels;
    vertex.num_occupied_voxels = v.second->vol_gain.num_occupied_voxels;
    vertex.num_free_voxels = v.second->vol_gain.num_free_voxels;
    vertex.is_frontier = v.second->vol_gain.is_frontier;
    graph_msg.vertices.emplace_back(vertex);
  }

  std::pair<Graph::GraphType::edge_iterator, Graph::GraphType::edge_iterator> ei;
  graph_->getEdgeIterator(ei);
  for (Graph::GraphType::edge_iterator it = ei.first; it != ei.second; ++it) {
    planner_msgs::msg::Edge edge;
    std::tie(edge.source_id, edge.target_id, edge.weight) =
        graph_->getEdgeProperty(it);
    graph_msg.edges.emplace_back(edge);
  }
}

void GraphManager::convertMsgToGraph(const planner_msgs::msg::Graph& graph_msg) {
  for (auto& v : graph_msg.vertices) {
    generateVertexID();
    StateVec state;
    state[0] = v.pose.position.x;
    state[1] = v.pose.position.y;
    state[2] = v.pose.position.z;
    state[3] = tf2::getYaw(v.pose.orientation);
    Vertex* vertex = new Vertex(v.id, state);
    vertex->vol_gain.num_unknown_voxels = v.num_unknown_voxels;
    vertex->vol_gain.num_occupied_voxels = v.num_occupied_voxels;
    vertex->vol_gain.num_free_voxels = v.num_free_voxels;
    vertex->vol_gain.is_frontier = v.is_frontier;
    if (v.is_frontier) vertex->type = VertexType::kFrontier;
    addVertex(vertex);
  }

  for (auto& e : graph_msg.edges) {
    addEdge(getVertex(e.source_id), getVertex(e.target_id), e.weight);
  }
}

void GraphManager::saveGraph(const std::string& /*path*/) {
  // ROS 2 port: graph save/load via ROS 1 message serialization is not
  // available. If you need disk persistence, serialise the planner_msgs::Graph
  // separately (e.g. as a YAML / protobuf / rclcpp serialization stream).
  if (global_verbosity >= Verbosity::WARN) {
    std::cout << "[GraphManager] saveGraph() is a no-op in this ROS 2 port."
              << std::endl;
  }
}

void GraphManager::loadGraph(const std::string& /*path*/) {
  if (global_verbosity >= Verbosity::WARN) {
    std::cout << "[GraphManager] loadGraph() is a no-op in this ROS 2 port."
              << std::endl;
  }
}
