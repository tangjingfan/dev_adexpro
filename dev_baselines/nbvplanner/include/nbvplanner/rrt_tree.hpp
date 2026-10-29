#ifndef NBVPLANNER_RRT_TREE_HPP
#define NBVPLANNER_RRT_TREE_HPP

#include "nav_msgs/msg/occupancy_grid.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace nbvplanner {

// A single node in the RRT tree.
// State is (x, y, yaw) in world frame (metres / radians).
struct Node {
  Eigen::Vector3d    state{0.0, 0.0, 0.0};
  Node*              parent{nullptr};
  std::vector<Node*> children;
  double             distance{0.0};  // cumulative arc-length from root
  double             gain{0.0};      // accumulated discounted information gain
};

// Parameters that drive the RRT and gain calculation.
struct Params {
  // Sampling bounding box [m] — set dynamically from the visible map at each
  // planning cycle (NBVPlanner::runPlanning). Default values are overwritten.
  double minX{0.0}, maxX{0.0};
  double minY{0.0}, maxY{0.0};

  // When true, the robot may sample slightly outside the bbox
  bool softBounds{false};

  // Robot footprint radius used for collision inflation [m]
  double vehicleRadius{0.5};

  // Maximum RRT edge length per iteration [m]
  double extensionRange{1.5};

  // Radius of the information-gain integration disc [m]
  double gainRange{3.0};

  // Per-cell gain weights
  double igUnmapped{1.0};  // weight for unknown cells
  double igFree{0.0};      // weight for free cells (usually 0)

  // Exponential distance discount: gain *= exp(-degressiveCoeff * distance)
  double degressiveCoeff{0.25};

  // Minimum accumulated gain that counts as "a gain was found"
  double zeroGain{0.05};

  // Planning iterations
  int initIterations{15};    // minimum extra iterations after first gain found
  int cutoffIterations{500}; // hard limit before giving up

  // Occupied-cell threshold in the OccupancyGrid [0-100]
  int occupiedThreshold{50};

  // If false, unknown cells (-1) are treated as non-traversable for tree edges.
  // This keeps the RRT inside known free space.
  bool allowUnknownTraversal{false};

};

// RRT tree that operates on a 2D nav_msgs::OccupancyGrid.
// Adapted from ETH ASL nbvplanner (Bircher et al., ICRA 2016) to:
//   - work in 2D (x, y, yaw) instead of 3D
//   - use OccupancyGrid instead of OctoMap
//   - remove multi-agent collision avoidance
class RrtTree {
public:
  using State = Eigen::Vector3d;  // (x, y, yaw)

  void setParams(const Params& params);

  // Must be called before initialize() / iterate().
  void setMap(const nav_msgs::msg::OccupancyGrid& map);
  void setRoot(const State& root);

  // Delete all nodes and reset counters, but keep the best-branch memory
  // for receding-horizon re-insertion.
  void clear();

  // Re-seed the tree at root and re-insert memorized best-branch nodes.
  void initialize();

  // Expand the tree by one random sample.
  void iterate();

  bool gainFound() const;
  int  getCounter() const;

  // Extract an interpolated path from the root to the highest-gain node.
  std::vector<State> getBestPath() const;

  // Return the state of the direct child of root on the best-gain branch.
  // This is the "first edge" target (receding-horizon style).
  // Returns root_ if no valid best node exists.
  State getBestEdge() const;

  // Remember the best branch so initialize() can re-use it next cycle.
  void memorizeBestBranch();

  // If the given world position is not traversable, BFS-snap it to the
  // nearest footprint-valid known-free cell and return true.
  // Returns false if the input state is already traversable.
  bool snapToFreeSpace(State& state) const;

  // Accessor for visualization.
  const std::vector<std::unique_ptr<Node>>& getNodes() const { return nodes_; }

private:
  // Return the node closest to 'state' in Euclidean (x,y) distance.
  Node* findNearest(const State& state) const;

  // Return true if the straight-line segment from->to is collision-free,
  // accounting for vehicleRadius footprint inflation.
  bool isCollisionFree(const State& from, const State& to) const;

  // Count information gain at 'state': sum of weighted unknown/free cells
  // within gainRange that are not occluded by obstacles (2-D ray casting).
  double computeGain(const State& state) const;

  // Cast a grid ray from (ox,oy) to (tx,ty).
  // Returns true if the destination cell is visible (no occupied cell blocks it).
  bool castRay(double ox, double oy, double tx, double ty) const;

  // Interpolate a motion-primitive path between two states.
  std::vector<State> samplePath(const State& from, const State& to) const;

  // OccupancyGrid helpers
  void   worldToGrid(double wx, double wy, int& gx, int& gy) const;
  bool   inBounds(int gx, int gy) const;
  int8_t cellAt(int gx, int gy) const;

  // Returns true when every grid cell within vehicleRadius of (wx, wy) is free.
  bool isWorldPointFree(double wx, double wy) const;

  Params params_;

  nav_msgs::msg::OccupancyGrid map_;
  bool   mapReady_{false};
  double res_{0.05};   // map resolution [m/cell], cached for speed
  int    width_{0};
  int    height_{0};
  double originX_{0.0};
  double originY_{0.0};

  State root_{0.0, 0.0, 0.0};

  // All nodes are owned here; pointers inside Node are non-owning.
  std::vector<std::unique_ptr<Node>> nodes_;

  Node*  bestNode_{nullptr};
  double bestGain_{0.0};
  int    counter_{0};

  // Holds the states along last best branch for receding-horizon re-insertion.
  std::vector<State> bestBranchMemory_;
};

}  // namespace nbvplanner

#endif  // NBVPLANNER_RRT_TREE_HPP
