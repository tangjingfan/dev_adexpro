#include "nbvplanner/rrt_tree.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <queue>

namespace nbvplanner {

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

void RrtTree::setParams(const Params& params) {
  params_ = params;
}

void RrtTree::setMap(const nav_msgs::msg::OccupancyGrid& map) {
  map_     = map;
  res_     = map.info.resolution;
  width_   = static_cast<int>(map.info.width);
  height_  = static_cast<int>(map.info.height);
  originX_ = map.info.origin.position.x;
  originY_ = map.info.origin.position.y;
  mapReady_ = (width_ > 0 && height_ > 0 && res_ > 0.0);
}

void RrtTree::setRoot(const State& root) {
  root_ = root;
}

// ---------------------------------------------------------------------------
// Tree lifecycle
// ---------------------------------------------------------------------------

void RrtTree::clear() {
  nodes_.clear();
  bestNode_  = nullptr;
  bestGain_  = params_.zeroGain;
  counter_   = 0;
}

void RrtTree::initialize() {
  // Seed the tree with the root node.
  auto rootNode       = std::make_unique<Node>();
  rootNode->state     = root_;
  rootNode->distance  = 0.0;
  rootNode->gain      = params_.zeroGain;
  rootNode->parent    = nullptr;
  bestNode_           = rootNode.get();
  nodes_.push_back(std::move(rootNode));

  // Re-insert memorized best-branch states (receding horizon).
  // Each state is checked for collisions before being added.
  for (auto it = bestBranchMemory_.rbegin(); it != bestBranchMemory_.rend(); ++it) {
    const State& newState = *it;

    Node* nearest = findNearest(newState);
    if (!nearest) continue;

    if (!isCollisionFree(nearest->state, newState)) continue;

    Eigen::Vector3d dir(newState[0] - nearest->state[0],
                        newState[1] - nearest->state[1], 0.0);

    auto newNode       = std::make_unique<Node>();
    newNode->state     = newState;
    newNode->parent    = nearest;
    newNode->distance  = nearest->distance + dir.norm();
    newNode->gain      = nearest->gain
                         + computeGain(newState)
                               * std::exp(-params_.degressiveCoeff * newNode->distance);
    nearest->children.push_back(newNode.get());

    if (newNode->gain > bestGain_) {
      bestGain_ = newNode->gain;
      bestNode_ = newNode.get();
    }
    counter_++;
    nodes_.push_back(std::move(newNode));
  }
}

// ---------------------------------------------------------------------------
// RRT expansion (one iteration)
// ---------------------------------------------------------------------------

void RrtTree::iterate() {
  if (!mapReady_) return;

  // Sample a random state inside the exploration sphere offset from root.
  // Reject samples that violate hard bounds (unless softBounds is set).
  const double radius = std::sqrt(
      std::pow(params_.maxX - params_.minX, 2.0) +
      std::pow(params_.maxY - params_.minY, 2.0));

  State newState;
  bool  solutionFound = false;
  for (int attempt = 0; attempt < 100 && !solutionFound; ++attempt) {
    // Uniform sample on a disc of the exploration diagonal
    for (int i = 0; i < 2; ++i) {
      newState[i] = 2.0 * radius * (static_cast<double>(rand()) / RAND_MAX - 0.5);
    }
    if (newState[0] * newState[0] + newState[1] * newState[1] > radius * radius) {
      continue;
    }
    newState += root_;  // offset by current root

    if (!params_.softBounds) {
      if (newState[0] < params_.minX || newState[0] > params_.maxX) continue;
      if (newState[1] < params_.minY || newState[1] > params_.maxY) continue;
    }
    solutionFound = true;
  }
  if (!solutionFound) return;

  // Steer: clamp edge length to extensionRange.
  Node* nearest = findNearest(newState);
  if (!nearest) return;

  Eigen::Vector3d dir(newState[0] - nearest->state[0],
                      newState[1] - nearest->state[1], 0.0);
  if (dir.norm() > params_.extensionRange) {
    dir = params_.extensionRange * dir.normalized();
  }
  newState[0] = nearest->state[0] + dir[0];
  newState[1] = nearest->state[1] + dir[1];
  newState[2] = 2.0 * M_PI * (static_cast<double>(rand()) / RAND_MAX - 0.5);

  // Reject if the edge hits an obstacle.
  if (!isCollisionFree(nearest->state, newState)) return;

  // Insert into tree and accumulate gain with exponential distance discount.
  auto newNode       = std::make_unique<Node>();
  newNode->state     = newState;
  newNode->parent    = nearest;
  newNode->distance  = nearest->distance + dir.norm();
  newNode->gain      = nearest->gain
                       + computeGain(newState)
                             * std::exp(-params_.degressiveCoeff * newNode->distance);
  nearest->children.push_back(newNode.get());

  if (newNode->gain > bestGain_) {
    bestGain_ = newNode->gain;
    bestNode_ = newNode.get();
  }
  counter_++;
  nodes_.push_back(std::move(newNode));
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

bool RrtTree::gainFound() const {
  return bestGain_ > params_.zeroGain;
}

int RrtTree::getCounter() const {
  return counter_;
}

// Walk from root to best node and interpolate the whole branch.
std::vector<RrtTree::State> RrtTree::getBestPath() const {
  if (!bestNode_ || !bestNode_->parent) return {};

  // Collect ancestor chain from best node up to (but not including) root.
  std::vector<Node*> chain;
  Node* cur = bestNode_;
  while (cur->parent != nullptr) {
    chain.push_back(cur);
    cur = cur->parent;
  }
  // chain is ordered best→root; reverse to get root→best
  std::reverse(chain.begin(), chain.end());

  std::vector<State> fullPath;
  for (Node* node : chain) {
    auto seg = samplePath(node->parent->state, node->state);
    if (seg.empty()) continue;

    // Keep segments continuous without duplicating junction points.
    if (fullPath.empty()) {
      fullPath.insert(fullPath.end(), seg.begin(), seg.end());
    } else {
      fullPath.insert(fullPath.end(), seg.begin() + 1, seg.end());
    }
  }
  return fullPath;
}

RrtTree::State RrtTree::getBestEdge() const {
  if (!bestNode_ || !bestNode_->parent) return root_;

  // Walk from bestNode_ toward root until we find the direct child of root.
  // The root node has parent == nullptr, so its direct child has parent->parent == nullptr.
  Node* cur = bestNode_;
  while (cur->parent && cur->parent->parent != nullptr) {
    cur = cur->parent;
  }
  // cur is now the direct child of root — the first-step target.
  return cur->state;
}

void RrtTree::memorizeBestBranch() {
  bestBranchMemory_.clear();
  Node* cur = bestNode_;
  // Walk from best node toward root, skipping root itself.
  while (cur->parent && cur->parent->parent) {
    bestBranchMemory_.push_back(cur->state);
    cur = cur->parent;
  }
}

// ---------------------------------------------------------------------------
// Snap-to-free-space (BFS from current cell outward)
// ---------------------------------------------------------------------------

bool RrtTree::snapToFreeSpace(State& state) const {
  if (!mapReady_) return false;

  // Already traversable with the configured footprint inflation.
  if (isWorldPointFree(state[0], state[1])) return false;

  int startGx, startGy;
  worldToGrid(state[0], state[1], startGx, startGy);
  startGx = std::clamp(startGx, 0, width_ - 1);
  startGy = std::clamp(startGy, 0, height_ - 1);

  auto isCellKnownFree = [&](int gx, int gy) -> bool {
    if (!inBounds(gx, gy)) return false;
    int8_t v = cellAt(gx, gy);
    return (v != -1) && (v < params_.occupiedThreshold);
  };

  // BFS outward until we find a free cell.
  // Use a flat visited array to avoid allocating a set per call.
  std::vector<bool> visited(static_cast<size_t>(width_ * height_), false);
  std::queue<std::pair<int,int>> q;

  auto enqueue = [&](int gx, int gy) {
    if (!inBounds(gx, gy)) return;
    const size_t idx = static_cast<size_t>(gy * width_ + gx);
    if (visited[idx]) return;
    visited[idx] = true;
    q.push({gx, gy});
  };

  enqueue(startGx, startGy);

  // 4-connectivity BFS — fast enough; typically converges in a few cells.
  static const int dx4[] = {1, -1, 0,  0};
  static const int dy4[] = {0,  0, 1, -1};

  while (!q.empty()) {
    auto [cx, cy] = q.front();
    q.pop();

    if (isCellKnownFree(cx, cy)) {
      // Candidate must also be traversable under footprint inflation.
      const double wx = originX_ + (cx + 0.5) * res_;
      const double wy = originY_ + (cy + 0.5) * res_;
      if (isWorldPointFree(wx, wy)) {
        state[0] = wx;
        state[1] = wy;
        return true;
      }
    }

    for (int d = 0; d < 4; ++d) {
      enqueue(cx + dx4[d], cy + dy4[d]);
    }
  }

  return false;  // entire map has no free cell (shouldn't happen in practice)
}

// ---------------------------------------------------------------------------
// Nearest-neighbour (linear scan — tree stays small, ≤500 nodes)
// ---------------------------------------------------------------------------

Node* RrtTree::findNearest(const State& state) const {
  Node* best = nullptr;
  double bestDist = std::numeric_limits<double>::max();
  for (const auto& n : nodes_) {
    double d = std::hypot(n->state[0] - state[0], n->state[1] - state[1]);
    if (d < bestDist) {
      bestDist = d;
      best     = n.get();
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Collision checking
// ---------------------------------------------------------------------------

bool RrtTree::isCollisionFree(const State& from, const State& to) const {
  if (!mapReady_) return false;

  Eigen::Vector3d dir(to[0] - from[0], to[1] - from[1], 0.0);
  double len = dir.norm();
  if (len < 1e-6) return true;

  // Step size: half a grid resolution; +1 gives one extra check past the endpoint.
  const double step  = res_ * 0.5;
  const int    steps = static_cast<int>(std::ceil(len / step)) + 1;

  for (int i = 0; i <= steps; ++i) {
    double t  = std::min(1.0, (i * step) / len);
    double wx = from[0] + t * dir[0];
    double wy = from[1] + t * dir[1];

    if (!isWorldPointFree(wx, wy)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Information-gain computation (2-D adaptation of original volume gain)
// ---------------------------------------------------------------------------

double RrtTree::computeGain(const State& state) const {
  if (!mapReady_) return 0.0;

  const double range    = params_.gainRange;
  const double rangeSq  = range * range;
  double       gain     = 0.0;

  // Iterate over all grid cells within the bounding square of the gain disc.
  int gxMin, gyMin, gxMax, gyMax;
  worldToGrid(state[0] - range, state[1] - range, gxMin, gyMin);
  worldToGrid(state[0] + range, state[1] + range, gxMax, gyMax);

  gxMin = std::max(gxMin, 0);
  gyMin = std::max(gyMin, 0);
  gxMax = std::min(gxMax, width_ - 1);
  gyMax = std::min(gyMax, height_ - 1);

  for (int gx = gxMin; gx <= gxMax; ++gx) {
    for (int gy = gyMin; gy <= gyMax; ++gy) {
      // World coordinates of cell centre
      double wx = originX_ + (gx + 0.5) * res_;
      double wy = originY_ + (gy + 0.5) * res_;

      double dx = wx - state[0];
      double dy = wy - state[1];
      if (dx * dx + dy * dy > rangeSq) continue;

      int8_t val = cellAt(gx, gy);

      if (val == -1) {
        // Unknown cell: count only if not occluded.
        if (castRay(state[0], state[1], wx, wy)) {
          gain += params_.igUnmapped;
        }
      } else if (val < params_.occupiedThreshold) {
        // Free cell
        if (params_.igFree > 0.0 && castRay(state[0], state[1], wx, wy)) {
          gain += params_.igFree;
        }
      }
      // Occupied cells contribute nothing.
    }
  }

  // Scale by cell area (analogous to disc^2 in 2D).
  gain *= res_ * res_;
  return gain;
}

// ---------------------------------------------------------------------------
// 2-D ray casting (Bresenham on the grid)
// ---------------------------------------------------------------------------

bool RrtTree::castRay(double ox, double oy, double tx, double ty) const {
  int gx0, gy0, gx1, gy1;
  worldToGrid(ox, oy, gx0, gy0);
  worldToGrid(tx, ty, gx1, gy1);

  // Bresenham's line algorithm
  int dx = std::abs(gx1 - gx0);
  int dy = std::abs(gy1 - gy0);
  int sx = (gx0 < gx1) ? 1 : -1;
  int sy = (gy0 < gy1) ? 1 : -1;
  int err = dx - dy;

  int cx = gx0;
  int cy = gy0;

  while (cx != gx1 || cy != gy1) {
    // Stop if we leave the map (target cell may be unmapped — that's fine)
    if (!inBounds(cx, cy)) return false;

    int8_t cell = cellAt(cx, cy);
    if (cell >= params_.occupiedThreshold) return false;  // blocked

    int e2 = 2 * err;
    if (e2 > -dy) { err -= dy; cx += sx; }
    if (e2 <  dx) { err += dx; cy += sy; }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Path interpolation (motion primitive)
// ---------------------------------------------------------------------------

std::vector<RrtTree::State> RrtTree::samplePath(
    const State& from, const State& to) const {
  std::vector<State> path;

  Eigen::Vector3d dist(to[0] - from[0], to[1] - from[1], 0.0);
  double yawDiff = to[2] - from[2];
  // Wrap yaw difference to [-π, π]
  while (yawDiff >  M_PI) yawDiff -= 2.0 * M_PI;
  while (yawDiff < -M_PI) yawDiff += 2.0 * M_PI;

  const double linLen = dist.norm();
  if (linLen < 1e-6) return path;

  // Fixed interpolation step: 1/10 of the extension range.
  const double disc = std::min(params_.extensionRange * 0.1 / linLen, 1.0);

  for (double it = 0.0; it <= 1.0; it += disc) {
    double t = std::min(it, 1.0);
    State s;
    s[0] = from[0] + t * dist[0];
    s[1] = from[1] + t * dist[1];
    s[2] = from[2] + t * yawDiff;
    // Wrap yaw
    while (s[2] >  M_PI) s[2] -= 2.0 * M_PI;
    while (s[2] < -M_PI) s[2] += 2.0 * M_PI;
    path.push_back(s);
  }
  // Always include the exact endpoint
  if (path.empty() || (path.back() - to).head<2>().norm() > 1e-3) {
    path.push_back(to);
  }
  return path;
}

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------

void RrtTree::worldToGrid(double wx, double wy, int& gx, int& gy) const {
  gx = static_cast<int>((wx - originX_) / res_);
  gy = static_cast<int>((wy - originY_) / res_);
}

bool RrtTree::inBounds(int gx, int gy) const {
  return gx >= 0 && gx < width_ && gy >= 0 && gy < height_;
}

int8_t RrtTree::cellAt(int gx, int gy) const {
  return map_.data[static_cast<size_t>(gy * width_ + gx)];
}

bool RrtTree::isWorldPointFree(double wx, double wy) const {
  // Check a disc of radius vehicleRadius around (wx, wy).
  const double r    = params_.vehicleRadius;
  const double rSq  = r * r;
  const int    dgxy = static_cast<int>(std::ceil(r / res_));

  int cgx, cgy;
  worldToGrid(wx, wy, cgx, cgy);

  for (int dx = -dgxy; dx <= dgxy; ++dx) {
    for (int dy = -dgxy; dy <= dgxy; ++dy) {
      // Only check cells whose centres fall inside the metric footprint.
      const double ox = dx * res_;
      const double oy = dy * res_;
      if (ox * ox + oy * oy > rSq) continue;

      int gx = cgx + dx;
      int gy = cgy + dy;
      if (!inBounds(gx, gy)) return false;  // treat out-of-bounds as occupied

      int8_t val = cellAt(gx, gy);
      if (!params_.allowUnknownTraversal && val == -1) return false;
      if (val >= params_.occupiedThreshold) return false;
    }
  }
  return true;
}

}  // namespace nbvplanner
