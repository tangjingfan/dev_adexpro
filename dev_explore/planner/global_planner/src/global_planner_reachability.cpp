// Reachability BFS, single-source shortest path on the visible map, A*
// pairwise distance matrix, and the per-cell traversability gate that the
// reachability BFS and A* share. The anonymous-namespace SearchNode /
// AStarScratch types live here because they are only used by aStarSearch.
// Split out of global_planner.cpp during the 2026-05 refactor.

#include "../include/global_planner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace {
  // Stores (x,y) alongside index to avoid expensive % and / in the A* hot loop.
  struct SearchNode {
    int index{0};
    int x{0};
    int y{0};
    double f{0.0};
    double g{0.0};
    int tightRun{0};  // current consecutive below-clearance run on this path
    int tightMax{0};  // longest such run anywhere on the path to this node
  };

  struct SearchNodeComparator {
    bool operator()(const SearchNode& lhs, const SearchNode& rhs) const {
      return lhs.f > rhs.f;
    }
  };

  // Per-thread scratch space for A*. Uses a generation counter so the gScore
  // array never needs to be zeroed between calls — O(1) reset instead of O(N).
  struct AStarScratch {
    std::vector<double>   gScore;
    std::vector<uint32_t> gGen;
    uint32_t              generation{0};
    int                   capacity{0};

    void prepare(int mapSize) {
      if (mapSize > capacity) {
        gScore.resize(mapSize);
        gGen.assign(mapSize, 0u);
        capacity = mapSize;
      }
      if (++generation == 0u) {
        // Rare uint32 wrap: reset all generation markers and restart from 1.
        std::fill(gGen.begin(), gGen.begin() + capacity, 0u);
        ++generation;
      }
    }

    double get(int i) const noexcept {
      return (gGen[i] == generation) ? gScore[i]
                                     : std::numeric_limits<double>::max();
    }

    void set(int i, double val) noexcept {
      gScore[i] = val;
      gGen[i]   = generation;
    }
  };
} // namespace

ReachabilityData GlobalPlanner::computeReachability(
    const nav_msgs::msg::OccupancyGrid& visible,
    const nav_msgs::msg::OccupancyGrid& predicted,
    const Point2D& robotMapPos
) const {
  auto start_time = now();
  ReachabilityData reachability;
  const int width = static_cast<int>(visible.info.width);
  const int height = static_cast<int>(visible.info.height);
  const int mapSize = width * height;

  // Predicted map is expanded relative to visible map. Compute the grid offset
  // so visible cell (vx, vy) maps to predicted cell (vx + expX, vy + expY).
  const int predWidth = static_cast<int>(predicted.info.width);
  const int predHeight = static_cast<int>(predicted.info.height);
  const int expX = static_cast<int>(std::round(
      (visible.info.origin.position.x - predicted.info.origin.position.x) /
      visible.info.resolution));
  const int expY = static_cast<int>(std::round(
      (visible.info.origin.position.y - predicted.info.origin.position.y) /
      visible.info.resolution));

  // Predicted-only cells (visible == -1) require confidence at or above
  // confidenceThreshold_ to be considered reachable — same gate the A* in
  // isIndexTraversable applies, so the BFS reachability view and the
  // downstream A* path costs never disagree about which predicted cells are
  // safe to plan through. confidence_ is published at predicted-map resolution
  // by map_prediction_node; visible-known cells already carry confidence=1.0
  // so the gate only filters the model-filled cells.
  const bool useConfidenceGate =
      static_cast<int>(confidence_.data.size()) == predWidth * predHeight;

  reachability.visible.assign(mapSize, 0);
  reachability.predicted.assign(mapSize, 0);
  reachability.visibleDistance.assign(mapSize, -1);
  reachability.predictedDistance.assign(mapSize, -1);

  if (robotMapPos.x < 0 || robotMapPos.x >= width || robotMapPos.y < 0 ||
      robotMapPos.y >= height) {
    RCLCPP_WARN(
        get_logger(),
        "Robot position is out of map bounds: cell (%d, %d) vs map %dx%d "
        "(origin %.2f, %.2f  res %.3f)",
        robotMapPos.x, robotMapPos.y, width, height,
        visible.info.origin.position.x, visible.info.origin.position.y,
        visible.info.resolution
    );
    return reachability;
  }

  const int startIndex = robotMapPos.y * width + robotMapPos.x;
  reachability.robotIndex = startIndex;
  if (visible.data[startIndex] >= occupiedCellThreshold_) {
    RCLCPP_WARN(get_logger(), "Robot is not currently on a free visible cell");
    return reachability;
  }

  // Flat vector used as a queue: head index advances instead of dequeuing,
  // giving contiguous memory access and no deque fragmentation.
  std::vector<int> bfsQueue;
  bfsQueue.reserve(mapSize);
  bfsQueue.push_back(startIndex);
  int bfsHead = 0;
  reachability.visible[startIndex] = 1;
  reachability.predicted[startIndex] = 1;
  reachability.visibleDistance[startIndex] = 0;
  reachability.predictedDistance[startIndex] = 0;

  while (bfsHead < static_cast<int>(bfsQueue.size())) {
    const int currentIndex = bfsQueue[bfsHead++];
    const int currentX = currentIndex % width;
    const int currentY = currentIndex / width;
    const int currentDistance = std::max(
        reachability.visibleDistance[currentIndex],
        reachability.predictedDistance[currentIndex]
    );

    for (const auto& [dx, dy] : bfsDirections_) {
      const int nextX = currentX + dx;
      const int nextY = currentY + dy;

      if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
        continue;
      }

      const int nextIndex = nextY * width + nextX;
      const int8_t visibleValue = visible.data[nextIndex];
      // Predicted map uses a different (expanded) grid: convert visible (vx,vy)
      // to predicted space before indexing predicted.data.
      const int predX = nextX + expX;
      const int predY = nextY + expY;
      const int8_t predictedValue =
          (predX >= 0 && predX < predWidth && predY >= 0 && predY < predHeight)
              ? predicted.data[predY * predWidth + predX]
              : static_cast<int8_t>(-1);
      const int nextDistance = currentDistance + 1;

      if (visibleValue != -1) {
        if (!reachability.visible[nextIndex] && visibleValue == 0) {
          reachability.visible[nextIndex] = 1;
          reachability.predicted[nextIndex] = 1;
          reachability.visibleDistance[nextIndex] = nextDistance;
          reachability.predictedDistance[nextIndex] = nextDistance;
          bfsQueue.push_back(nextIndex);
        }
        continue;
      }

      // Use the same predicted-traversability threshold A* applies in
      // isIndexTraversable so the two reachability views stay in lockstep:
      // predicted-unknown (-1) is blocked; predicted in [0, threshold) is
      // walkable; predicted >= threshold is treated as obstacle.
      if (predictedValue >= 0 &&
          predictedValue < static_cast<int8_t>(predictedTraversableThreshold_) &&
          !reachability.predicted[nextIndex]) {
        if (useConfidenceGate) {
          const int predIdx = predY * predWidth + predX;
          if (confidence_.data[predIdx] <
              static_cast<float>(confidenceThreshold_)) {
            continue;  // low-confidence prediction — don't grow BFS into it
          }
        }
        reachability.predicted[nextIndex] = 1;
        reachability.predictedDistance[nextIndex] = nextDistance;
        bfsQueue.push_back(nextIndex);
      }
    }
  }

  compute_reachability_time_ms_ = (now() - start_time).seconds() * 1000.0;
  return reachability;
}

std::vector<float> GlobalPlanner::computeObstacleClearance(
    const nav_msgs::msg::OccupancyGrid& visible
) const {
  // visible is the padded visible map (already grown to the predicted grid).
  const int width  = static_cast<int>(visible.info.width);
  const int height = static_cast<int>(visible.info.height);
  const int mapSize = width * height;

  std::vector<float> clearance(mapSize, std::numeric_limits<float>::infinity());

  using Entry = std::pair<float, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;

  // Seed only observed (visible) obstacle cells as wavefront origins: clearance
  // measures distance to *known* walls only. Predicted obstacles are not used,
  // and unknown cells are not seeded, so frontiers on the free/unknown boundary
  // keep full clearance and stay selectable.
  for (int i = 0; i < mapSize; ++i) {
    if (visible.data[i] > 0) {
      clearance[i] = 0.0f;
      open.push({0.0f, i});
    }
  }

  constexpr std::array<std::pair<int, int>, 8> dirs = {{
      {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};

  while (!open.empty()) {
    auto [dist, curIdx] = open.top();
    open.pop();
    if (dist > clearance[curIdx]) continue;

    const int cx = curIdx % width;
    const int cy = curIdx / width;
    for (const auto& [dx, dy] : dirs) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const float step = (dx != 0 && dy != 0) ? 1.414f : 1.0f;
      const float nd = dist + step;
      const int ni = ny * width + nx;
      if (nd < clearance[ni]) {
        clearance[ni] = nd;
        open.push({nd, ni});
      }
    }
  }

  return clearance;
}

std::vector<double> GlobalPlanner::buildReachableDistanceMap(
    const nav_msgs::msg::OccupancyGrid& visible,
    const Point2D& robotMapPos,
    const ReachabilityData& reachability
) const {
  const int width = static_cast<int>(visible.info.width);
  const int height = static_cast<int>(visible.info.height);
  std::vector<double> distanceMap(
      width * height, std::numeric_limits<double>::max()
  );
  std::priority_queue<
      std::pair<double, int>,
      std::vector<std::pair<double, int>>,
      std::greater<>>
      openList;

  // computeReachability bails out with an all-unreachable result (robotIndex
  // -1) when the robot cell is off the grid, but this function used to index
  // straight in and write out of bounds — a hard crash of the planning thread,
  // seen as "[ros2run]: Segmentation fault" right after the "Robot position is
  // out of map bounds" warning. Return the all-max map instead: every frontier
  // then classifies as unreachable and the cycle exits through the normal
  // "No valid local frontiers" path.
  const int startIndex = reachability.robotIndex;
  if (startIndex < 0) {
    RCLCPP_WARN(
        get_logger(),
        "Robot cell (%d, %d) is outside the %dx%d map — skipping distance map",
        robotMapPos.x, robotMapPos.y, width, height
    );
    return distanceMap;
  }

  distanceMap[startIndex] = 0.0;
  openList.push({0.0, startIndex});

  while (!openList.empty()) {
    const auto [distance, currentIndex] = openList.top();
    openList.pop();

    if (distance > distanceMap[currentIndex]) {
      continue;
    }

    const int currentX = currentIndex % width;
    const int currentY = currentIndex / width;
    for (const auto& [dx, dy] : neighborDirections_) {
      const int nextX = currentX + dx;
      const int nextY = currentY + dy;

      if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
        continue;
      }

      const int nextIndex = nextY * width + nextX;
      if (!reachability.visible[nextIndex] && !reachability.predicted[nextIndex]) {
        continue;
      }

      const double stepCost =
          (dx == 0 || dy == 0) ? cardinalMoveCost_ : diagonalMoveCost_;
      const double nextDistance = distance + stepCost;
      if (nextDistance < distanceMap[nextIndex]) {
        distanceMap[nextIndex] = nextDistance;
        openList.push({nextDistance, nextIndex});
      }
    }
  }

  return distanceMap;
}

std::vector<std::vector<double>> GlobalPlanner::computeDistanceMatrix(
    const PlanningContext& ctx, const std::vector<Vertex>& vertices
) const {
  const int vertexCount = static_cast<int>(vertices.size());
  const int totalPairs = vertexCount * (vertexCount - 1) / 2;
  std::vector<std::vector<double>> distanceMatrix(
      vertexCount, std::vector<double>(vertexCount, -1.0)
  );
  const double robotYaw = getYawFromQuaternion(ctx.currentPose.orientation);

  const unsigned int hardwareConcurrency = std::thread::hardware_concurrency();
  const int workerCount =
      distanceMatrixWorkerCount_ > 0
          ? distanceMatrixWorkerCount_
          : static_cast<int>(hardwareConcurrency > 0 ? hardwareConcurrency : 1U);
  const bool useParallelExecution =
      workerCount > 1 && vertexCount >= distanceMatrixParallelMinVertices_;

  RCLCPP_INFO(
      get_logger(),
      "Distance matrix: %d vertices, %d pairs, %s (%d workers)",
      vertexCount, totalPairs,
      useParallelExecution ? "parallel" : "sequential",
      useParallelExecution ? workerCount : 1
  );

  const auto startTime = now();

  std::atomic<int> completedPairs{0};
  std::atomic<int> unreachablePairs{0};

  auto computePair = [&](int sourceIndex, int targetIndex) {
    const double dist = computeVertexPairDistance(
        ctx, vertices, sourceIndex, targetIndex, robotYaw);
    distanceMatrix[sourceIndex][targetIndex] = dist;
    ++completedPairs;
    if (dist < 0.0) {
      ++unreachablePairs;
    }
  };

  if (!useParallelExecution) {
    for (int sourceIndex = 0; sourceIndex < vertexCount; ++sourceIndex) {
      for (int targetIndex = sourceIndex + 1; targetIndex < vertexCount; ++targetIndex) {
        computePair(sourceIndex, targetIndex);
      }
    }
  } else {
    // Strided row assignment: worker i owns rows i, i+workerCount, i+2*workerCount, …
    // This interleaves heavy early rows (many pairs) with light late rows (few pairs)
    // across all workers, giving near-perfect load balance vs. contiguous block splits.
    std::vector<std::future<void>> workers;
    workers.reserve(workerCount);
    for (int workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
      workers.push_back(std::async(
          std::launch::async,
          [&computePair, workerIndex, workerCount, vertexCount]() {
            for (int sourceIndex = workerIndex; sourceIndex < vertexCount;
                 sourceIndex += workerCount) {
              for (int targetIndex = sourceIndex + 1; targetIndex < vertexCount;
                   ++targetIndex) {
                computePair(sourceIndex, targetIndex);
              }
            }
          }
      ));
    }
    for (auto& worker : workers) {
      worker.get();
    }
  }

  const double elapsedMs = (now() - startTime).seconds() * 1000.0;
  build_distance_matrix_time_ms_ = elapsedMs;
  RCLCPP_INFO(
      get_logger(),
      "Distance matrix done: %d/%d reachable pairs in %.1f ms",
      totalPairs - unreachablePairs.load(), totalPairs, elapsedMs
  );

  return distanceMatrix;
}

double GlobalPlanner::computeVertexPairDistance(
    const PlanningContext& ctx, const std::vector<Vertex>& vertices, int sourceIndex,
    int targetIndex, double robotYaw
) const {
  const auto [startX, startY] = worldToGrid(vertices[sourceIndex], ctx.predictedMap);
  const auto [goalX, goalY]   = worldToGrid(vertices[targetIndex], ctx.predictedMap);

  // The narrow-passage budget only gates robot→frontier reachability (which
  // decides whether a frontier is kept). Frontier↔frontier edges stay pure
  // shortest paths so the RPV graph isn't fragmented by intermediate pinches.
  const int tightCellBudget = (sourceIndex == 0) ? clearanceTightCellBudget_ : -1;
  const double pathDistance =
      aStarSearch(ctx, startX, startY, goalX, goalY, tightCellBudget);

  if (pathDistance < 0.0 || sourceIndex != 0) {
    return pathDistance;
  }

  const double pathDirection = std::atan2(goalY - startY, goalX - startX);
  double angleDifference = pathDirection - robotYaw;
  while (angleDifference >  M_PI) angleDifference -= 2.0 * M_PI;
  while (angleDifference < -M_PI) angleDifference += 2.0 * M_PI;

  return pathDistance + turnCostFactor_ * std::abs(angleDifference);
}

// Returns the path cost to (goalX,goalY) in grid-step units, or -1.0 if unreachable.
// Uses thread-local scratch so no O(N) allocation/zeroing happens per call.
// The octile-distance heuristic is consistent on 8-connected grids with these
// move costs (including the penalty for predicted-occupied cells), so a stale-entry
// check replaces the closed set without sacrificing optimality.
double GlobalPlanner::aStarSearch(
    const PlanningContext& ctx, int startX, int startY, int goalX, int goalY,
    int tightCellBudget
) const {
  const int width  = static_cast<int>(ctx.predictedMap.info.width);
  const int height = static_cast<int>(ctx.predictedMap.info.height);
  if (startX < 0 || startX >= width || startY < 0 || startY >= height ||
      goalX  < 0 || goalX  >= width || goalY  < 0 || goalY  >= height) {
    return -1.0;
  }

  const int startIndex = startY * width + startX;
  const int goalIndex  = goalY  * width + goalX;
  if (!isIndexTraversable(ctx, goalIndex)) {
    return -1.0;
  }

  // Narrow-passage gate: active only when a budget is requested and a clearance
  // threshold is set. A cell is "tight" when it sits closer than minClearanceCells
  // to a known obstacle; the path may cross at most tightCellBudget of them. The
  // goal cell itself is exempt (a frontier may legitimately sit at the edge).
  const bool tightGate = tightCellBudget >= 0 && ctx.minClearanceCells > 0.0f &&
                         !ctx.clearanceMap.empty();
  const auto isTightCell = [&](int index) noexcept {
    return index < static_cast<int>(ctx.clearanceMap.size()) &&
           ctx.clearanceMap[index] < ctx.minClearanceCells;
  };

  // One AStarScratch and one heap vector per thread; reused across all calls.
  thread_local AStarScratch            scratch;
  thread_local std::vector<SearchNode> heap;
  scratch.prepare(width * height);
  heap.clear();

  const SearchNodeComparator cmp{};
  const auto heuristic = [&](int x, int y) noexcept {
    const double dx   = std::abs(x - goalX);
    const double dy   = std::abs(y - goalY);
    const double diag = std::min(dx, dy);
    return diag * diagonalMoveCost_ + (std::max(dx, dy) - diag) * cardinalMoveCost_;
  };

  scratch.set(startIndex, 0.0);
  heap.push_back({startIndex, startX, startY, heuristic(startX, startY), 0.0, 0, 0});
  std::push_heap(heap.begin(), heap.end(), cmp);

  const auto occThreshold = static_cast<int8_t>(occupiedCellThreshold_);

  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), cmp);
    const SearchNode current = heap.back();
    heap.pop_back();

    // Stale duplicate: a shorter path to this node was already settled.
    if (current.g > scratch.get(current.index)) {
      continue;
    }

    if (current.index == goalIndex) {
      // Drop the frontier only if reaching it requires threading a single tight
      // stretch longer than the budget (a genuine narrow tunnel). We test the
      // LONGEST CONSECUTIVE tight run, not the total count — total grows with
      // path length, so a long journey that merely brushes walls here and there
      // would be wrongly rejected, stranding far-away regions.
      // Returns -2.0 (not -1.0) so the caller can distinguish clearance-dropped
      // vertices from truly unreachable ones for visualization.
      if (tightGate && current.tightMax > tightCellBudget) {
        return -2.0;
      }
      return current.g;
    }

    for (const auto& [dx, dy] : neighborDirections_) {
      const int nextX = current.x + dx;
      const int nextY = current.y + dy;
      if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
        continue;
      }

      const int nextIndex = nextY * width + nextX;
      if (!isIndexTraversable(ctx, nextIndex)) {
        continue;
      }

      if (dx != 0 && dy != 0) {
        const int sideA = current.y * width + nextX;
        const int sideB = nextY    * width + current.x;
        if (!isIndexTraversable(ctx, sideA) || !isIndexTraversable(ctx, sideB)) {
          continue;
        }
      }

      const double baseStepCost =
          (dx == 0 || dy == 0) ? cardinalMoveCost_ : diagonalMoveCost_;
      double stepCost = baseStepCost;
      if (ctx.predictedMap.data[nextIndex] >= occThreshold) {
        stepCost += predictedOccupiedPenalty_;
      }
      // Surcharge for cells whose only justification for being traversable
      // is the prediction (visible map says unknown). Summed along the
      // path this yields total ≈ length * (1 + alpha * predicted_fraction).
      // Keeps the A* heuristic admissible (it remains a lower bound).
      if (predictedEdgeInflationAlpha_ > 0.0 &&
          nextIndex < static_cast<int>(ctx.visibleMap.data.size()) &&
          ctx.visibleMap.data[nextIndex] == -1) {
        stepCost += predictedEdgeInflationAlpha_ * baseStepCost;
      }

      // Tight = below the clearance threshold. Mildly penalize so A* prefers a
      // corridor's wider centre when routes of similar length exist; this keeps
      // the consecutive-run measurement below anchored to the genuinely
      // unavoidable narrow stretch rather than an arbitrary equal-cost path that
      // happens to hug a wall. The goal cell is exempt (a frontier may sit at an
      // edge).
      const bool tight =
          tightGate && nextIndex != goalIndex && isTightCell(nextIndex);
      if (tight) {
        stepCost += 0.25 * baseStepCost;
      }

      const double tentativeG = current.g + stepCost;
      if (tentativeG >= scratch.get(nextIndex)) {
        continue;
      }

      // Longest consecutive tight run; resets on any non-tight cell.
      const int nextRun = tight ? current.tightRun + 1 : 0;
      const int nextMax = std::max(current.tightMax, nextRun);

      scratch.set(nextIndex, tentativeG);
      heap.push_back({nextIndex, nextX, nextY,
                      tentativeG + heuristic(nextX, nextY), tentativeG,
                      nextRun, nextMax});
      std::push_heap(heap.begin(), heap.end(), cmp);
    }
  }

  return -1.0;
}

bool GlobalPlanner::isIndexTraversable(
    const PlanningContext& ctx, int index) const {
  // Clearance is no longer a hard per-cell gate here — low-clearance cells stay
  // traversable and are instead counted against the narrow-passage budget in
  // aStarSearch (see isTightCell). This stops a single pinch from disconnecting
  // a whole region behind it, which used to strand large explorable areas.

  // A* operates in predictedMap space. ctx.visibleMap has been padded to the
  // same grid (padVisibleToPredicted in backgroundPlanningLoop), so cells from
  // both maps share indices 1:1, and ctx.confidenceMap is produced at the same
  // predictedMap resolution by map_prediction_node.

  // Observed ground truth overrides prediction in BOTH directions. visibleMap
  // updates far more often than the (slow) predictedMap, so a cell the robot
  // has freshly observed as free can still read stale-unknown/occupied in
  // predictedMap (its known region was overwritten with the older inference-time
  // visible map). Trust the observation: visible-free is always traversable,
  // mirroring the visible>0 hard-block below. Without this, freshly-revealed
  // free cells at the exploration edge are vetoed by the stale predicted map,
  // isolating local-frontier vertices and breaking the RPV local plan.
  if (index < static_cast<int>(ctx.visibleMap.data.size()) &&
      ctx.visibleMap.data[index] == 0) {
    return true;
  }

  const int8_t value = ctx.predictedMap.data[index];
  // Match computeReachability: predicted-unknown is NOT traversable, and the
  // upper bound uses the same predictedTraversableThreshold_ everywhere.
  if (value < 0) return false;
  if (value >= static_cast<int8_t>(predictedTraversableThreshold_)) return false;

  // Hard-block if visible reports any occupancy probability (>0). Visible-known
  // obstacles can't be overridden by prediction even at high confidence; this
  // mirrors waypoint_bridge's gate so the two A* implementations agree on
  // traversability.
  if (index < static_cast<int>(ctx.visibleMap.data.size()) &&
      ctx.visibleMap.data[index] > 0) {
    return false;
  }

  // Require prediction confidence above threshold; visible-known cells already
  // carry confidence == 1.0 in map_prediction_node's output, so this only
  // gates the cells the model has filled in.
  if (index < static_cast<int>(ctx.confidenceMap.data.size()) &&
      ctx.confidenceMap.data[index] < static_cast<float>(confidenceThreshold_)) {
    return false;
  }

  return true;
}
