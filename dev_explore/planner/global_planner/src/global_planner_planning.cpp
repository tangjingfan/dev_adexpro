// Planning core: builds vertices from frontiers, drives the RPV solver,
// publishes the resulting nav_msgs::Path, and handles the failure /
// fallback chain (local → global → "stay put" 1-pose path). Split out of
// global_planner.cpp during the 2026-05 refactor.

#include "../include/global_planner.hpp"

#include "graph.hpp"
#include "search_rpv.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

std::vector<Vertex> GlobalPlanner::prepareLocalPlanningVertices(
    const PlanningContext& ctx
) const {
  std::vector<Vertex> vertices;
  vertices.push_back(
      {0, ctx.currentPose.position.x, ctx.currentPose.position.y, 1.0});

  // ctx.localFrontiers is already the output of classifyAndPublishFrontiers,
  // which (via getClusteredMapEdgeFrontiers / getClusteredPredictedFrontiers)
  // has filtered each candidate through isLocalFrontierTraversable plus an
  // isPointInFreeSpace gate. Re-running those checks here used to be a no-op
  // round-trip; we just promote each frontier cell to world coords now.
  const double originX = ctx.visibleMap.info.origin.position.x;
  const double originY = ctx.visibleMap.info.origin.position.y;
  const double resolution = ctx.visibleMap.info.resolution;
  int vertexId = 1;
  for (const auto& frontier : ctx.localFrontiers) {
    const double worldX = originX + (frontier.x + 0.5) * resolution;
    const double worldY = originY + (frontier.y + 0.5) * resolution;
    vertices.push_back({vertexId++, worldX, worldY, frontier.prob});
  }

  return vertices;
}

std::vector<FrontierTask> GlobalPlanner::collectGlobalPlanningTasks(
    const PlanningContext& ctx
) const {
  std::vector<FrontierTask> tasks;
  const int width = static_cast<int>(ctx.visibleMap.info.width);
  const int originalWidth = static_cast<int>(ctx.originalVisibleMap.info.width);
  const int originalHeight = static_cast<int>(ctx.originalVisibleMap.info.height);
  const auto [robotX, robotY] = worldToGrid(
      ctx.currentPose.position.x, ctx.currentPose.position.y, ctx.visibleMap
  );

  for (const auto& frontierPoint : ctx.globalFrontierMsg.points) {
    const auto [originalFrontierX, originalFrontierY] = worldToGrid(
        frontierPoint.point.point.x,
        frontierPoint.point.point.y,
        ctx.originalVisibleMap
    );
    if (originalFrontierX < 0 || originalFrontierX >= originalWidth ||
        originalFrontierY < 0 || originalFrontierY >= originalHeight) {
      continue;
    }

    const auto [frontierX, frontierY] = worldToGrid(
        frontierPoint.point.point.x, frontierPoint.point.point.y, ctx.visibleMap
    );
    if (frontierX < 0 || frontierX >= width || frontierY < 0 ||
        frontierY >= static_cast<int>(ctx.visibleMap.info.height)) {
      continue;
    }

    const int index = frontierY * width + frontierX;
    const int8_t visibleValue = ctx.visibleMap.data[index];
    const int8_t predictedValue = ctx.predictedMap.data[index];
    if (visibleValue == -1 && predictedValue == -1) {
      continue;
    }

    const double gridDistance = std::hypot(frontierX - robotX, frontierY - robotY);
    if (gridDistance < rangeThresholdGlobalFrontier_) {
      continue;
    }

    const bool usePredictedReachability = (visibleValue == -1);
    const bool reachable = usePredictedReachability
                               ? static_cast<bool>(ctx.reachability.predicted[index])
                               : static_cast<bool>(ctx.reachability.visible[index]);
    const int pathDistance = usePredictedReachability
                                 ? ctx.reachability.predictedDistance[index]
                                 : ctx.reachability.visibleDistance[index];
    if (!reachable || pathDistance < 0) {
      continue;
    }

    FrontierTask task;
    task.msgPoint = frontierPoint;
    task.pathDistance = static_cast<double>(pathDistance);
    task.vertex = {
        0,
        frontierPoint.point.point.x,
        frontierPoint.point.point.y,
        frontierPoint.prob};
    tasks.push_back(task);
  }

  std::sort(
      tasks.begin(),
      tasks.end(),
      [](const FrontierTask& lhs, const FrontierTask& rhs) {
        return lhs.pathDistance < rhs.pathDistance;
      }
  );
  return tasks;
}

std::vector<Vertex> GlobalPlanner::buildGlobalPlanningVertices(
    const PlanningContext& ctx, const std::vector<FrontierTask>& tasks
) const {
  std::vector<Vertex> vertices;
  vertices.push_back({0, ctx.currentPose.position.x, ctx.currentPose.position.y, 1.0}
  );

  const int count = std::min(static_cast<int>(tasks.size()), maxFrontiersToPlan_);
  for (int index = 0; index < count; ++index) {
    Vertex vertex = tasks[index].vertex;
    vertex.id = index + 1;
    vertices.push_back(vertex);
  }

  return vertices;
}

void GlobalPlanner::updateGlobalGuide(const PlanningContext& ctx) {
  if (!globalGuideEnabled_) {
    globalGuide_.reset();
    return;
  }

  // Keep the cached guide until its period is up. The guide answers "which way
  // do I leave the local horizon", which is stable over many local replans,
  // and refreshing it costs a full distance matrix + RPV solve over the global
  // frontier set.
  if (globalGuide_ && ++cyclesSinceGlobalGuide_ < globalGuidePeriod_) {
    return;
  }
  cyclesSinceGlobalGuide_ = 0;
  // Every exit below leaves either a fresh guide or none (only this planning
  // thread touches globalGuide_).
  globalGuide_.reset();

  const auto tasks = collectGlobalPlanningTasks(ctx);
  const auto vertices = buildGlobalPlanningVertices(ctx, tasks);
  if (vertices.size() < 2) {
    // Nothing outside the local horizon (or nothing reachable out there):
    // the local set is the whole problem, so plan without a boundary
    // condition rather than steering toward a stale guide.
    return;
  }

  std::vector<long> pathIndices;
  if (!solvePlanningProblem(ctx, vertices, pathIndices)) {
    return;
  }

  // First non-robot stop of the coarse tour: where the global level wants the
  // robot to end up once the local horizon is exhausted.
  for (const long pathIndex : pathIndices) {
    if (pathIndex > 0 && pathIndex < static_cast<long>(vertices.size())) {
      globalGuide_ = vertices[pathIndex];
      RCLCPP_INFO(
          get_logger(), "Global guide -> (%.2f, %.2f), prob %.2f, from %zu global frontiers",
          globalGuide_->x, globalGuide_->y, globalGuide_->probability, vertices.size() - 1
      );
      return;
    }
  }
}

void GlobalPlanner::appendGlobalGuideVertex(std::vector<Vertex>& vertices) const {
  if (!globalGuide_ || vertices.empty()) {
    return;
  }
  Vertex guide = *globalGuide_;
  guide.id = static_cast<int>(vertices.size());
  vertices.push_back(guide);
}

int GlobalPlanner::findCommittedTargetVertexIndex(
    const std::vector<Vertex>& vertices
) const {
  if (!lastCommittedTarget_ || localHysteresisMatchRadius_ <= 0.0) {
    return -1;
  }
  const double radiusSq =
      localHysteresisMatchRadius_ * localHysteresisMatchRadius_;
  int bestIndex = -1;
  double bestSq = radiusSq;
  for (size_t i = 1; i < vertices.size(); ++i) {
    const double dx = vertices[i].x - lastCommittedTarget_->first;
    const double dy = vertices[i].y - lastCommittedTarget_->second;
    const double sq = dx * dx + dy * dy;
    if (sq <= bestSq) {
      bestSq = sq;
      bestIndex = static_cast<int>(i);
    }
  }
  return bestIndex;
}

bool GlobalPlanner::runLocalPlanning(
    const PlanningContext& ctx, const std::vector<Vertex>& vertices
) {
  auto start_time = now();

  std::vector<long> pathIndices;
  const int preferredTargetIndex = findCommittedTargetVertexIndex(vertices);
  if (!solvePlanningProblem(ctx, vertices, pathIndices, preferredTargetIndex)) {
    local_plan_time_ms_ = (now() - start_time).seconds() * 1000.0;
    return false;
  }

  publishPath(ctx, pathIndices, vertices);
  // Record the first non-robot vertex as the next committed target so the
  // following cycle can bias its robot-edge cost toward staying with it.
  for (const long pathIndex : pathIndices) {
    if (pathIndex > 0 && pathIndex < static_cast<long>(vertices.size())) {
      lastCommittedTarget_ = {vertices[pathIndex].x, vertices[pathIndex].y};
      break;
    }
  }
  handlePlanningSuccess(ctx);
  local_plan_time_ms_ = (now() - start_time).seconds() * 1000.0;
  return true;
}

bool GlobalPlanner::runGlobalPlanning(const PlanningContext& ctx) {
  auto start_time = now();
  const auto tasks = collectGlobalPlanningTasks(ctx);
  const auto vertices = buildGlobalPlanningVertices(ctx, tasks);
  if (vertices.size() < 2) {
    RCLCPP_WARN(get_logger(), "No valid global planning vertices are available");
    global_plan_time_ms_ = (now() - start_time).seconds() * 1000.0;
    return false;
  }

  std::vector<long> pathIndices;
  if (!solvePlanningProblem(ctx, vertices, pathIndices)) {
    RCLCPP_WARN(get_logger(), "Global planning failed");
    global_plan_time_ms_ = (now() - start_time).seconds() * 1000.0;
    return false;
  }

  publishPath(ctx, pathIndices, vertices);
  // Global planning has just replaced whatever local intent we had; the
  // previous committed target no longer applies.
  lastCommittedTarget_.reset();
  handlePlanningSuccess(ctx);
  global_plan_time_ms_ = (now() - start_time).seconds() * 1000.0;
  return true;
}

bool GlobalPlanner::solvePlanningProblem(
    const PlanningContext& ctx, const std::vector<Vertex>& vertices,
    std::vector<long>& pathIndices, int preferredTargetIndex
) const {
  if (vertices.size() < 2) {
    return false;
  }

  const auto distanceMatrix = computeDistanceMatrix(ctx, vertices);

  // Drop every frontier the robot (vertex 0) cannot reach by A*
  std::vector<size_t> keep;  // list of indices of vertices to keep
  keep.reserve(vertices.size());
  keep.push_back(0);
  for (size_t j = 1; j < vertices.size(); ++j) {
    if (distanceMatrix[0][j] > 0.0) {
      keep.push_back(j);
      continue;
    }
    const auto [vx, vy] = worldToGrid(vertices[j], ctx.predictedMap);
    const int width = static_cast<int>(ctx.predictedMap.info.width);
    const int idx = vy * width + vx;
    const int vis = (idx >= 0 && idx < static_cast<int>(ctx.visibleMap.data.size())) ? ctx.visibleMap.data[idx] : -99;
    const int pred = (idx >= 0 && idx < static_cast<int>(ctx.predictedMap.data.size())) ? ctx.predictedMap.data[idx] : -99;
    const float conf = (idx >= 0 && idx < static_cast<int>(ctx.confidenceMap.data.size())) ? ctx.confidenceMap.data[idx] : -1.0f;
    RCLCPP_WARN(get_logger(),
        "Dropped robot-unreachable vertex %zu @grid(%d,%d): visible=%d predicted=%d confidence=%.2f", j, vx, vy, vis, pred, conf);
  }

  // Visualize vertices that were dropped specifically by the narrow-passage
  // clearance gate (tightMax > tightCellBudget) so the operator can see which
  // frontiers are being sacrificed to the clearance budget.
  publishClearanceDroppedVertices(vertices, distanceMatrix, ctx.predictedMap);

  // solve failed when only the robot vertex remains
  if (keep.size() < 2) {
    RCLCPP_WARN(get_logger(), "Robot vertex has no reachable frontiers — skipping RPV solver");
    return false;
  }

  // The RPV* solver packs the visited set into a uint64_t bitmask, so it rejects
  // more than 64 vertices (a 2^n search never finishes near that count anyway).
  // With sensible frontier spacing the survivor set stays well below; if it ever
  // exceeds, keep the robot plus the 63 nearest reachable frontiers — nearest is
  // the order the anytime search commits to first — instead of letting SetGraph
  // throw. keep must stay ascending for the upper-triangle graph build below.
  constexpr size_t kMaxSolverVertices = raplab::kRPVMaxVertices;
  if (keep.size() > kMaxSolverVertices) {
    std::sort(keep.begin() + 1, keep.end(), [&](size_t l, size_t r) {
      return distanceMatrix[0][l] < distanceMatrix[0][r];
    });
    keep.resize(kMaxSolverVertices);
    std::sort(keep.begin(), keep.end());
    RCLCPP_WARN(
        get_logger(),
        "Frontier set exceeded %zu vertices; capped to the nearest for the RPV solver",
        kMaxSolverVertices
    );
  }

  raplab::SparseGraph graph;
  std::vector<double> probabilities;
  probabilities.reserve(keep.size());

  // Soft-commitment hysteresis: discount the robot's edge to the previously
  // committed next-hop target so the RPV solver biases toward keeping the same
  // first hop across replans. preferredTargetIndex is an index into the
  // original `vertices`; it may have been dropped above, in which case no edge
  // matches and the discount is simply a no-op.
  const bool applyHysteresis =
      preferredTargetIndex > 0 &&
      preferredTargetIndex < static_cast<int>(vertices.size()) &&
      localHysteresisFactor_ < 1.0 && localHysteresisFactor_ > 0.0;

  // keep is ascending, so keep[a] < keep[b] always indexes the upper triangle
  // computeDistanceMatrix actually filled. Every survivor has distanceMatrix
  // [0][keep[a]] > 0.0, so vertex 0 gains an edge to all of them and the graph
  // is connected by construction.
  for (size_t a = 0; a < keep.size(); ++a) {
    graph.AddVertex(static_cast<long>(a));
    probabilities.push_back(vertices[keep[a]].probability);

    for (size_t b = a + 1; b < keep.size(); ++b) {
      double cost = distanceMatrix[keep[a]][keep[b]];
      if (cost <= 0.0) {
        continue;
      }
      if (applyHysteresis && a == 0 && static_cast<int>(keep[b]) == preferredTargetIndex) {
        cost *= localHysteresisFactor_;
        const auto hits = softCommitmentHits_.fetch_add(1) + 1;
        RCLCPP_INFO(
            get_logger(),
            "Soft-commitment hysteresis triggered: discounted robot→vertex %d by ×%.3f (total: %llu)",
            preferredTargetIndex, localHysteresisFactor_, hits
        );
      }
      graph.AddEdge(static_cast<long>(a), static_cast<long>(b), {cost});
    }
  }

  const auto solveStart = now();
  raplab::RPVResult result;
  // Anytime focal search (RPV*): seeded with a greedy nearest-neighbour tour and
  // an admissible PT heuristic, it returns the best complete tour found within
  // the time limit instead of failing outright on timeout — so a large vertex
  // set degrades to a bounded-suboptimal plan rather than stalling the bridge.
  // It still returns the exact optimum whenever it converges before the limit.
  const int solverStatus = raplab::RunMinSumRPVFocal(
      &graph, 0, probabilities, planningTimeLimit_,
      /*use_heuristic=*/true, raplab::HeuristicMode::PT, result
  );
  const double solveMs = (now() - solveStart).seconds() * 1000.0;

  // Vertex count still drives the search exponentially; the focal seed means a
  // timeout now degrades to a suboptimal tour rather than no tour, but a genuine
  // "no complete tour" (path empty / status -1) still fails the plan. Log the
  // pair so the cost cliff stays visible in the run logs.
  const bool solved = solverStatus != -1 && !result.path.empty();
  RCLCPP_INFO(
      get_logger(),
      "RPV: %zu vertices (%zu dropped unreachable) -> %.1f ms, expanded=%ld%s%s",
      keep.size(), vertices.size() - keep.size(), solveMs, result.n_expanded,
      result.timeout ? "  TIMEOUT" : "",
      solved ? "" : "  NO SOLUTION"
  );

  if (!solved) {
    return false;
  }

  // result.path is in the renumbered space; map it back to original indices so
  // publishPath can index the caller's untouched `vertices`.
  pathIndices.clear();
  pathIndices.reserve(result.path.size());
  for (const long id : result.path) {
    if (id >= 0 && id < static_cast<long>(keep.size())) {
      pathIndices.push_back(static_cast<long>(keep[id]));
    }
  }
  return true;
}

void GlobalPlanner::publishClearanceDroppedVertices(
    const std::vector<Vertex>& vertices,
    const std::vector<std::vector<double>>& distanceMatrix,
    const nav_msgs::msg::OccupancyGrid& map
) const {
  // Collect vertices whose robot→frontier edge was rejected by the
  // narrow-passage clearance gate (aStarSearch returned -2.0, not -1.0
  // for unreachable or -inf for other failures).
  std::vector<Point2D> dropped;
  for (size_t j = 1; j < vertices.size(); ++j) {
    if (distanceMatrix[0][j] == -2.0) {
      const auto [gx, gy] = worldToGrid(vertices[j], map);
      dropped.emplace_back(gx, gy, 0.0);
    }
  }

  clearanceDroppedPub_->publish(convertToPointCloud2(dropped, map));
}

void GlobalPlanner::publishPath(
    const PlanningContext& ctx, const std::vector<long>& pathIndices,
    const std::vector<Vertex>& vertices
) const {
  auto pathMsg = std::make_unique<nav_msgs::msg::Path>();
  pathMsg->header.stamp = now();
  pathMsg->header.frame_id = "map";

  geometry_msgs::msg::PoseStamped startPose;
  startPose.header = pathMsg->header;
  startPose.pose = currentPose_;
  pathMsg->poses.push_back(startPose);

  for (const long pathIndex : pathIndices) {
    if (pathIndex < 0 || pathIndex >= static_cast<long>(vertices.size())) {
      continue;
    }

    const Vertex& vertex = vertices[pathIndex];
    if (pathIndex == 0) {
      continue;
    }

    geometry_msgs::msg::PoseStamped waypointPose;
    waypointPose.header = pathMsg->header;
    waypointPose.pose.position.x = vertex.x;
    waypointPose.pose.position.y = vertex.y;
    waypointPose.pose.position.z = ctx.currentPose.position.z;
    waypointPose.pose.orientation.w = 1.0;
    pathMsg->poses.push_back(waypointPose);
  }

  solutionPathPublisher_->publish(std::move(pathMsg));
}

void GlobalPlanner::handlePlanningSuccess(const PlanningContext& ctx) {
  if (replanNeeded_) {
    total_planning_time_ms_ = (now() - replan_start_time_).seconds() * 1000.0;
  }
  replanNeeded_ = false;
  consecutiveLocalFailures_ = 0;
  {
    std::lock_guard<std::mutex> lock(replanStateMutex_);
    lastReplanTime_ = now();
    lastReplanWorldPos_ = ctx.currentPose.position;
  }
}

void GlobalPlanner::publishExplorationComplete(bool complete) {
  if (!explorationCompletePub_) {
    return;
  }
  std_msgs::msg::Bool msg;
  msg.data = complete;
  explorationCompletePub_->publish(msg);
}

void GlobalPlanner::handleLocalPlanningFailure(
    const PlanningContext& ctx, const std::string& reason
) {
  ++consecutiveLocalFailures_;
  RCLCPP_WARN(
      get_logger(),
      "Local planning failed: %s (%d/%d)",
      reason.c_str(),
      consecutiveLocalFailures_,
      globalPlanningFailureThreshold_
  );
  publishPath(ctx, {}, {});

  if (consecutiveLocalFailures_ < globalPlanningFailureThreshold_) {
    return;
  }

  consecutiveLocalFailures_ = 0;
  runGlobalPlanning(ctx);
}
