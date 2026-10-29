// Dispatch loop + background planning thread + checker callback.
// These together orchestrate when planning runs:
//   - checkerCallback:       decides whether replanning is needed, dispatches
//   - dispatchPlanning:      snapshots maps and signals the background thread
//   - backgroundPlanningLoop: runs the heavy planning off the executor thread
// Split out of global_planner.cpp during the 2026-05 refactor.

#include "../include/global_planner.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

void GlobalPlanner::dispatchPlanning() {
  if (!mapsReady() || planningThreadActive_) {
    return;
  }

  // Snapshot the latest raw map data; the background thread computes frontiers
  // from this snapshot right before solving, ensuring the freshest possible input.
  auto ctx = std::make_unique<PlanningContext>();
  ctx->currentPose = currentPose_;
  ctx->originalVisibleMap = visible_;
  ctx->visibleMap = visible_;
  ctx->predictedMap = predicted_;
  ctx->confidenceMap = confidence_;
  ctx->robotMapPos = robotMapPos_;

  {
    std::lock_guard<std::mutex> lock(planningMutex_);
    pendingPlanContext_ = std::move(ctx);
    planRequested_ = true;
  }
  planningCv_.notify_one();
}

void GlobalPlanner::backgroundPlanningLoop() {
  while (true) {
    std::unique_ptr<PlanningContext> ctx;
    {
      std::unique_lock<std::mutex> lock(planningMutex_);
      planningCv_.wait(lock, [this]() {
        return killPlanningThread_ || planRequested_;
      });

      if (killPlanningThread_) {
        break;
      }

      ctx = std::move(pendingPlanContext_);
      planRequested_ = false;
      planningThreadActive_ = true;
    }

    if (!ctx) {
      planningThreadActive_ = false;
      continue;
    }

    replan_start_time_ = now();

    // Pad the visible map to the predicted map's dimensions so that cells outside
    // the original visible bounds are treated as unknown and generate predicted frontiers.
    ctx->visibleMap = padVisibleToPredicted(ctx->visibleMap, ctx->predictedMap);
    ctx->robotMapPos = {
        static_cast<int>((ctx->currentPose.position.x -
                          ctx->visibleMap.info.origin.position.x) /
                         ctx->visibleMap.info.resolution),
        static_cast<int>((ctx->currentPose.position.y -
                          ctx->visibleMap.info.origin.position.y) /
                         ctx->visibleMap.info.resolution)};
    paddedMapPub_->publish(ctx->visibleMap);

    // Build the obstacle-clearance map once per cycle so every traversability
    // gate (A* distance matrix, frontier reachability) shares one view. Relax
    // the requirement toward the robot's own clearance so it never traps itself
    // when it legitimately sits close to a wall. Off entirely when the param is
    // 0, leaving ctx.clearanceMap empty and the gate dormant.
    if (minClearanceCells_ > 0.0) {
      ctx->clearanceMap = computeObstacleClearance(ctx->visibleMap);
      const int width = static_cast<int>(ctx->visibleMap.info.width);
      const int robotIndex = ctx->robotMapPos.y * width + ctx->robotMapPos.x;
      float robotClearance = std::numeric_limits<float>::infinity();
      if (robotIndex >= 0 &&
          robotIndex < static_cast<int>(ctx->clearanceMap.size())) {
        robotClearance = ctx->clearanceMap[robotIndex];
      }
      ctx->minClearanceCells = std::min(static_cast<float>(minClearanceCells_), robotClearance);
    }

    // Compute frontiers from the map snapshot taken at dispatch time.
    // This runs immediately before planning so the frontier data is as fresh
    // as possible relative to the maps the solver will use.
    ctx->reachability = computeReachability(ctx->visibleMap, ctx->predictedMap, ctx->robotMapPos);

    // The two frontier extraction + downsample pipelines read disjoint inputs
    // (mapEdges scans the padded visible map, predicted scans the predicted
    // map + confidence) and write disjoint outputs, so we can run them in
    // parallel and overlap the heavier predicted-side scan with the map-edge
    // side. classifyAndPublishFrontiers afterward joins both results.
    auto t = now();
    auto futureMapEdges = std::async(std::launch::async, [this, &ctx]() {
      return getClusteredMapEdgeFrontiers(
          ctx->visibleMap, ctx->predictedMap, ctx->robotMapPos,
          ctx->reachability);
    });
    const auto clusteredPredicted = getClusteredPredictedFrontiers(
        ctx->predictedMap, ctx->visibleMap, ctx->confidenceMap,
        ctx->robotMapPos, ctx->reachability);
    const auto clusteredMapEdges = futureMapEdges.get();
    auto [localFrontiers, globalFrontierMsg] = classifyAndPublishFrontiers(
        ctx->visibleMap, ctx->predictedMap,
        clusteredMapEdges, clusteredPredicted, ctx->reachability,
        ctx->robotMapPos);
    cluster_frontiers_time_ms_ = (now() - t).seconds() * 1000.0;

    ctx->localFrontiers = std::move(localFrontiers);
    ctx->globalFrontierMsg = std::move(globalFrontierMsg);

    // Exploration-complete gate. clusteredMapEdges holds only the REAL
    // (observed-free/unknown boundary) frontiers; clusteredPredicted holds the
    // model's guesses at space beyond walls. When no map-edge frontier remains,
    // the reachable real map is fully explored — the predicted frontiers left do
    // not justify planning (they are usually unreachable / may not exist), so
    // halt instead of chasing them or spinning through the failure path.
    if (clusteredMapEdges.empty()) {
      publishPath(*ctx, {}, {});  // one-pose path: stop the follower where it is
      if (!explorationComplete_) {
        explorationComplete_ = true;
        RCLCPP_INFO(
            get_logger(),
            "Exploration complete: no map-edge frontiers remain "
            "(%zu predicted frontiers ignored); halting planning.",
            clusteredPredicted.size());
        publishExplorationComplete(true);
      }
      replanNeeded_ = false;
      planningThreadActive_ = false;
      continue;
    }
    if (explorationComplete_) {
      explorationComplete_ = false;
      RCLCPP_INFO(get_logger(),
                  "Map-edge frontiers present again; resuming exploration.");
      publishExplorationComplete(false);
    }

    // Refresh the coarse tour over the global frontiers (on its own cadence)
    // and hand its first stop to the local solve as a boundary condition. The
    // guide sits outside the local horizon, so the local ordering is now
    // computed around leaving toward it rather than in isolation — and when
    // the local set is empty the guide alone still yields a plan, instead of
    // dropping into the local-failure path.
    updateGlobalGuide(*ctx);
    if (globalGuide_ && globalGuidePub_) {
      const auto [guideX, guideY] =
          worldToGrid(globalGuide_->x, globalGuide_->y, ctx->visibleMap);
      globalGuidePub_->publish(
          convertToPointCloud2({Point2D(guideX, guideY, 0.0)}, ctx->visibleMap));
    }

    auto localVertices = prepareLocalPlanningVertices(*ctx);
    appendGlobalGuideVertex(localVertices);
    if (localVertices.size() < 2) {
      handleLocalPlanningFailure(*ctx, "No valid local frontiers");
      planningThreadActive_ = false;
      continue;
    }

    if (!runLocalPlanning(*ctx, localVertices)) {
      handleLocalPlanningFailure(*ctx, "Unable to solve the local planning graph");
    }

    planningThreadActive_ = false;
  }
}

void GlobalPlanner::checkerCallback() {
  if (!explorationEnabled_) {
    return;
  }

  if (!replanNeeded_) {
    rclcpp::Time lastTime;
    geometry_msgs::msg::Point lastPos;
    {
      std::lock_guard<std::mutex> lock(replanStateMutex_);
      lastTime = lastReplanTime_;
      lastPos = lastReplanWorldPos_;
    }
    const double elapsed_time = (now() - lastTime).seconds();
    const bool timeExceeded = elapsed_time > replanTimeThreshold_;

    const double replannedDistance = std::hypot(
        currentPose_.position.x - lastPos.x,
        currentPose_.position.y - lastPos.y
    );
    const bool pathDistanceExceeded = replannedDistance > replanPathLengthThreshold_;
    if (timeExceeded && pathDistanceExceeded) {
      replanNeeded_ = true;
    }

    time_exceeded_ = timeExceeded;
    time_exceeded_current_ = elapsed_time;
    path_distance_exceeded_ = pathDistanceExceeded;
    path_distance_current_ = replannedDistance;
  }

  if (replanNeeded_ && explorationEnabled_) {
    dispatchPlanning();
  }

  if (planningStatsPub_) {
    rpv_msgs::msg::PlanningStats msg;
    msg.header.stamp = now();
    msg.compute_reachability_time_ms = compute_reachability_time_ms_;
    msg.cluster_frontiers_time_ms = cluster_frontiers_time_ms_;
    msg.build_distance_matrix_time_ms = build_distance_matrix_time_ms_;
    msg.local_plan_time_ms = local_plan_time_ms_;
    msg.global_plan_time_ms = global_plan_time_ms_;
    msg.total_planning_time_ms = total_planning_time_ms_;
    msg.replan_needed = replanNeeded_;
    msg.time_exceeded_current = time_exceeded_current_;
    msg.time_exceeded_target = replanTimeThreshold_;
    msg.time_exceeded = time_exceeded_;
    msg.path_distance_current = path_distance_current_;
    msg.path_distance_target = replanPathLengthThreshold_;
    msg.path_distance_exceeded = path_distance_exceeded_;
    planningStatsPub_->publish(msg);
  }

  plannerRuntimePub_->publish(std_msgs::msg::Float32().set__data(static_cast<float>(total_planning_time_ms_ / 1000.0)));
}
