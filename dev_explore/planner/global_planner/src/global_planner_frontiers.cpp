// Frontier extraction, traversability filtering, classification, and the
// Poisson-disk thinning downsample (FindNearest::downsampleFrontiers).
// Split out of global_planner.cpp during the 2026-05 refactor; all method
// definitions still belong to the single GlobalPlanner class declared in
// include/global_planner.hpp.

#include "../include/global_planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<Point2D> GlobalPlanner::findMapEdgeFrontiers(
    const nav_msgs::msg::OccupancyGrid& visible,
    const Point2D& robotMapPos,
    double detectionRange
) const {
  std::vector<Point2D> frontiers;
  const int width = static_cast<int>(visible.info.width);
  const int height = static_cast<int>(visible.info.height);

  int minX = 0;
  int maxX = width - 1;
  int minY = 0;
  int maxY = height - 1;
  if (detectionRange > 0.0) {
    const int gridRange =
        static_cast<int>(detectionRange / visible.info.resolution);
    minX = std::max(0, robotMapPos.x - gridRange);
    maxX = std::min(width - 1, robotMapPos.x + gridRange);
    minY = std::max(0, robotMapPos.y - gridRange);
    maxY = std::min(height - 1, robotMapPos.y + gridRange);
  }

  const int threshold = frontierUnknownNeighborThreshold_;
  for (int y = minY; y <= maxY; ++y) {
    for (int x = minX; x <= maxX; ++x) {
      const int index = y * width + x;
      if (visible.data[index] != 0) {
        continue;
      }

      // Bail as soon as we hit the threshold — the exact count past that
      // doesn't change classification, and free cells in the interior visit
      // all 8 neighbors otherwise.
      int unknownNeighborCount = 0;
      for (int dy = -1; dy <= 1 && unknownNeighborCount < threshold; ++dy) {
        for (int dx = -1; dx <= 1 && unknownNeighborCount < threshold; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }

          const int neighborX = x + dx;
          const int neighborY = y + dy;
          if (neighborX < 0 || neighborX >= width || neighborY < 0 ||
              neighborY >= height) {
            continue;
          }

          const int neighborIndex = neighborY * width + neighborX;
          if (visible.data[neighborIndex] == -1) {
            ++unknownNeighborCount;
          }
        }
      }

      if (unknownNeighborCount >= threshold) {
        frontiers.emplace_back(x, y, 1.0);
      }
    }
  }

  return frontiers;
}

std::vector<Point2D> GlobalPlanner::findPredictedSpaceFrontiers(
    const nav_msgs::msg::OccupancyGrid& predicted,
    const nav_msgs::msg::OccupancyGrid& visible,
    const prob_msgs::msg::ProbGrid& confidence,
    const Point2D& robotMapPos,
    double detectionRange
) const {
  std::vector<Point2D> frontiers;
  const int width = static_cast<int>(predicted.info.width);
  const int height = static_cast<int>(predicted.info.height);

  int minX = 0;
  int maxX = width - 1;
  int minY = 0;
  int maxY = height - 1;
  if (detectionRange > 0.0) {
    const int gridRange =
        static_cast<int>(detectionRange / predicted.info.resolution);
    minX = std::max(0, robotMapPos.x - gridRange);
    maxX = std::min(width - 1, robotMapPos.x + gridRange);
    minY = std::max(0, robotMapPos.y - gridRange);
    maxY = std::min(height - 1, robotMapPos.y + gridRange);
  }

  const int visWidth  = static_cast<int>(visible.info.width);
  const int visHeight = static_cast<int>(visible.info.height);

  // pred/padded x,y = vis(x + expX, y + expY)
  const int expX = static_cast<int>(std::round(
      (visible.info.origin.position.x - predicted.info.origin.position.x) /
      predicted.info.resolution));
  const int expY = static_cast<int>(std::round(
      (visible.info.origin.position.y - predicted.info.origin.position.y) /
      predicted.info.resolution));

  for (int y = minY; y <= maxY; ++y) {
    for (int x = minX; x <= maxX; ++x) {
      const int index = y * width + x;
      if (predicted.data[index] != 0) {
        continue;
      }

      const int visX = x - expX;
      const int visY = y - expY;

      if (visX < 0 || visX >= visWidth || visY < 0 || visY >= visHeight) {
        continue;
      }

      if (visible.data[visY * visWidth + visX] != -1) {
        continue;
      }

      const int threshold = frontierUnknownNeighborThreshold_;
      int unknownNeighborCount = 0;
      for (int dy = -1; dy <= 1 && unknownNeighborCount < threshold; ++dy) {
        for (int dx = -1; dx <= 1 && unknownNeighborCount < threshold; ++dx) {
          if (dx == 0 && dy == 0) continue;
          const int nx = visX + dx;
          const int ny = visY + dy;
          if (nx < 0 || nx >= visWidth || ny < 0 || ny >= visHeight) {
            ++unknownNeighborCount;
          } else if (visible.data[ny * visWidth + nx] == -1) {
            ++unknownNeighborCount;
          }
        }
      }
      if (unknownNeighborCount < threshold) {
        continue;
      }

      const double confidenceValue =
          predictedConfidenceMultiplier_ * confidence.data[index];
      if (confidenceValue > 0.0) {
        frontiers.emplace_back(visX, visY, confidenceValue);
      }
    }
  }

  return frontiers;
}

bool GlobalPlanner::checkNeighborSafety(
    const Point2D& point, const nav_msgs::msg::OccupancyGrid& map
) const {
  for (int dy = -neighborSafetyRadius_; dy <= neighborSafetyRadius_; ++dy) {
    for (int dx = -neighborSafetyRadius_; dx <= neighborSafetyRadius_; ++dx) {
      const int neighborX = point.x + dx;
      const int neighborY = point.y + dy;
      if (neighborX < 0 || neighborX >= static_cast<int>(map.info.width) ||
          neighborY < 0 || neighborY >= static_cast<int>(map.info.height)) {
        continue;
      }

      const int neighborIndex = neighborY * map.info.width + neighborX;
      if (map.data[neighborIndex] > 0) {
        return false;
      }
    }
  }

  return true;
}

bool GlobalPlanner::isLocalFrontierTraversable(
    const nav_msgs::msg::OccupancyGrid& visible,
    const nav_msgs::msg::OccupancyGrid& predicted,
    const Point2D& point, bool useVisibleRegion,
    const ReachabilityData& reachability
) const {
  const int width = static_cast<int>(visible.info.width);
  const int height = static_cast<int>(visible.info.height);
  if (point.x < 0 || point.x >= width || point.y < 0 || point.y >= height) {
    return false;
  }

  const nav_msgs::msg::OccupancyGrid& map = useVisibleRegion ? visible : predicted;
  if (!checkNeighborSafety(point, map)) {
    return false;
  }

  const int index = point.y * width + point.x;
  return useVisibleRegion ? static_cast<bool>(reachability.visible[index])
                          : static_cast<bool>(reachability.predicted[index]);
}

std::vector<Point2D> GlobalPlanner::getClusteredMapEdgeFrontiers(
    const nav_msgs::msg::OccupancyGrid& visible,
    const nav_msgs::msg::OccupancyGrid& predicted,
    const Point2D& robotMapPos,
    const ReachabilityData& reachability
) const {
  const auto rawFrontiers =
      findMapEdgeFrontiers(visible, robotMapPos, frontierDetectionRange_);
  std::vector<Point2D> traversableFrontiers;

  for (const auto& frontier : rawFrontiers) {
    if (isLocalFrontierTraversable(visible, predicted, frontier, true, reachability)) {
      traversableFrontiers.push_back(frontier);
    }
  }

  return FindNearest::downsampleFrontiers(
      traversableFrontiers, localClusterSize_, localMinMarginalGain_);
}

std::vector<Point2D> GlobalPlanner::groupPredictedRegions(
    const std::vector<Point2D>& candidates,
    int width, int height,
    const ReachabilityData& reachability
) const {
  std::vector<Point2D> regions;
  if (candidates.empty()) {
    return regions;
  }

  // Membership grid over the candidate set. Two candidates belong to the same
  // region when they are 8-adjacent, so a predicted room (whose interior is all
  // candidates) collapses to one region, and two rooms joined by a predicted
  // corridor collapse together — correctly, since reaching either means
  // travelling through the same predicted opening.
  std::vector<int> cellIndex(static_cast<size_t>(width) * height, -1);
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& c = candidates[i];
    if (c.x < 0 || c.x >= width || c.y < 0 || c.y >= height) continue;
    cellIndex[static_cast<size_t>(c.y) * width + c.x] = static_cast<int>(i);
  }

  std::vector<uint8_t> visited(candidates.size(), 0);
  std::vector<int> stack;
  constexpr int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  constexpr int dy8[8] = {0, 0, 1, -1, 1, -1, 1, -1};

  int dropped = 0;
  for (size_t seed = 0; seed < candidates.size(); ++seed) {
    if (visited[seed]) continue;

    stack.clear();
    stack.push_back(static_cast<int>(seed));
    visited[seed] = 1;

    double probSum = 0.0;
    int memberCount = 0;
    // Entrance = the member the robot reaches first through predicted space.
    // That is where the tour actually "arrives" at the region, which is the
    // arrival time the RPV objective weights, and it is stable across cycles
    // in a way a centroid of a growing region is not.
    int bestX = candidates[seed].x;
    int bestY = candidates[seed].y;
    int bestDistance = std::numeric_limits<int>::max();

    while (!stack.empty()) {
      const int current = stack.back();
      stack.pop_back();
      const auto& cell = candidates[current];

      probSum += cell.prob;
      ++memberCount;

      const size_t flat = static_cast<size_t>(cell.y) * width + cell.x;
      const int distance = (flat < reachability.predictedDistance.size())
                               ? reachability.predictedDistance[flat]
                               : -1;
      if (distance >= 0 && distance < bestDistance) {
        bestDistance = distance;
        bestX = cell.x;
        bestY = cell.y;
      }

      for (int d = 0; d < 8; ++d) {
        const int nx = cell.x + dx8[d];
        const int ny = cell.y + dy8[d];
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const int neighbor = cellIndex[static_cast<size_t>(ny) * width + nx];
        if (neighbor < 0 || visited[neighbor]) continue;
        visited[neighbor] = 1;
        stack.push_back(neighbor);
      }
    }

    // Specks on a jagged prediction boundary are noise, not places.
    if (memberCount < predictedRegionMinCells_) {
      ++dropped;
      continue;
    }

    // Mean confidence over the region as its existence probability: robust to
    // single-cell noise, unlike taking the entrance's own value (which is
    // systematically the most optimistic, confidence decaying with depth).
    regions.emplace_back(bestX, bestY, probSum / memberCount);
  }

  RCLCPP_INFO(
      get_logger(),
      "Predicted regions: %zu cells -> %zu regions (%d dropped below %d cells)",
      candidates.size(), regions.size(), dropped, predictedRegionMinCells_
  );
  return regions;
}

std::vector<Point2D> GlobalPlanner::getClusteredPredictedFrontiers(
    const nav_msgs::msg::OccupancyGrid& predicted,
    const nav_msgs::msg::OccupancyGrid& visible,
    const prob_msgs::msg::ProbGrid& confidence,
    const Point2D& robotMapPos,
    const ReachabilityData& reachability
) const {
  const auto rawFrontiers = findPredictedSpaceFrontiers(
      predicted, visible, confidence, robotMapPos, frontierDetectionRange_);
  std::vector<Point2D> traversableFrontiers;

  for (const auto& frontier : rawFrontiers) {
    if (isLocalFrontierTraversable(visible, predicted, frontier, false, reachability)) {
      traversableFrontiers.push_back(frontier);
    }
  }

  switch (predictedFrontierMode_) {
    case PredictedFrontierMode::Region:
      return groupPredictedRegions(
          traversableFrontiers,
          static_cast<int>(visible.info.width),
          static_cast<int>(visible.info.height),
          reachability);
    case PredictedFrontierMode::Tile:
      return tilePredictedFrontiers(traversableFrontiers, visible);
    case PredictedFrontierMode::Poisson:
      break;
  }
  return FindNearest::downsampleFrontiers(
      traversableFrontiers, globalClusterSize_, globalMinMarginalGain_);
}

PredictedTile GlobalPlanner::predictedTileOf(
    int x, int y, const nav_msgs::msg::OccupancyGrid& map
) const {
  // The lattice is anchored at the world origin, not the map's, so a tile
  // keeps its place (and its vertex stays put) as the map grows.
  const double resolution = map.info.resolution;
  const double worldX = map.info.origin.position.x + (x + 0.5) * resolution;
  const double worldY = map.info.origin.position.y + (y + 0.5) * resolution;
  return {static_cast<int>(std::floor(worldX / predictedTileSizeM_)),
          static_cast<int>(std::floor(worldY / predictedTileSizeM_))};
}

std::vector<Point2D> GlobalPlanner::tilePredictedFrontiers(
    const std::vector<Point2D>& candidates,
    const nav_msgs::msg::OccupancyGrid& map
) const {
  struct TileCells {
    int count{0};
    double probSum{0.0};
    double bestSq{std::numeric_limits<double>::max()};
    int bestX{0};
    int bestY{0};
  };
  std::map<PredictedTile, TileCells> tiles;

  const double resolution = map.info.resolution;
  const double originX = map.info.origin.position.x;
  const double originY = map.info.origin.position.y;
  const double size = predictedTileSizeM_;
  for (const auto& c : candidates) {
    const PredictedTile tile = predictedTileOf(c.x, c.y, map);
    const double dx = originX + (c.x + 0.5) * resolution - (tile.kx + 0.5) * size;
    const double dy = originY + (c.y + 0.5) * resolution - (tile.ky + 0.5) * size;
    auto& cells = tiles[tile];
    ++cells.count;
    cells.probSum += c.prob;
    if (dx * dx + dy * dy < cells.bestSq) {
      cells.bestSq = dx * dx + dy * dy;
      cells.bestX = c.x;
      cells.bestY = c.y;
    }
  }

  const double cellsPerSide = size / resolution;
  const double minCells = predictedTileMinFraction_ * cellsPerSide * cellsPerSide;
  std::vector<Point2D> vertices;
  int sparse = 0;
  for (const auto& [tile, cells] : tiles) {
    if (cells.count < minCells) {
      ++sparse;
      continue;
    }
    vertices.emplace_back(cells.bestX, cells.bestY, cells.probSum / cells.count);
  }

  RCLCPP_INFO(
      get_logger(),
      "Predicted tiles: %zu cells -> %zu tiles of %.1f m (%d below %.0f%% filled)",
      candidates.size(), vertices.size(), size, sparse, predictedTileMinFraction_ * 100.0);
  return vertices;
}

void GlobalPlanner::publishPredictedTiles(
    const std::vector<PredictedTile>& local,
    const std::vector<PredictedTile>& pool,
    const std::vector<PredictedTile>& suppressed
) const {
  visualization_msgs::msg::MarkerArray markers;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear);

  const auto stamp = now();
  const double size = predictedTileSizeM_;
  auto outlines = [&](const std::vector<PredictedTile>& tiles, const char* ns, double width,
                      double z, float r, float g, float b, float a) {
    if (tiles.empty()) return;
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.id = static_cast<int>(markers.markers.size());
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = width;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = a;
    for (const auto& tile : tiles) {
      const double x0 = tile.kx * size, y0 = tile.ky * size;
      const double xs[4] = {x0, x0 + size, x0 + size, x0};
      const double ys[4] = {y0, y0, y0 + size, y0 + size};
      for (int i = 0; i < 4; ++i) {
        geometry_msgs::msg::Point p, q;
        p.x = xs[i];
        p.y = ys[i];
        q.x = xs[(i + 1) % 4];
        q.y = ys[(i + 1) % 4];
        p.z = q.z = z;
        marker.points.push_back(p);
        marker.points.push_back(q);
      }
    }
    markers.markers.push_back(marker);
  };
  // Drawn bottom to top, so a local tile's outline wins on shared edges.
  outlines(suppressed, "suppressed_tiles", 0.08, 0.20, 0.6f, 0.6f, 0.6f, 0.6f);
  outlines(pool, "pool_tiles", 0.08, 0.21, 0.35f, 0.65f, 1.0f, 0.8f);
  outlines(local, "local_tiles", 0.20, 0.22, 0.05f, 0.30f, 0.85f, 1.0f);
  predictedTilesPub_->publish(markers);
}

bool GlobalPlanner::isPointInFreeSpace(
    const nav_msgs::msg::OccupancyGrid& visible,
    const nav_msgs::msg::OccupancyGrid& predicted,
    double worldX, double worldY
) const {
  const int gridX = static_cast<int>(
      (worldX - visible.info.origin.position.x) / visible.info.resolution
  );
  const int gridY = static_cast<int>(
      (worldY - visible.info.origin.position.y) / visible.info.resolution
  );

  if (gridX < 0 || gridX >= static_cast<int>(visible.info.width) || gridY < 0 ||
      gridY >= static_cast<int>(visible.info.height)) {
    return false;
  }

  const int index = gridY * visible.info.width + gridX;
  if (visible.data[index] == 0) {
    return true;
  }
  if (visible.data[index] != -1) {
    return false;
  }

  // Unknown in visible — check predicted map at the correct predicted-space index.
  const int expX = static_cast<int>(std::round(
      (visible.info.origin.position.x - predicted.info.origin.position.x) /
      visible.info.resolution));
  const int expY = static_cast<int>(std::round(
      (visible.info.origin.position.y - predicted.info.origin.position.y) /
      visible.info.resolution));
  const int predX = gridX + expX;
  const int predY = gridY + expY;
  const int predWidth = static_cast<int>(predicted.info.width);
  const int predHeight = static_cast<int>(predicted.info.height);
  if (predX < 0 || predX >= predWidth || predY < 0 || predY >= predHeight) {
    return false;
  }
  return predicted.data[predY * predWidth + predX] == 0;
}

std::pair<std::vector<Point2D>, prob_msgs::msg::ProbPointVec>
GlobalPlanner::classifyAndPublishFrontiers(
    const nav_msgs::msg::OccupancyGrid& visible,
    const nav_msgs::msg::OccupancyGrid& predicted,
    const std::vector<Point2D>& mapEdges,
    const std::vector<Point2D>& predictedFrontiers,
    const ReachabilityData& reachability,
    const Point2D& robotMapPos
) {
  std::vector<Point2D> localFrontiers;
  std::vector<Point2D> globalFrontiers;
  prob_msgs::msg::ProbPointVec globalFrontierMsg;

  const auto distanceMap =
      buildReachableDistanceMap(visible, robotMapPos, reachability);
  const int width = static_cast<int>(visible.info.width);
  const double originX = visible.info.origin.position.x;
  const double originY = visible.info.origin.position.y;
  const double resolution = visible.info.resolution;

  // A frontier within max_range_of_local_frontiers (geodesic, cells) is a
  // local candidate, a farther one goes to the global pool. Returns, per
  // frontier, whether it went local, went global, or was dropped as
  // unreachable. Local candidates are collected first and capped below.
  enum class Placement { Local, Global, Dropped };
  struct LocalCandidate {
    Point2D frontier;
    double distance;
    bool predicted;
    size_t index;  // into the classified vector
  };
  std::vector<LocalCandidate> localCandidates;
  auto addGlobal = [&](const Point2D& frontier, double distance) {
    globalFrontiers.push_back(frontier);
    prob_msgs::msg::ProbPoint frontierMsg;
    frontierMsg.point.point.x = originX + (frontier.x + 0.5) * resolution;
    frontierMsg.point.point.y = originY + (frontier.y + 0.5) * resolution;
    frontierMsg.prob = frontier.prob;
    frontierMsg.dist = distance;
    globalFrontierMsg.points.push_back(frontierMsg);
  };
  auto classify = [&](const std::vector<Point2D>& frontiers, bool predictedSource) {
    std::vector<Placement> placements;
    placements.reserve(frontiers.size());
    for (const auto& frontier : frontiers) {
      placements.push_back(Placement::Dropped);
      const int index = frontier.y * width + frontier.x;
      if (distanceMap[index] == std::numeric_limits<double>::max()) {
        continue;
      }

      const double worldX = originX + (frontier.x + 0.5) * resolution;
      const double worldY = originY + (frontier.y + 0.5) * resolution;
      if (!isPointInFreeSpace(visible, predicted, worldX, worldY)) {
        continue;
      }

      if (distanceMap[index] <= maxRangeOfLocalFrontiers_) {
        placements.back() = Placement::Local;
        localCandidates.push_back(
            {frontier, distanceMap[index], predictedSource, placements.size() - 1});
        continue;
      }

      placements.back() = Placement::Global;
      addGlobal(frontier, distanceMap[index]);
    }
    return placements;
  };

  // The two extraction pipelines are thinned independently — map edges at
  // local_cluster_size, predicted at global_cluster_size — so nothing stops a
  // predicted representative from landing right next to a map-edge one. They
  // then describe the same opening, cost two solver vertices, and the tour
  // gains nothing from visiting both. Drop the predicted copy: the map-edge
  // frontier is the one that is certain (p = 1).
  const auto squaredDistance = [](const Point2D& a, const Point2D& b) {
    const double dx = static_cast<double>(a.x - b.x);
    const double dy = static_cast<double>(a.y - b.y);
    return dx * dx + dy * dy;
  };
  std::vector<Point2D> separatedPredicted;
  int suppressed = 0;
  const bool tileMode = predictedFrontierMode_ == PredictedFrontierMode::Tile;
  std::vector<PredictedTile> suppressedTiles;
  if (tileMode) {
    // A tile that already holds a map-edge frontier is represented by it, so
    // the tile's predicted vertex is dropped.
    std::set<PredictedTile> mapEdgeTiles;
    for (const auto& mapEdge : mapEdges) {
      mapEdgeTiles.insert(predictedTileOf(mapEdge.x, mapEdge.y, visible));
    }
    for (const auto& predictedFrontier : predictedFrontiers) {
      const PredictedTile tile = predictedTileOf(predictedFrontier.x, predictedFrontier.y, visible);
      if (mapEdgeTiles.count(tile)) {
        ++suppressed;
        suppressedTiles.push_back(tile);
        continue;
      }
      separatedPredicted.push_back(predictedFrontier);
    }
  } else if (crossSourceMinSeparationCells_ > 0.0) {
    separatedPredicted.reserve(predictedFrontiers.size());
    const double crossSq = crossSourceMinSeparationCells_ * crossSourceMinSeparationCells_;
    for (const auto& predictedFrontier : predictedFrontiers) {
      const bool tooCloseToMapEdge = std::any_of(
          mapEdges.begin(), mapEdges.end(), [&](const Point2D& mapEdge) {
            return squaredDistance(mapEdge, predictedFrontier) <= crossSq;
          });
      if (tooCloseToMapEdge) {
        ++suppressed;
        continue;
      }
      separatedPredicted.push_back(predictedFrontier);
    }
  } else {
    separatedPredicted = predictedFrontiers;
  }

  // Closest-pair diagnostics on the *unsuppressed* sets, so the run log says
  // which pairing actually produces near-duplicates rather than leaving it to
  // be guessed from an RViz screenshot. Minima are taken over squared
  // distances; fmt applies the single sqrt (or -1 when there is no pair).
  auto closestWithinSq = [&](const std::vector<Point2D>& points) {
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < points.size(); ++i) {
      for (size_t j = i + 1; j < points.size(); ++j) {
        best = std::min(best, squaredDistance(points[i], points[j]));
      }
    }
    return best;
  };
  double closestCrossSq = std::numeric_limits<double>::max();
  for (const auto& mapEdge : mapEdges) {
    for (const auto& predictedFrontier : predictedFrontiers) {
      closestCrossSq = std::min(closestCrossSq, squaredDistance(mapEdge, predictedFrontier));
    }
  }
  auto fmt = [](double sq) {
    return sq == std::numeric_limits<double>::max() ? -1.0 : std::sqrt(sq);
  };
  const std::string suppressionRule =
      predictedFrontierMode_ == PredictedFrontierMode::Tile
          ? "sharing a tile with a map edge"
          : "within " + std::to_string(crossSourceMinSeparationCells_) + " cells";
  RCLCPP_INFO(
      get_logger(),
      "Closest pair (cells): map_edge<->map_edge=%.1f  predicted<->predicted=%.1f  "
      "map_edge<->predicted=%.1f  (%d predicted suppressed %s)",
      fmt(closestWithinSq(mapEdges)), fmt(closestWithinSq(predictedFrontiers)),
      fmt(closestCrossSq), suppressed, suppressionRule.c_str()
  );

  classify(mapEdges, false);
  const size_t mapEdgeGlobal = globalFrontiers.size();
  auto predictedPlacements = classify(separatedPredicted, true);

  // Local cap: map-edge and predicted candidates compete on geodesic distance
  // alone; the nearest max_local_frontiers_to_plan go to the RPV, the rest to
  // the global pool. The RPV is exponential in vertex count, and the robot
  // only executes the first hop before replanning, so ordering far-away
  // vertices exactly buys nothing.
  std::stable_sort(localCandidates.begin(), localCandidates.end(),
                   [](const LocalCandidate& a, const LocalCandidate& b) {
                     return a.distance < b.distance;
                   });
  size_t mapEdgeLocal = 0;
  size_t overflow = 0;
  for (size_t i = 0; i < localCandidates.size(); ++i) {
    const auto& c = localCandidates[i];
    if (maxLocalFrontiersToPlan_ > 0 && i >= static_cast<size_t>(maxLocalFrontiersToPlan_)) {
      ++overflow;
      addGlobal(c.frontier, c.distance);
      if (c.predicted) predictedPlacements[c.index] = Placement::Global;
      continue;
    }
    localFrontiers.push_back(c.frontier);
    if (!c.predicted) ++mapEdgeLocal;
  }

  if (tileMode) {
    std::vector<PredictedTile> localTiles;
    std::vector<PredictedTile> poolTiles;
    for (size_t i = 0; i < separatedPredicted.size(); ++i) {
      const PredictedTile tile =
          predictedTileOf(separatedPredicted[i].x, separatedPredicted[i].y, visible);
      (predictedPlacements[i] == Placement::Local ? localTiles : poolTiles).push_back(tile);
    }
    publishPredictedTiles(localTiles, poolTiles, suppressedTiles);
  }

  // Log the split so an overlong local list is visible before it shows up as
  // a solver timeout. global's map_edge count excludes local overflow.
  RCLCPP_INFO(
      get_logger(),
      "Frontiers: local=%zu (map_edge=%zu predicted=%zu, %zu more beyond the cap of %d)  "
      "global=%zu (beyond range: map_edge=%zu)",
      localFrontiers.size(), mapEdgeLocal, localFrontiers.size() - mapEdgeLocal, overflow,
      maxLocalFrontiersToPlan_, globalFrontiers.size(), mapEdgeGlobal
  );

  globalFrontierMsg.header.stamp = now();
  globalFrontierMsg.header.frame_id = "map";

  localFrontierPub_->publish(convertToPointCloud2(localFrontiers, visible));
  globalFrontierPub_->publish(convertToPointCloud2(globalFrontiers, visible));
  globalFrontiersProbPub_->publish(globalFrontierMsg);

  // the passed in visible is actually padded visible
  mapEdgeFrontierPub_->publish(convertToPointCloud2(mapEdges, visible));
  // Post-suppression set, so /predicted_frontiers shows what actually reaches
  // the solver rather than the pre-filter candidates.
  predictedFrontierPub_->publish(convertToPointCloud2(separatedPredicted, visible));

  return {std::move(localFrontiers), std::move(globalFrontierMsg)};
}

sensor_msgs::msg::PointCloud2 GlobalPlanner::convertToPointCloud2(
    const std::vector<Point2D>& frontiers, const nav_msgs::msg::OccupancyGrid& map
) const {
  sensor_msgs::msg::PointCloud2 cloudMsg;
  cloudMsg.header.stamp = now();
  cloudMsg.header.frame_id = map.header.frame_id;
  cloudMsg.height = 1;
  cloudMsg.width = static_cast<uint32_t>(frontiers.size());

  sensor_msgs::PointCloud2Modifier modifier(cloudMsg);
  modifier.setPointCloud2Fields(
      3,
      "x",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "y",
      1,
      sensor_msgs::msg::PointField::FLOAT32,
      "z",
      1,
      sensor_msgs::msg::PointField::FLOAT32
  );
  modifier.resize(frontiers.size());

  sensor_msgs::PointCloud2Iterator<float> xIterator(cloudMsg, "x");
  sensor_msgs::PointCloud2Iterator<float> yIterator(cloudMsg, "y");
  sensor_msgs::PointCloud2Iterator<float> zIterator(cloudMsg, "z");
  for (const auto& frontier : frontiers) {
    *xIterator =
        map.info.origin.position.x + (frontier.x + 0.5) * map.info.resolution;
    *yIterator =
        map.info.origin.position.y + (frontier.y + 0.5) * map.info.resolution;
    *zIterator = map.info.origin.position.z;
    ++xIterator;
    ++yIterator;
    ++zIterator;
  }

  return cloudMsg;
}

std::vector<Point2D> FindNearest::downsampleFrontiers(
    const std::vector<Point2D>& rawFrontiers,
    int radiusCells, int minMarginalGain
) {
  const int N = static_cast<int>(rawFrontiers.size());
  if (N == 0 || radiusCells <= 0) {
    return rawFrontiers;
  }

  const long long radiusSq =
      static_cast<long long>(radiusCells) * radiusCells;

  // 64-bit pack of any int32 bucket coords. Using uint32_t casts before the
  // shift makes the layout well-defined for negative bucket indices (which
  // arise when iterating the 3x3 neighborhood around bucket (0, 0)).
  auto packKey = [](int bx, int by) -> int64_t {
    return (static_cast<int64_t>(static_cast<uint32_t>(bx)) << 32) |
           static_cast<int64_t>(static_cast<uint32_t>(by));
  };

  // Poisson-disk thinning: scan candidates in input order and accept one only
  // if it lies farther than R from every already-accepted point. The result is
  // a maximal set of representatives pairwise > R apart — the same geometry the
  // old greedy set-cover produced, but without its O(N·k) all-pairs neighbor
  // precompute (the measured 600–3600ms cluster cost). Bucket side == R, so an
  // accepted point's exclusion disk is covered by its own bucket plus the 8
  // neighbors; accepted points are > R apart, so each bucket holds O(1) of them
  // and every accept-check is O(1). Total O(N).
  std::unordered_map<int64_t, std::vector<int>> acceptedBuckets;
  acceptedBuckets.reserve(static_cast<size_t>(N / 4 + 1));

  // Optional noise gate (minMarginalGain > 1): drop frontiers with fewer than
  // minMarginalGain raw neighbors within R (counting themselves) — preserving
  // the old min_marginal_gain "drop isolated specks" behaviour. The raw hash is
  // built and queried only when the gate is active, and the per-candidate count
  // runs only after the distance test passes (≈ the sparse accepted set), so it
  // stays O(N) overall.
  const bool applyNoiseGate = minMarginalGain > 1;
  std::unordered_map<int64_t, std::vector<int>> rawBuckets;
  if (applyNoiseGate) {
    rawBuckets.reserve(static_cast<size_t>(N / 2 + 1));
    for (int i = 0; i < N; ++i) {
      rawBuckets[packKey(rawFrontiers[i].x / radiusCells,
                         rawFrontiers[i].y / radiusCells)]
          .push_back(i);
    }
  }

  std::vector<Point2D> selected;
  selected.reserve(static_cast<size_t>(N / 4 + 1));

  for (int i = 0; i < N; ++i) {
    const Point2D& f = rawFrontiers[i];
    const int bx = f.x / radiusCells;
    const int by = f.y / radiusCells;

    // Reject if within R of an already-accepted point.
    bool tooClose = false;
    for (int dx = -1; dx <= 1 && !tooClose; ++dx) {
      for (int dy = -1; dy <= 1 && !tooClose; ++dy) {
        auto it = acceptedBuckets.find(packKey(bx + dx, by + dy));
        if (it == acceptedBuckets.end()) continue;
        for (int s : it->second) {
          const long long ddx = selected[s].x - f.x;
          const long long ddy = selected[s].y - f.y;
          if (ddx * ddx + ddy * ddy <= radiusSq) {
            tooClose = true;
            break;
          }
        }
      }
    }
    if (tooClose) continue;

    // Noise gate: count raw frontiers within R (including f) and drop f if that
    // falls below minMarginalGain.
    if (applyNoiseGate) {
      int rawNeighbors = 0;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          auto it = rawBuckets.find(packKey(bx + dx, by + dy));
          if (it == rawBuckets.end()) continue;
          for (int j : it->second) {
            const long long ddx = rawFrontiers[j].x - f.x;
            const long long ddy = rawFrontiers[j].y - f.y;
            if (ddx * ddx + ddy * ddy <= radiusSq) ++rawNeighbors;
          }
        }
      }
      if (rawNeighbors < minMarginalGain) continue;
    }

    acceptedBuckets[packKey(bx, by)].push_back(
        static_cast<int>(selected.size()));
    selected.push_back(f);
  }

  return selected;
}
