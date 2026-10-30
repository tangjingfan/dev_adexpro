#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <prob_msgs/msg/prob_grid.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <queue>
#include <vector>

struct Cell { int x, y; };

// Per-tick view of the maps + thresholds A* needs. Bundling lets the
// helper functions stay free of class state.
struct PlanContext {
  const nav_msgs::msg::OccupancyGrid* grid;       // base grid for A* (predicted if available, else visible)
  const prob_msgs::msg::ProbGrid*     confidence; // may be nullptr when predicted is unavailable
  // visibleBlocked[i] == true means the cell at predicted-grid index i lands
  // on a visible-map cell whose value is > 0 (any occupied probability,
  // including padding/inflation). Such cells are hard-blocked irrespective of
  // what the prediction says, so the bridge never plans through the padded
  // visible obstacle zone.  nullptr when visible_map hasn't been received yet
  // (then visibility gating is skipped).
  const std::vector<uint8_t>* visibleBlocked;
  // inflated[i] == true: within inflation_cells (square kernel) of an obstacle
  // (a blocked cell above, or grid >= occupiedThreshold). A* may not enter such
  // cells except near its start and goal — the same rule as global_planner's
  // reachability_inflation_cells, so the bridge never threads a gap the
  // planner treated as closed. nullptr when inflation is off.
  const std::vector<uint8_t>* inflated;
  int     occupiedThreshold;
  double  confidenceThreshold;
};

// A cell is traversable iff:
//   1. predicted_map says it's free (< occupied_threshold, not unknown);
//   2. confidence (if provided) is at or above confidence_threshold; and
//   3. visible_map at that location is NOT > 0 (must be 0=free or -1=unknown).
// Rule 3 is the "if visible > 0, never traverse" gate — even high-confidence
// prediction can't unlock cells the visible map has flagged as occupied, so
// padded_visible's inflation zone stays off-limits.
static inline bool isCellFree(const PlanContext& ctx, int index) {
  const int8_t v = ctx.grid->data[index];
  if (v >= static_cast<int8_t>(ctx.occupiedThreshold)) return false;  // known/predicted obstacle
  if (v < 0) return false;                                            // still unknown after prediction
  if (ctx.confidence != nullptr &&
      ctx.confidence->data[index] < static_cast<float>(ctx.confidenceThreshold)) {
    return false;
  }
  if (ctx.visibleBlocked != nullptr && (*ctx.visibleBlocked)[index]) return false;
  return true;
}

// Cells within this many cells (Chebyshev) of the A* start or goal ignore the
// inflation, so the robot can leave a wall and reach a waypoint beside one.
// Matches global_planner's kReachabilityEscapeCells.
constexpr int kInflationEscapeCells = 3;

// Square-kernel dilation of the obstacle cells (see PlanContext::inflated),
// as a row pass then a column pass.
static std::vector<uint8_t> computeInflatedMask(const PlanContext& ctx, int inflation) {
  const int W = static_cast<int>(ctx.grid->info.width);
  const int H = static_cast<int>(ctx.grid->info.height);
  std::vector<uint8_t> rows(W * H, 0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int i = y * W + x;
      const bool obstacle =
          ctx.grid->data[i] >= static_cast<int8_t>(ctx.occupiedThreshold) ||
          (ctx.visibleBlocked != nullptr && (*ctx.visibleBlocked)[i]);
      if (!obstacle) continue;
      for (int nx = std::max(0, x - inflation); nx <= std::min(W - 1, x + inflation); ++nx) {
        rows[y * W + nx] = 1;
      }
    }
  }
  std::vector<uint8_t> inflated(W * H, 0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (!rows[y * W + x]) continue;
      for (int ny = std::max(0, y - inflation); ny <= std::min(H - 1, y + inflation); ++ny) {
        inflated[ny * W + x] = 1;
      }
    }
  }
  return inflated;
}

static std::vector<float> computeClearanceMap(const PlanContext& ctx) {
  const int W = static_cast<int>(ctx.grid->info.width);
  const int H = static_cast<int>(ctx.grid->info.height);
  const int mapSize = W * H;

  std::vector<float> clearance(mapSize, std::numeric_limits<float>::infinity());

  using Entry = std::pair<float, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;

  // Seed every blocked cell with distance 0 (the wavefront origin). Anything
  // not free is treated as a clearance source — known obstacles, unknown
  // cells, and low-confidence predicted cells alike — so the planner stays
  // away from all three.
  for (int index = 0; index < mapSize; ++index) {
    if (!isCellFree(ctx, index)) {
      clearance[index] = 0.0f;
      open.push({0.0f, index});
    }
  }

  auto inBounds = [&](int x, int y) { return x >= 0 && x < W && y >= 0 && y < H; };
  auto toIdx    = [&](int x, int y) { return y * W + x; };

  constexpr std::array<std::pair<int,int>, 8> dirs = {{
    {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
  }};

  while (!open.empty()) {
    auto [dist, curIdx] = open.top(); open.pop();
    if (dist > clearance[curIdx]) continue;

    const int cx = curIdx % W;
    const int cy = curIdx / W;
    for (auto [dx, dy] : dirs) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (!inBounds(nx, ny)) continue;

      const float step = (dx != 0 && dy != 0) ? 1.414f : 1.0f;
      const float nd = dist + step;
      const int ni = toIdx(nx, ny);
      if (nd < clearance[ni]) {
        clearance[ni] = nd;
        open.push({nd, ni});
      }
    }
  }

  return clearance;
}

// Optional bias steering the *start* of the path toward the robot's recent
// travel direction (the direction it has actually been moving, which — for a
// platform that can drive both forwards and backwards — is independent of
// heading). Disabled when hasDirection is false or weight <= 0.
struct DirectionBias {
  bool   hasDirection;     // a valid travel direction is available
  double dirX, dirY;       // unit travel vector in world/grid axes
  double weight;           // penalty scale; <= 0 disables the bias
  double rangeCells;       // bias fades linearly to 0 over this many cells from start
};

// Returns the A* path (start → goal) as grid cells, or empty if unreachable.
//
// NOTE on corridor width: besides the *soft* clearance preference
// (preferredClearanceCells / clearancePenaltyWeight), the hard gate is the
// obstacle inflation (ctx.inflated, inflation_cells): with it on, this A*
// cannot take a narrow shortcut the global planner's BFS and A* treat as
// closed. With inflation_cells = 0 there is no hard minimum-width gate.
static std::vector<Cell> astarPath(
  const PlanContext& ctx,
  const Cell& start, const Cell& goal,
  int maxNodes,
  const std::vector<float>& clearanceMap,
  double preferredClearanceCells,
  double clearancePenaltyWeight,
  const DirectionBias& dirBias)
{
  const int W = static_cast<int>(ctx.grid->info.width);
  const int H = static_cast<int>(ctx.grid->info.height);

  auto inBounds = [&](int x, int y) { return x >= 0 && x < W && y >= 0 && y < H; };
  auto toIdx    = [&](int x, int y) { return y * W + x; };

  if (!inBounds(start.x, start.y) || !inBounds(goal.x, goal.y)) return {};
  if (!isCellFree(ctx, toIdx(goal.x, goal.y))) return {};

  const int startIdx = toIdx(start.x, start.y);
  const int goalIdx  = toIdx(goal.x,  goal.y);

  std::vector<float> g(W * H, std::numeric_limits<float>::max());
  std::vector<int>   parent(W * H, -1);
  g[startIdx] = 0.0f;

  auto h = [&](int x, int y) {
    return std::hypot(static_cast<float>(x - goal.x), static_cast<float>(y - goal.y));
  };

  using Entry = std::pair<float, int>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> open;
  open.push({h(start.x, start.y), startIdx});

  constexpr std::array<std::pair<int,int>, 8> dirs = {{
    {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
  }};

  int expanded = 0;
  bool found = false;
  while (!open.empty() && expanded < maxNodes) {
    auto [f, curIdx] = open.top(); open.pop();
    if (curIdx == goalIdx) { found = true; break; }
    ++expanded;

    const int cx = curIdx % W;
    const int cy = curIdx / W;
    for (auto [dx, dy] : dirs) {
      const int nx = cx + dx;
      const int ny = cy + dy;
      if (!inBounds(nx, ny)) continue;
      const int ni = toIdx(nx, ny);
      if (!isCellFree(ctx, ni)) continue;
      if (ctx.inflated != nullptr && (*ctx.inflated)[ni] &&
          std::max(std::abs(nx - start.x), std::abs(ny - start.y)) > kInflationEscapeCells &&
          std::max(std::abs(nx - goal.x), std::abs(ny - goal.y)) > kInflationEscapeCells) {
        continue;
      }
      // Forbid diagonal corner-cutting: a diagonal step squeezes between the
      // two orthogonally-adjacent cells, so require BOTH of them to be free.
      // Otherwise the center-to-center segment clips the corner of a hard-
      // blocked cell and the drawn path appears to cross an obstacle it never
      // actually entered.
      if (dx != 0 && dy != 0 &&
          (!isCellFree(ctx, toIdx(cx + dx, cy)) ||
           !isCellFree(ctx, toIdx(cx, cy + dy)))) {
        continue;
      }
      const float stepCost = (dx != 0 && dy != 0) ? 1.414f : 1.0f;
      float clearancePenalty = 0.0f;
      if (ni < static_cast<int>(clearanceMap.size()) &&
          clearanceMap[ni] < preferredClearanceCells) {
        clearancePenalty = static_cast<float>(
          clearancePenaltyWeight * (preferredClearanceCells - clearanceMap[ni]));
      }
      // Travel-direction bias: penalize early steps that diverge from the
      // robot's recent motion so the path commits to roughly the same heading
      // it was already going, instead of doubling back. Strongest at the start
      // cell and fading linearly to zero over rangeCells, so only the opening
      // of the path is shaped — the remainder is free to route optimally.
      float directionPenalty = 0.0f;
      if (dirBias.hasDirection && dirBias.weight > 0.0) {
        const float distFromStart = std::hypot(static_cast<float>(cx - start.x),
                                                static_cast<float>(cy - start.y));
        if (distFromStart < dirBias.rangeCells) {
          const float invLen = 1.0f / stepCost;        // step is 1 or √2 long
          const float dot = (dx * invLen) * static_cast<float>(dirBias.dirX) +
                            (dy * invLen) * static_cast<float>(dirBias.dirY);
          const float decay = 1.0f - distFromStart / static_cast<float>(dirBias.rangeCells);
          // (1 - dot) is 0 when perfectly aligned, 2 when reversed.
          directionPenalty = static_cast<float>(dirBias.weight) * decay * (1.0f - dot);
        }
      }
      const float ng = g[curIdx] + stepCost + clearancePenalty + directionPenalty;
      if (ng < g[ni]) {
        g[ni]      = ng;
        parent[ni] = curIdx;
        open.push({ng + h(nx, ny), ni});
      }
    }
  }

  if (expanded >= maxNodes) {
    RCLCPP_WARN(rclcpp::get_logger("WaypointBridge"),
      "A* search exceeded max nodes (%d > %d)", expanded, maxNodes);
  }

  if (!found) return {};

  std::vector<Cell> path;
  for (int cur = goalIdx; cur != startIdx; cur = parent[cur]) {
    path.push_back({cur % W, cur / W});
  }
  path.push_back(start);
  std::reverse(path.begin(), path.end());
  return path;
}

// ──────────────────────────────────────────────────────────────────────────────

class WaypointBridge : public rclcpp::Node {
public:
  WaypointBridge() : Node("waypoint_bridge") {
    declare_parameter("lookahead_distance", 3.0);
    declare_parameter("reach_threshold", 1.5);
    declare_parameter("occupied_threshold", 50);
    declare_parameter("astar_max_nodes", 5000);
    declare_parameter("inflation_cells", 0);
    declare_parameter("preferred_clearance_cells", 3.0);
    declare_parameter("clearance_penalty_weight", 1.0);
    declare_parameter("publish_rate", 5.0);
    declare_parameter("confidence_threshold", 0.5);
    declare_parameter("direction_bias_weight", 1.0);
    declare_parameter("direction_bias_range_cells", 5.0);
    declare_parameter("direction_min_dist", 0.2);

    lookaheadDistance_       = get_parameter("lookahead_distance").as_double();
    reachThreshold_          = get_parameter("reach_threshold").as_double();
    occupiedThreshold_       = get_parameter("occupied_threshold").as_int();
    astarMaxNodes_           = get_parameter("astar_max_nodes").as_int();
    inflationCells_          = std::max(0, static_cast<int>(get_parameter("inflation_cells").as_int()));
    preferredClearanceCells_ = get_parameter("preferred_clearance_cells").as_double();
    clearancePenaltyWeight_  = get_parameter("clearance_penalty_weight").as_double();
    confidenceThreshold_     = get_parameter("confidence_threshold").as_double();
    directionBiasWeight_     = get_parameter("direction_bias_weight").as_double();
    directionBiasRangeCells_ = get_parameter("direction_bias_range_cells").as_double();
    directionMinDist_        = get_parameter("direction_min_dist").as_double();
    const double rate        = get_parameter("publish_rate").as_double();

    pathSub_ = create_subscription<nav_msgs::msg::Path>(
      "/solution_path", 10,
      std::bind(&WaypointBridge::pathCallback, this, std::placeholders::_1));

    odomSub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/state_estimation", 10,
      std::bind(&WaypointBridge::odomCallback, this, std::placeholders::_1));

    visibleMapSub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/padded_visible_map", 10,
      std::bind(&WaypointBridge::visibleMapCallback, this, std::placeholders::_1));

    // Optional inputs. If map_prediction isn't running these stay empty and
    // the bridge falls back to visible-only behaviour (same as before).
    predictedMapSub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/predicted_map", 10,
      std::bind(&WaypointBridge::predictedMapCallback, this, std::placeholders::_1));

    confidenceMapSub_ = create_subscription<prob_msgs::msg::ProbGrid>(
      "/confidence_map", 10,
      std::bind(&WaypointBridge::confidenceMapCallback, this, std::placeholders::_1));

    waypointPub_ = create_publisher<geometry_msgs::msg::PointStamped>("/way_point", 10);
    astarPathPub_ = create_publisher<nav_msgs::msg::Path>("/astar_path", 5);

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate),
      std::bind(&WaypointBridge::timerCallback, this));

    RCLCPP_INFO(get_logger(),
      "Waypoint bridge started (lookahead=%.1fm, confidence_threshold=%.2f)",
      lookaheadDistance_, confidenceThreshold_);
  }

private:
  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.empty()) return;
    path_ = *msg;
    pathProgressIndex_ = 1;
    RCLCPP_INFO(get_logger(), "New solution path: %zu waypoints", path_.poses.size());
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    const double newX = msg->pose.pose.position.x;
    const double newY = msg->pose.pose.position.y;

    // Track travel direction from position deltas rather than orientation: the
    // robot can drive forwards or backwards, so heading doesn't tell us which
    // way it is actually moving. Only refresh once it has moved at least
    // directionMinDist_ since the last anchor, which filters odometry jitter
    // while standing still.
    if (odomReceived_) {
      const double dx = newX - lastDirPosX_;
      const double dy = newY - lastDirPosY_;
      const double dist = std::hypot(dx, dy);
      if (dist >= directionMinDist_) {
        movementDirX_ = dx / dist;
        movementDirY_ = dy / dist;
        hasMovementDir_ = true;
        lastDirPosX_ = newX;
        lastDirPosY_ = newY;
      }
    } else {
      lastDirPosX_ = newX;
      lastDirPosY_ = newY;
    }

    robotX_ = newX;
    robotY_ = newY;
    robotZ_ = msg->pose.pose.position.z;
    odomReceived_ = true;
  }

  void visibleMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    visibleMap_ = *msg;
    visibleMapReceived_ = true;
  }

  void predictedMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    predictedMap_ = *msg;
    predictedMapReceived_ = true;
  }

  void confidenceMapCallback(const prob_msgs::msg::ProbGrid::SharedPtr msg) {
    confidenceMap_ = *msg;
    confidenceMapReceived_ = true;
  }

  // Picks which grid A* runs on. Prefer predictedMap if it has been received
  // (it covers a strictly larger area than visibleMap and already embeds the
  // visible cells), and pair it with confidenceMap so cell traversability
  // can require high prediction confidence. Otherwise fall back to the older
  // visible-only behaviour for compatibility with mapex/greedy baselines.
  PlanContext buildContext() const {
    PlanContext ctx{};
    ctx.occupiedThreshold = occupiedThreshold_;
    ctx.confidenceThreshold = confidenceThreshold_;
    if (predictedMapReceived_) {
      ctx.grid = &predictedMap_;
      ctx.confidence = (confidenceMapReceived_ &&
                        confidenceMap_.data.size() == predictedMap_.data.size())
                           ? &confidenceMap_
                           : nullptr;
    } else {
      ctx.grid = &visibleMap_;
      ctx.confidence = nullptr;
    }
    return ctx;
  }

  bool worldToCell(double wx, double wy,
                   const nav_msgs::msg::OccupancyGrid& grid,
                   Cell& c) const {
    const auto& info = grid.info;
    c.x = static_cast<int>((wx - info.origin.position.x) / info.resolution);
    c.y = static_cast<int>((wy - info.origin.position.y) / info.resolution);
    return c.x >= 0 && c.x < static_cast<int>(info.width) &&
           c.y >= 0 && c.y < static_cast<int>(info.height);
  }

  void timerCallback() {
    if (!odomReceived_ || path_.poses.size() < 2) return;
    if (!predictedMapReceived_ && !visibleMapReceived_) return;

    PlanContext ctx = buildContext();
    const nav_msgs::msg::OccupancyGrid& activeGrid = *ctx.grid;

    // Pre-compute the visible-blocked mask in predicted-grid coordinates. Any
    // predicted-grid cell whose corresponding visible_map cell is > 0 (i.e.
    // padded-visible says "occupied or inflated") gets hard-blocked here, so
    // A* cannot route through visible obstacles even when prediction would
    // otherwise allow them. Done once per tick — O(W*H), negligible at the
    // 170x250-ish grids we see in practice.
    std::vector<uint8_t> visibleBlocked;
    if (visibleMapReceived_ && &activeGrid != &visibleMap_) {
      const int pW = static_cast<int>(activeGrid.info.width);
      const int pH = static_cast<int>(activeGrid.info.height);
      const double pRes = activeGrid.info.resolution;
      const double pOx = activeGrid.info.origin.position.x;
      const double pOy = activeGrid.info.origin.position.y;
      visibleBlocked.assign(pW * pH, 0);
      int blockedCount = 0;
      for (int py = 0; py < pH; ++py) {
        for (int px = 0; px < pW; ++px) {
          const double wx = pOx + (px + 0.5) * pRes;
          const double wy = pOy + (py + 0.5) * pRes;
          Cell vc;
          if (!worldToCell(wx, wy, visibleMap_, vc)) continue;
          const int vIdx =
              vc.y * static_cast<int>(visibleMap_.info.width) + vc.x;
          if (visibleMap_.data[vIdx] > 0) {
            visibleBlocked[py * pW + px] = 1;
            ++blockedCount;
          }
        }
      }
      ctx.visibleBlocked = &visibleBlocked;
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
        "visibleBlocked mask: %d/%d cells (%.1f%%) hard-blocked by /padded_visible_map > 0",
        blockedCount, pW * pH,
        100.0 * blockedCount / std::max(1, pW * pH));
    }

    // Surface which planning mode the bridge is in right now. Throttled so the
    // log isn't flooded by the timer firing at publish_rate.
    if (ctx.confidence != nullptr) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "A* on /predicted_map (%dx%d) gated by /confidence_map >= %.2f, "
        "+ visible>0 hard-block",
        static_cast<int>(activeGrid.info.width),
        static_cast<int>(activeGrid.info.height),
        confidenceThreshold_);
    } else if (predictedMapReceived_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "A* on /predicted_map (%dx%d), no /confidence_map yet — "
        "predicted-free cells admissible (visible>0 still blocked)",
        static_cast<int>(activeGrid.info.width),
        static_cast<int>(activeGrid.info.height));
    } else {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "A* on /padded_visible_map only (%dx%d) — /predicted_map not received",
        static_cast<int>(activeGrid.info.width),
        static_cast<int>(activeGrid.info.height));
    }

    // Follow the solution path in the order the planner produced it, resuming
    // from the last pose actually reached. poses[0] is the robot's own pose at
    // planning time (see global_planner publishPath), so index 1 is the first
    // real target — pathProgressIndex_ starts there for every new path.
    //
    // Deliberately NOT "jump to the pose nearest the robot": the planner's
    // ordering is a solved visiting sequence, and picking the nearest pose
    // instead breaks it in two ways. It silently skips a stop the planner put
    // first when that stop happens to lie behind the robot (the robot then
    // never returns to it, since progress only moves forward), and it lets the
    // target slide backwards onto a pose the robot just reached and drove away
    // from — which made the robot turn around at the reach ring and oscillate.
    int targetIndex = pathProgressIndex_;

    // On reaching a pose, move on to the next one. The last pose has no
    // successor, so once reached the target stays there.
    while (targetIndex < static_cast<int>(path_.poses.size()) - 1) {
      const auto& t = path_.poses[targetIndex].pose.position;
      if (std::hypot(robotX_ - t.x, robotY_ - t.y) < reachThreshold_) {
        ++targetIndex;
        RCLCPP_DEBUG(get_logger(), "Advancing to solution waypoint %d", targetIndex);
      } else {
        break;
      }
    }

    // Latch the arrival progress so the next tick resumes here. Only arrivals
    // are latched — the unreachable-skip below stays local to this tick, since
    // unreachability is usually a transient map artefact and shouldn't retire a
    // waypoint forever.
    pathProgressIndex_ = targetIndex;

    Cell robotCell;
    if (!worldToCell(robotX_, robotY_, activeGrid, robotCell)) {
      RCLCPP_WARN(get_logger(), "Robot is out of map bounds");
      return;
    }

    const DirectionBias dirBias{
      hasMovementDir_, movementDirX_, movementDirY_,
      directionBiasWeight_, directionBiasRangeCells_};

    std::vector<uint8_t> inflated;
    if (inflationCells_ > 0) {
      inflated = computeInflatedMask(ctx, inflationCells_);
      ctx.inflated = &inflated;
    }
    const auto clearanceMap = computeClearanceMap(ctx);
    std::vector<Cell> astar;
    while (targetIndex < static_cast<int>(path_.poses.size())) {
      const auto& frontierPos = path_.poses[targetIndex].pose.position;
      Cell frontierCell;
      if (worldToCell(frontierPos.x, frontierPos.y, activeGrid, frontierCell)) {
        astar = astarPath(ctx, robotCell, frontierCell,
                          astarMaxNodes_, clearanceMap,
                          preferredClearanceCells_,
                          clearancePenaltyWeight_,
                          dirBias);
        if (!astar.empty()) break;
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Solution waypoint %d unreachable, skipping", targetIndex);
      ++targetIndex;
    }

    if (astar.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "No reachable waypoint in solution path");
      return;
    }

    // Build a Path message + classify each cell against the visible map to
    // surface whether prediction is actually contributing. A cell is
    // "prediction_only" if A* walked through it but the visible map still
    // reports unknown (or it falls outside the visible bounds entirely).
    const double res = activeGrid.info.resolution;
    {
      nav_msgs::msg::Path astarMsg;
      astarMsg.header.stamp = now();
      astarMsg.header.frame_id = activeGrid.header.frame_id.empty()
                                     ? std::string("map")
                                     : activeGrid.header.frame_id;
      astarMsg.poses.reserve(astar.size());

      int visibleKnown = 0;
      int predictionOnly = 0;
      int other = 0;
      for (const auto& c : astar) {
        const double wx = activeGrid.info.origin.position.x + (c.x + 0.5) * res;
        const double wy = activeGrid.info.origin.position.y + (c.y + 0.5) * res;

        geometry_msgs::msg::PoseStamped ps;
        ps.header = astarMsg.header;
        ps.pose.position.x = wx;
        ps.pose.position.y = wy;
        ps.pose.position.z = robotZ_;
        ps.pose.orientation.w = 1.0;
        astarMsg.poses.push_back(ps);

        if (!visibleMapReceived_) { ++other; continue; }
        Cell vc;
        if (!worldToCell(wx, wy, visibleMap_, vc)) {
          // Outside the visible map bounds — only prediction could have admitted
          // this cell.
          ++predictionOnly;
          continue;
        }
        const int vIdx = vc.y * static_cast<int>(visibleMap_.info.width) + vc.x;
        const int8_t v = visibleMap_.data[vIdx];
        if (v >= 0 && v < static_cast<int8_t>(occupiedThreshold_)) {
          ++visibleKnown;
        } else if (v < 0) {
          // Visible says unknown but A* walked through here -> prediction did it.
          ++predictionOnly;
        } else {
          ++other;  // visible says occupied yet A* used it -> map disagreement
        }
      }

      astarPathPub_->publish(astarMsg);

      const double ratio = astar.empty()
          ? 0.0
          : 100.0 * static_cast<double>(predictionOnly) /
                static_cast<double>(astar.size());
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "A* path: %zu cells (visible_known=%d, prediction_only=%d, other=%d, "
        "prediction_share=%.1f%%)",
        astar.size(), visibleKnown, predictionOnly, other, ratio);
    }

    // Walk forward along the A* path until we've accumulated lookaheadDistance_.
    double accumulated = 0.0;
    int chosen = static_cast<int>(astar.size()) - 1;

    for (int i = 1; i < static_cast<int>(astar.size()); ++i) {
      const double dx = (astar[i].x - astar[i - 1].x) * res;
      const double dy = (astar[i].y - astar[i - 1].y) * res;
      accumulated += std::hypot(dx, dy);
      if (accumulated >= lookaheadDistance_) {
        chosen = i;
        break;
      }
    }

    const auto& info = activeGrid.info;
    const double wx = info.origin.position.x + (astar[chosen].x + 0.5) * res;
    const double wy = info.origin.position.y + (astar[chosen].y + 0.5) * res;

    geometry_msgs::msg::PointStamped wpMsg;
    wpMsg.header.stamp    = now();
    wpMsg.header.frame_id = "map";
    wpMsg.point.x = wx;
    wpMsg.point.y = wy;
    wpMsg.point.z = robotZ_;
    waypointPub_->publish(wpMsg);
  }

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr          pathSub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odomSub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibleMapSub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr predictedMapSub_;
  rclcpp::Subscription<prob_msgs::msg::ProbGrid>::SharedPtr     confidenceMapSub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr waypointPub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               astarPathPub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Path          path_{};
  nav_msgs::msg::OccupancyGrid visibleMap_{};
  nav_msgs::msg::OccupancyGrid predictedMap_{};
  prob_msgs::msg::ProbGrid     confidenceMap_{};
  // Index of the solution-path pose currently being visited: 1 for a fresh path
  // (0 is the robot's own pose at planning time), stepped forward only on
  // arrival, reset in pathCallback. Monotonic within one path, so the target
  // never slides back onto a pose already reached.
  int    pathProgressIndex_{1};
  double robotX_{0.0}, robotY_{0.0}, robotZ_{0.0};
  double lookaheadDistance_{3.0};
  double reachThreshold_{1.5};
  int    occupiedThreshold_{50};
  int    astarMaxNodes_{5000};
  int    inflationCells_{0};
  double preferredClearanceCells_{3.0};
  double clearancePenaltyWeight_{1.0};
  double confidenceThreshold_{0.5};
  double directionBiasWeight_{1.0};
  double directionBiasRangeCells_{5.0};
  double directionMinDist_{0.2};
  // Robot travel-direction tracking (direction of motion, not heading — the
  // platform can drive both forwards and backwards).
  double movementDirX_{0.0}, movementDirY_{0.0};
  double lastDirPosX_{0.0}, lastDirPosY_{0.0};
  bool   hasMovementDir_{false};
  bool   odomReceived_{false};
  bool   visibleMapReceived_{false};
  bool   predictedMapReceived_{false};
  bool   confidenceMapReceived_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WaypointBridge>());
  rclcpp::shutdown();
  return 0;
}
