#ifndef GLOBAL_PLANNER_GLOBAL_PLANNER_H
#define GLOBAL_PLANNER_GLOBAL_PLANNER_H

#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "prob_msgs/msg/prob_grid.hpp"
#include "prob_msgs/msg/prob_point.hpp"
#include "prob_msgs/msg/prob_point_vec.hpp"
#include "rpv_msgs/msg/planning_stats.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using OccupancyGridSubscriberPtr = rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr;
using ProbGridSubscriberPtr      = rclcpp::Subscription<prob_msgs::msg::ProbGrid>::SharedPtr;
using OdomSubscriberPtr          = rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr;
using ReplanSubscriberPtr        = rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr;
using BoolSubscriberPtr          = rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr;
using PointCloud2PublisherPtr    = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr;
using NavPathPublisherPtr        = rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr;
using ProbPointVecPublisherPtr   = rclcpp::Publisher<prob_msgs::msg::ProbPointVec>::SharedPtr;
using PlanningStatsPublisherPtr  = rclcpp::Publisher<rpv_msgs::msg::PlanningStats>::SharedPtr;
using PlannerRuntimePublisherPtr = rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr;

struct Point2D {
  int x{0};
  int y{0};
  double prob{0.0};

  Point2D() = default;
  Point2D(int x_value, int y_value, double prob_value = 0.0)
      : x(x_value), y(y_value), prob(prob_value) {}

  bool operator<(const Point2D &other) const {
    return (x != other.x) ? (x < other.x) : (y < other.y);
  }
};

struct Vertex {
  int id{0};
  double x{0.0};
  double y{0.0};
  double probability{0.0};
};

// How predicted-space candidate cells become planning vertices (see
// getClusteredPredictedFrontiers).
enum class PredictedFrontierMode {
  Poisson,  // Poisson-disk thinning at global_cluster_size
  Region,   // one vertex per connected predicted region (groupPredictedRegions)
  Tile,     // one vertex per world-anchored tile, TARE-style (tilePredictedFrontiers)
};

// A world-anchored square tile of the predicted-frontier lattice.
struct PredictedTile {
  int kx{0};
  int ky{0};

  bool operator<(const PredictedTile &other) const {
    return (kx != other.kx) ? (kx < other.kx) : (ky < other.ky);
  }
};

struct FrontierTask {
  prob_msgs::msg::ProbPoint msgPoint{};
  Vertex vertex{};
  double pathDistance{0.0};
};

struct ReachabilityData {
  std::vector<char> visible{};
  std::vector<char> predicted{};
  std::vector<int> visibleDistance{};
  std::vector<int> predictedDistance{};
  // Cells closed by the obstacle inflation (reachability_inflation_cells),
  // outside the robot's escape zone. Shared by the BFS and A* so both agree on
  // what is traversable. Empty when inflation is off.
  std::vector<char> blocked{};
  // Flat index of the robot's cell; -1 when it lies off the grid. Set even if
  // that cell is not free (the BFS then seeds nothing).
  int robotIndex{-1};
};

struct PlanningContext {
  ReachabilityData reachability{};
  std::vector<Point2D> localFrontiers{};
  prob_msgs::msg::ProbPointVec globalFrontierMsg{};
  geometry_msgs::msg::Pose currentPose{};
  nav_msgs::msg::OccupancyGrid originalVisibleMap{};
  nav_msgs::msg::OccupancyGrid visibleMap{};
  nav_msgs::msg::OccupancyGrid predictedMap{};
  prob_msgs::msg::ProbGrid confidenceMap{};
  Point2D robotMapPos{};
  // Per-cell distance (in cells) to the nearest occupied obstacle, in the
  // padded predicted-grid frame. Empty when the min-clearance gate is off.
  std::vector<float> clearanceMap{};
  // Effective minimum clearance (cells) a traversable cell must have. 0 = gate
  // disabled. Relaxed toward the robot's own clearance so it can escape a tight
  // pocket it already occupies.
  float minClearanceCells{0.0f};
};

class FindNearest {
public:
  // Poisson-disk thinning of frontier cells: accept candidates in input order,
  // keeping one only if it lies farther than radiusCells from every already-
  // accepted point, so the survivors are pairwise > radiusCells apart.
  // minMarginalGain is a noise gate — a candidate with fewer than that many raw
  // frontiers within radiusCells (itself included) is dropped, so isolated
  // specks on jagged boundaries do not survive. See the implementation in
  // global_planner_frontiers.cpp for the O(N) spatial-hash details.
  static std::vector<Point2D> downsampleFrontiers(const std::vector<Point2D> &rawFrontiers,
                                                  int radiusCells, int minMarginalGain);
};

class GlobalPlanner : public rclcpp::Node {
public:
  GlobalPlanner();
  ~GlobalPlanner();

private:
  void declareParameters();
  void loadParameters();
  void initializeSubscribers();
  void initializePublishers();
  void initializeTimers();

  void visibleCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void predictedCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void confidenceCallback(const prob_msgs::msg::ProbGrid::SharedPtr msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void replanCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void explorationEnabledCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void checkerCallback();
  void dispatchPlanning();
  void backgroundPlanningLoop();

  bool mapsReady() const;
  ReachabilityData computeReachability(
      const nav_msgs::msg::OccupancyGrid &visible,
      const nav_msgs::msg::OccupancyGrid &predicted,
      const Point2D &robotMapPos) const;
  std::vector<double> buildReachableDistanceMap(
      const nav_msgs::msg::OccupancyGrid &visible,
      const Point2D &robotMapPos,
      const ReachabilityData &reachability) const;
  std::vector<float> computeObstacleClearance(
      const nav_msgs::msg::OccupancyGrid &visible) const;

  std::vector<Point2D> findMapEdgeFrontiers(
      const nav_msgs::msg::OccupancyGrid &visible,
      const Point2D &robotMapPos,
      double detectionRange) const;
  std::vector<Point2D> findPredictedSpaceFrontiers(
      const nav_msgs::msg::OccupancyGrid &predicted,
      const nav_msgs::msg::OccupancyGrid &visible,
      const prob_msgs::msg::ProbGrid &confidence,
      const Point2D &robotMapPos,
      double detectionRange) const;
  std::vector<Point2D> getClusteredMapEdgeFrontiers(
      const nav_msgs::msg::OccupancyGrid &visible,
      const nav_msgs::msg::OccupancyGrid &predicted,
      const Point2D &robotMapPos,
      const ReachabilityData &reachability) const;
  std::vector<Point2D> getClusteredPredictedFrontiers(
      const nav_msgs::msg::OccupancyGrid &predicted,
      const nav_msgs::msg::OccupancyGrid &visible,
      const prob_msgs::msg::ProbGrid &confidence,
      const Point2D &robotMapPos,
      const ReachabilityData &reachability) const;

  bool isPointInFreeSpace(
      const nav_msgs::msg::OccupancyGrid &visible,
      const nav_msgs::msg::OccupancyGrid &predicted,
      double worldX, double worldY) const;
  bool checkNeighborSafety(
      const Point2D &point,
      const nav_msgs::msg::OccupancyGrid &map) const;
  bool isLocalFrontierTraversable(
      const nav_msgs::msg::OccupancyGrid &visible,
      const nav_msgs::msg::OccupancyGrid &predicted,
      const Point2D &point,
      bool useVisibleRegion,
      const ReachabilityData &reachability) const;

  // Returns {localFrontiers, globalFrontierMsg}; also publishes for visualization.
  std::pair<std::vector<Point2D>, prob_msgs::msg::ProbPointVec> classifyAndPublishFrontiers(
      const nav_msgs::msg::OccupancyGrid &visible,
      const nav_msgs::msg::OccupancyGrid &predicted,
      const std::vector<Point2D> &mapEdges,
      const std::vector<Point2D> &predictedFrontiers,
      const ReachabilityData &reachability,
      const Point2D &robotMapPos);
  // Collapse predicted-space candidate cells into one vertex per connected
  // predicted region. findPredictedSpaceFrontiers accepts every predicted-free
  // cell the robot hasn't observed — including the whole interior of a
  // predicted room — so Poisson thinning yields a vertex count proportional to
  // predicted *area*. The RPV solver is exponential in vertex count, so a large
  // predicted region alone can push it past its time limit. Grouping makes the
  // count proportional to the number of distinct predicted *places* instead,
  // which is also what the "probability this region exists" weight is about.
  //
  // Each region contributes its entrance (the member cell the robot reaches
  // first through predicted space) carrying the region's mean confidence.
  std::vector<Point2D> groupPredictedRegions(
      const std::vector<Point2D> &candidates,
      int width, int height,
      const ReachabilityData &reachability) const;
  // One vertex per tile holding enough candidate cells, TARE-style: the
  // candidates are the predicted-free cells the robot has not seen, so a tile
  // with many of them is unexplored predicted space. The vertex is the member
  // nearest the tile's center (the center itself may be a wall or unreachable)
  // and carries the members' mean confidence.
  std::vector<Point2D> tilePredictedFrontiers(
      const std::vector<Point2D> &candidates,
      const nav_msgs::msg::OccupancyGrid &map) const;
  // Tile holding cell (x, y) of `map`.
  PredictedTile predictedTileOf(
      int x, int y, const nav_msgs::msg::OccupancyGrid &map) const;
  // Tile outlines on /predicted_tiles: tiles whose vertex is in the local RPV
  // set, tiles left to the global pool, and tiles suppressed by a map edge.
  void publishPredictedTiles(
      const std::vector<PredictedTile> &local,
      const std::vector<PredictedTile> &pool,
      const std::vector<PredictedTile> &suppressed) const;
  std::vector<Vertex> prepareLocalPlanningVertices(
      const PlanningContext &ctx) const;
  std::vector<FrontierTask> collectGlobalPlanningTasks(
      const PlanningContext &ctx) const;
  std::vector<Vertex> buildGlobalPlanningVertices(
      const PlanningContext &ctx,
      const std::vector<FrontierTask> &tasks) const;

  bool runLocalPlanning(const PlanningContext &ctx, const std::vector<Vertex> &vertices);
  bool runGlobalPlanning(const PlanningContext &ctx);
  // Refresh the cached global guide: solve the coarse tour over the global
  // frontiers and keep its first stop. Runs on its own (slower) cadence — the
  // guide only has to say which way the robot leaves the local horizon, which
  // changes far less often than the local ordering does. Leaves globalGuide_
  // empty when no guide is available.
  void updateGlobalGuide(const PlanningContext &ctx);
  // Append the cached global guide to a local vertex list as the boundary
  // condition for the local solve (TARE's global-path-into-local-TSP coupling).
  // The guide sits outside the local horizon by construction, so the solver
  // orders the local frontiers around leaving toward it instead of solving the
  // local set in isolation. No-op when no guide is cached.
  void appendGlobalGuideVertex(std::vector<Vertex> &vertices) const;
  // preferredTargetIndex: index into `vertices` of a previously committed
  // next-hop target whose edge from the robot vertex (0) should be discounted
  // by localHysteresisFactor_ before being fed to the RPV solver. -1 disables.
  bool solvePlanningProblem(
      const PlanningContext &ctx,
      const std::vector<Vertex> &vertices,
      std::vector<long> &pathIndices,
      int preferredTargetIndex = -1) const;
  int findCommittedTargetVertexIndex(
      const std::vector<Vertex> &vertices) const;
  std::vector<std::vector<double>>computeDistanceMatrix(
      const PlanningContext &ctx,
      const std::vector<Vertex> &vertices) const;
  double computeVertexPairDistance(
      const PlanningContext &ctx,
      const std::vector<Vertex> &vertices,
      int sourceIndex, int targetIndex,
      double robotYaw) const;
  // Returns path cost in grid steps (cardinalMoveCost/diagonalMoveCost units),
  // or -1.0 if the goal is unreachable. Does not reconstruct the path.
  // tightCellBudget >= 0 enables the narrow-passage gate: A* may cross cells
  // whose clearance is below ctx.minClearanceCells, but if the chosen path
  // enters more than tightCellBudget such cells the goal is reported
  // unreachable (-1). < 0 disables the gate (pure shortest path).
  double aStarSearch(
      const PlanningContext &ctx,
      int startX, int startY, int goalX, int goalY,
      int tightCellBudget = -1) const;

  sensor_msgs::msg::PointCloud2 convertToPointCloud2(
      const std::vector<Point2D> &frontiers,
      const nav_msgs::msg::OccupancyGrid &map) const;
  void publishClearanceDroppedVertices(
      const std::vector<Vertex> &vertices,
      const std::vector<std::vector<double>> &distanceMatrix,
      const nav_msgs::msg::OccupancyGrid &map) const;
  void publishPath(
      const PlanningContext &ctx,
      const std::vector<long> &pathIndices,
      const std::vector<Vertex> &vertices) const;
  void handlePlanningSuccess(const PlanningContext &ctx);
  void handleLocalPlanningFailure(const PlanningContext &ctx, const std::string &reason);
  // Publish the current exploration-complete state on /exploration_complete.
  void publishExplorationComplete(bool complete);

  nav_msgs::msg::OccupancyGrid padVisibleToPredicted(
      const nav_msgs::msg::OccupancyGrid &visible,
      const nav_msgs::msg::OccupancyGrid &predicted) const;
  std::pair<int, int> worldToGrid(
      double x, double y,
      const nav_msgs::msg::OccupancyGrid &map) const;
  std::pair<int, int> worldToGrid(
      const Vertex &vertex,
      const nav_msgs::msg::OccupancyGrid &map) const;
  double getYawFromQuaternion(
      const geometry_msgs::msg::Quaternion &quat) const;
  bool isIndexTraversable(const PlanningContext &ctx, int index) const;

  OccupancyGridSubscriberPtr visibleSub_{};
  OccupancyGridSubscriberPtr predictedSub_{};
  ProbGridSubscriberPtr confidenceSub_{};
  OdomSubscriberPtr odomSub_{};
  ReplanSubscriberPtr replanSub_{};
  BoolSubscriberPtr explorationEnabledSub_{};
  PlannerRuntimePublisherPtr plannerRuntimePub_{};

  PointCloud2PublisherPtr localFrontierPub_{};
  PointCloud2PublisherPtr globalFrontierPub_{};
  PointCloud2PublisherPtr mapEdgeFrontierPub_{};
  PointCloud2PublisherPtr predictedFrontierPub_{};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr predictedTilesPub_{};
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr paddedMapPub_{};
  ProbPointVecPublisherPtr globalFrontiersProbPub_{};
  PlanningStatsPublisherPtr planningStatsPub_{};
  NavPathPublisherPtr solutionPathPublisher_{};
  PointCloud2PublisherPtr clearanceDroppedPub_{};
  PointCloud2PublisherPtr globalGuidePub_{};
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr explorationCompletePub_{};

  std::thread planningThread_;
  std::mutex planningMutex_;
  std::condition_variable planningCv_;
  std::unique_ptr<PlanningContext> pendingPlanContext_{};
  std::atomic<bool> planningThreadActive_{false};
  bool killPlanningThread_{false};
  bool planRequested_{false};

  rclcpp::TimerBase::SharedPtr checkerTimer_{};

  nav_msgs::msg::OccupancyGrid visible_{};
  nav_msgs::msg::OccupancyGrid predicted_{};
  prob_msgs::msg::ProbGrid confidence_{};
  geometry_msgs::msg::Pose currentPose_{};
  Point2D robotMapPos_;

  bool visibleReceived_{false};
  bool predictedReceived_{false};
  bool confidenceReceived_{false};
  bool odomReceived_{false};
  bool explorationEnabled_{true};
  // Latched "exploration complete" state: set when no REAL (map-edge) frontiers
  // remain — the reachable observed/unknown boundary is fully explored, so any
  // remaining frontiers are only the model's predictions and do not justify
  // planning. Published on /exploration_complete (transient-local) on change.
  bool explorationComplete_{false};
  // Set by the executor thread on external triggers (replanCallback,
  // explorationEnabledCallback, checkerCallback's stale-window heuristic)
  // and cleared by the background planning thread in handlePlanningSuccess.
  // Atomic because of that cross-thread access pattern.
  std::atomic<bool> replanNeeded_{true};
  int consecutiveLocalFailures_{0};

  std::string explorationEnabledTopic_{};
  double checkerCycle_{0.0};
  double maxRangeOfLocalFrontiers_{0.0};
  double replanPathLengthThreshold_{0.0};
  double replanTimeThreshold_{0.0};
  double frontierDetectionRange_{0.0};
  int globalPlanningFailureThreshold_{1};
  int localClusterSize_{1};
  int globalClusterSize_{1};
  int localMinMarginalGain_{2};
  int globalMinMarginalGain_{2};
  double predictedConfidenceMultiplier_{1.0};
  double turnCostFactor_{0.0};
  double rangeThresholdGlobalFrontier_{0.0};
  int maxFrontiersToPlan_{20};
  // Cap on the local RPV set: the nearest (geodesic) this many local frontiers,
  // map-edge and predicted alike; the rest go to the global pool. 0 = no cap.
  int maxLocalFrontiersToPlan_{0};
  double planningTimeLimit_{1.0};
  int distanceMatrixWorkerCount_{0};
  int distanceMatrixParallelMinVertices_{8};
  int frontierUnknownNeighborThreshold_{2};
  int neighborSafetyRadius_{1};
  // Minimum corridor half-width (in cells): a cell closer than this to an
  // occupied obstacle counts as "tight". Tight cells are still traversable —
  // the narrow-passage gate is a budget on how many of them a robot→frontier
  // path may cross (clearanceTightCellBudget_), not a hard per-cell block, so a
  // short pinch stays passable while a long narrow tunnel is rejected. 0 = off.
  double minClearanceCells_{0.0};
  // Max length of a single uninterrupted run of tight cells (clearance <
  // minClearanceCells) a robot→frontier A* path may thread before the frontier
  // is dropped as "only reachable via a too-narrow passage". Measured as the
  // longest CONSECUTIVE run, not the total count, so a long path that merely
  // brushes walls now and then is kept (large regions behind a brief pinch stay
  // explorable) while a genuine long narrow tunnel is discarded.
  int clearanceTightCellBudget_{5};
  int occupiedCellThreshold_{50};
  int predictedTraversableThreshold_{65};
  double predictedOccupiedPenalty_{10.0};
  // A* step-cost inflation for cells whose traversability is only justified
  // by the prediction (visible map == unknown). Each such step contributes
  // baseCost * (1 + alpha) instead of baseCost, so a path that depends
  // heavily on predicted-only cells ends up with inflated total cost ≈
  // length * (1 + alpha * predicted_fraction). 0.0 = trust prediction
  // fully; ~0.5 = mild conservatism; > 1.0 = strong bias toward visible
  // routes. Prevents predicted "shortcuts" from flipping the RPV solver's
  // ordering of frontier visits when only some edges in the matrix
  // depend on prediction.
  double predictedEdgeInflationAlpha_{0.0};
  double cardinalMoveCost_{1.0};
  double diagonalMoveCost_{1.414};
  // Mirrors waypoint_bridge's prediction-aware A* gates. A cell is admissible
  // for traversal only if the prediction confidence is >= confidenceThreshold_
  // AND the visible map at that location is not > 0 (any occupancy probability
  // including padding). Together they prevent A* from routing through
  // low-confidence predictions or through visible-known obstacles even when
  // predictedMap claims they're free.
  double confidenceThreshold_{0.5};
  bool explorationEnabledInitial_{true};
  // Soft-commitment (TARE-style hysteresis): discount the edge from the
  // robot vertex to the previously committed next-hop target so the RPV
  // solver biases toward keeping the same first hop across replans. Cleared
  // on global planning success (intent has been replaced).
  double localHysteresisFactor_{1.0};
  double localHysteresisMatchRadius_{0.0};
  std::optional<std::pair<double, double>> lastCommittedTarget_{};
  mutable std::atomic<unsigned long long> softCommitmentHits_{0};

  // How predicted candidates become vertices (predicted_frontier_mode).
  PredictedFrontierMode predictedFrontierMode_{PredictedFrontierMode::Poisson};
  // Region mode: regions smaller than this are dropped as noise.
  int predictedRegionMinCells_{20};
  // Obstacles (visible, or predicted where the visible map is unknown) are
  // inflated by this many cells (square kernel) before the reachability BFS,
  // which may not enter inflated cells: a passage narrower than
  // 2 * inflation + 1 cells is not traversable. 0 disables. Inflated cells
  // within kReachabilityEscapeCells of the robot stay open so it can leave a
  // wall it is standing next to.
  int reachabilityInflationCells_{0};
  static constexpr int kReachabilityEscapeCells = 3;
  // Tile mode: tile edge (m) and the share of a tile's cells that must be
  // candidates for it to become a vertex.
  double predictedTileSizeM_{4.0};
  double predictedTileMinFraction_{0.1};
  // Minimum separation (cells) between a predicted frontier and a map-edge
  // one. The two pipelines are thinned independently, so without this they can
  // coincide. 0 disables.
  double crossSourceMinSeparationCells_{5.0};

  // Global guide: the first stop of the coarse tour over global frontiers,
  // carried into the local solve as its boundary condition. Refreshed every
  // globalGuidePeriod_ planning cycles (and whenever the cached guide dies)
  // rather than every cycle, since the coarse tour costs a second RPV solve
  // plus its distance matrix.
  bool globalGuideEnabled_{true};
  int globalGuidePeriod_{3};
  std::optional<Vertex> globalGuide_{};
  int cyclesSinceGlobalGuide_{0};

  // Snapshot of where/when the last successful plan was issued. Written by
  // handlePlanningSuccess (background thread), read by checkerCallback
  // (executor thread) — both behind replanStateMutex_. replan_start_time_
  // stays single-threaded (only the background thread touches it).
  std::mutex replanStateMutex_;
  rclcpp::Time lastReplanTime_{0, 0, RCL_ROS_TIME};
  geometry_msgs::msg::Point lastReplanWorldPos_{};
  rclcpp::Time replan_start_time_{0, 0, RCL_ROS_TIME};

  // Stats for RPV Rviz Plugin
  mutable std::atomic<double> compute_reachability_time_ms_{0.0};
  mutable std::atomic<double> cluster_frontiers_time_ms_{0.0};
  mutable std::atomic<double> build_distance_matrix_time_ms_{0.0};
  mutable std::atomic<double> local_plan_time_ms_{0.0};
  mutable std::atomic<double> global_plan_time_ms_{0.0};
  mutable std::atomic<double> total_planning_time_ms_{0.0};
  mutable std::atomic<bool>   time_exceeded_{false};
  mutable std::atomic<double> time_exceeded_current_{0.0};
  mutable std::atomic<bool>   path_distance_exceeded_{false};
  mutable std::atomic<double> path_distance_current_{0.0};

  static constexpr std::array<std::pair<int, int>, 8> neighborDirections_ = {
      {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}};
  static constexpr std::array<std::pair<int, int>, 4> bfsDirections_ = {
      {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
};

#endif
