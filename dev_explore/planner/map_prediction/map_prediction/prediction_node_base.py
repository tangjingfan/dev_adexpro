"""Shared skeleton for the map-prediction nodes.

Every prediction backend serves the same ROS contract — it turns /visible_map +
/state_estimation into /predicted_map, /confidence_map and /all_frontier — and
differs only in how the frontier windows get filled in. This module owns
everything that belongs to the contract rather than to a backend: the common
parameters, the interfaces, the background-inference loop, frontier extraction
and the grid <-> numpy plumbing.

Kept torch-free so the oracle backend does not pull in the learned model.
"""

import array
import threading
import time
from copy import deepcopy

import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from prob_msgs.msg import ProbGrid
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from sklearn.cluster import DBSCAN
from std_msgs.msg import Header

from .grid_encoding import MODEL_TO_OCCUPANCY, OCCUPANCY_TO_MODEL, UNKNOWN_VALUE


class PredictionNodeBase(Node):
    """Template for a prediction backend.

    A subclass must implement `_build_prediction_outputs`. It may also extend
    `_declare_parameters` / `_load_parameters` / `_create_interfaces` (call
    super() first), gate the cycle on extra inputs via `_inputs_ready`, and carry
    per-cycle state through the `_begin_cycle` / `_end_cycle` pair.
    """

    #: Prefix of the per-cycle timing log, so the backends stay distinguishable.
    inference_label = "Inference"

    def __init__(self, node_name):
        super().__init__(node_name)
        self._declare_parameters()
        self._load_parameters()
        self._create_interfaces()

        self.visible_map = None
        self.current_odom = None

        self._inference_thread = None

    # ------------------------------------------------------------------
    # Parameters and interfaces
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        self.declare_parameter("prediction_cycle", 2.0)
        self.declare_parameter("model_input_size", 128)
        self.declare_parameter("enable_prediction", True)
        self.declare_parameter("frontier_cluster_eps", 0.2)
        self.declare_parameter("frontier_cluster_min_samples", 1)
        self.declare_parameter("frontier_kernel_size", 3)
        self.declare_parameter("prediction_expansion_cells", 50)
        self.declare_parameter("max_frontier_distance", 20.0)
        self.declare_parameter("max_frontier_count", 20)

    def _load_parameters(self):
        self.prediction_cycle             = self.get_parameter("prediction_cycle").value
        self.model_input_size             = self.get_parameter("model_input_size").value
        self.enable_prediction            = self.get_parameter("enable_prediction").value
        self.frontier_cluster_eps         = self.get_parameter("frontier_cluster_eps").value
        self.frontier_cluster_min_samples = self.get_parameter("frontier_cluster_min_samples").value
        self.frontier_kernel_size         = self.get_parameter("frontier_kernel_size").value
        self.prediction_expansion_cells   = self.get_parameter("prediction_expansion_cells").value
        self.max_frontier_distance        = self.get_parameter("max_frontier_distance").value
        self.max_frontier_count           = self.get_parameter("max_frontier_count").value
        self._frontier_kernel             = np.ones(
            (self.frontier_kernel_size, self.frontier_kernel_size), dtype=np.uint8
        )

    def _create_interfaces(self):
        self.visible_map_subscription       = self.create_subscription(OccupancyGrid, "/visible_map", self.visible_map_callback, 10)
        self.odom_subscription              = self.create_subscription(Odometry, "/state_estimation", self.odom_callback, 10)
        self.predicted_map_publisher        = self.create_publisher(OccupancyGrid, "/predicted_map", 10)
        self.predicted_confidence_publisher = self.create_publisher(ProbGrid, "/confidence_map", 10)
        self.frontier_publisher             = self.create_publisher(PointCloud2, "/all_frontier", 10)
        self.timer                          = self.create_timer(self.prediction_cycle, self.prediction_timer_callback)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _build_prediction_outputs(self, visible_map, clustered_frontiers, cycle_ctx):
        """Return (predicted OccupancyGrid, confidence ProbGrid) for this cycle."""
        raise NotImplementedError

    def _inputs_ready(self):
        """Extra readiness gate on top of visible map + odometry."""
        return True

    def _begin_cycle(self):
        """Per-cycle context handed to `_build_prediction_outputs` and `_end_cycle`."""
        return None

    def _end_cycle(self, visible_map, cycle_ctx):
        """Called after the outputs are published, still on the worker thread."""

    # ------------------------------------------------------------------
    # Callbacks and the inference loop
    # ------------------------------------------------------------------

    def visible_map_callback(self, msg: OccupancyGrid):
        if not msg.data or msg.info.width == 0 or msg.info.height == 0:
            self.get_logger().warn("Received empty visible map, skipping.")
            return
        self.visible_map = msg

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg

    def prediction_timer_callback(self):
        if self.visible_map is None:
            self.get_logger().info("No visible map received yet")
            return

        if self.current_odom is None:
            self.get_logger().info("No odometry received yet")
            return

        if not self._inputs_ready():
            return

        # Skip if previous inference is still running
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return

        # Snapshot current state and launch inference in background
        map_snapshot = self.visible_map
        odom_snapshot = self.current_odom
        self._inference_thread = threading.Thread(
            target=self._run_inference_background,
            args=(map_snapshot, odom_snapshot, self._begin_cycle()),
            daemon=True,
        )
        self._inference_thread.start()

    def _run_inference_background(self, visible_map, odom, cycle_ctx):
        t0 = time.time()
        clustered_frontiers = self.extract_clustered_frontiers(visible_map, odom)[:self.max_frontier_count]
        self.publish_frontiers(clustered_frontiers)

        result = self._build_prediction_outputs(visible_map, clustered_frontiers, cycle_ctx)

        # Publish immediately on completion so the planner gets fresh data without
        # waiting an entire prediction_cycle for the next timer tick.
        if result is not None:
            self.predicted_map_publisher.publish(result[0])
            self.predicted_confidence_publisher.publish(result[1])

        self._end_cycle(visible_map, cycle_ctx)

        dt = (time.time() - t0) * 1000.0
        self.get_logger().info(
            f"{self.inference_label} done: {dt:.1f}ms  "
            f"frontiers={len(clustered_frontiers)}  "
            f"map={visible_map.info.width}x{visible_map.info.height}"
        )

    # ------------------------------------------------------------------
    # Map helpers
    # ------------------------------------------------------------------

    def _expand_map(self, visible_map_np, expansion, visible_map):
        """Create a larger map with visible_map placed at centre, surrounded by UNKNOWN."""
        h, w = visible_map_np.shape
        expanded_h = h + 2 * expansion
        expanded_w = w + 2 * expansion

        expanded_map = np.full((expanded_h, expanded_w), UNKNOWN_VALUE, dtype=np.uint8)
        expanded_map[expansion:expansion + h, expansion:expansion + w] = visible_map_np

        expanded_info = deepcopy(visible_map.info)
        expanded_info.height = expanded_h
        expanded_info.width = expanded_w
        expanded_info.origin.position.x -= expansion * visible_map.info.resolution
        expanded_info.origin.position.y -= expansion * visible_map.info.resolution

        return expanded_map, expanded_info

    def _build_passthrough_outputs(self, visible_map):
        """Outputs with no prediction at all: the expanded visible map, confident
        only where it was actually observed. Used whenever a backend has nothing
        to add this cycle (prediction disabled, or no frontiers)."""
        visible_map_np = self.occupancy_grid_to_numpy(visible_map)
        exp = self.prediction_expansion_cells
        expanded_map_np, expanded_info = self._expand_map(visible_map_np, exp, visible_map)
        predicted_confidence = np.zeros(expanded_map_np.shape, dtype=np.float32)
        h, w = visible_map_np.shape
        known_mask = visible_map_np != UNKNOWN_VALUE
        predicted_confidence[exp:exp + h, exp:exp + w][known_mask] = 1.0
        return (
            self.numpy_to_occupancy_grid(expanded_map_np, expanded_info),
            self.numpy_to_confidence_grid(predicted_confidence, expanded_info),
        )

    # ------------------------------------------------------------------
    # Frontiers
    # ------------------------------------------------------------------

    def extract_clustered_frontiers(self, occupancy_grid: OccupancyGrid, robot_pos: Odometry):
        grid = np.asarray(occupancy_grid.data, dtype=np.int8).reshape(
            (occupancy_grid.info.height, occupancy_grid.info.width)
        )
        free_mask = grid == 0
        unknown_mask = (grid == -1).astype(np.uint8)
        frontier_mask = free_mask & cv2.dilate(unknown_mask, self._frontier_kernel)
        row_indices, col_indices = np.where(frontier_mask)
        if row_indices.size == 0:
            return []

        origin = occupancy_grid.info.origin.position
        resolution = occupancy_grid.info.resolution
        points = np.column_stack((
            origin.x + (col_indices.astype(np.float32) + 0.5) * resolution,
            origin.y + (row_indices.astype(np.float32) + 0.5) * resolution,
        ))
        # Filter out frontiers too far from the robot
        rx = robot_pos.pose.pose.position.x
        ry = robot_pos.pose.pose.position.y
        if self.max_frontier_distance > 0:
            points = points[
                np.linalg.norm(points - np.array([rx, ry]), axis=1)
                <= self.max_frontier_distance
            ]
        # Sort clustered frontiers nearest-first so the caller's
        # [:max_frontier_count] keeps the CLOSEST frontiers, not an arbitrary
        # DBSCAN-label-order slice. Prediction is only actionable near the robot
        # (distant guesses are observed directly before it ever arrives, and only
        # add far-field phantom that drives wasted trips), so the budget should be
        # spent on the local neighbourhood.
        clustered = self.cluster_frontiers(points, origin.z)
        clustered.sort(key=lambda f: (f[0] - rx) ** 2 + (f[1] - ry) ** 2)
        return clustered

    def cluster_frontiers(self, points, z=0.0):
        if len(points) == 0:
            return []

        labels = DBSCAN(
            eps=self.frontier_cluster_eps,
            min_samples=self.frontier_cluster_min_samples,
        ).fit(points).labels_

        clustered_frontiers = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            x, y = cluster_points[np.argmin(distances)]
            clustered_frontiers.append((float(x), float(y), z))

        return clustered_frontiers

    def publish_frontiers(self, frontiers):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        frontier_cloud = point_cloud2.create_cloud_xyz32(header, frontiers)
        self.frontier_publisher.publish(frontier_cloud)

    # ------------------------------------------------------------------
    # Grid <-> numpy conversion
    # ------------------------------------------------------------------

    @staticmethod
    def occupancy_grid_to_numpy(occupancy_grid: OccupancyGrid):
        if not occupancy_grid.data:
            return None

        grid_array = np.asarray(occupancy_grid.data, dtype=np.int8).reshape(
            (occupancy_grid.info.height, occupancy_grid.info.width)
        )
        return OCCUPANCY_TO_MODEL[grid_array.view(np.uint8)]

    @staticmethod
    def world_to_grid(world_points, occupancy_grid):
        if not occupancy_grid or not world_points:
            return []

        resolution = occupancy_grid.info.resolution
        origin = occupancy_grid.info.origin.position
        return [
            (
                int((world_x - origin.x) / resolution),
                int((world_y - origin.y) / resolution),
            )
            for world_x, world_y, *_ in world_points
        ]

    def numpy_to_occupancy_grid(self, grid_array, grid_info):
        if grid_array is None or grid_info is None:
            return None

        occupancy_grid = OccupancyGrid()
        occupancy_grid.info = grid_info
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = "map"

        data = MODEL_TO_OCCUPANCY[grid_array.ravel()]
        occupancy_grid.data = array.array('b', data)
        return occupancy_grid

    def numpy_to_confidence_grid(self, grid_array, grid_info):
        if grid_array is None or grid_info is None:
            return None

        prob_grid = ProbGrid()
        prob_grid.info = grid_info
        prob_grid.header.stamp = self.get_clock().now().to_msg()
        prob_grid.header.frame_id = "map"
        prob_grid.data = array.array(
            'f', np.asarray(grid_array, dtype=np.float32).ravel()
        )
        return prob_grid
