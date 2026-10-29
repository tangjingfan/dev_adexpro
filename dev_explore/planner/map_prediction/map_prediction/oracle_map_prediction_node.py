"""Oracle prediction backend: instead of running the learned model, fill each
frontier's local window from a ground-truth occupancy grid. Serves the same
contract as the learned backend, so it is a drop-in replacement for measuring
how much a perfect predictor would buy."""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid

from .grid_encoding import UNKNOWN_VALUE
from .prediction_node_base import PredictionNodeBase


class OracleMapPredictionNode(PredictionNodeBase):
    inference_label = "Oracle inference"

    def __init__(self):
        super().__init__("oracle_map_prediction_node")

        self.gt_map = None
        self._resolution_warned = False

        self.get_logger().info(
            f"Oracle ready: gt_map_topic={self.gt_map_topic} "
            f"model_input_size={self.model_input_size} "
            f"expansion={self.prediction_expansion_cells}"
        )

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter("gt_map_topic", "/gt_map")

    def _load_parameters(self):
        super()._load_parameters()
        self.gt_map_topic = self.get_parameter("gt_map_topic").value

    def _create_interfaces(self):
        super()._create_interfaces()
        # Match /visible_map's volatile QoS — our GT source (octomap_server fed
        # by /overall_map with latch=false) republishes on every map change, so
        # no transient-local is needed and using it would break the QoS handshake.
        self.gt_map_subscription = self.create_subscription(
            OccupancyGrid, self.gt_map_topic, self.gt_map_callback, 10
        )

    def gt_map_callback(self, msg: OccupancyGrid):
        if not msg.data or msg.info.width == 0 or msg.info.height == 0:
            self.get_logger().warn("Received empty GT map, skipping.")
            return
        self.gt_map = msg

    def _inputs_ready(self):
        if self.gt_map is None:
            self.get_logger().info(f"No GT map received yet on {self.gt_map_topic}")
            return False
        return True

    def _begin_cycle(self):
        return self.gt_map

    def _build_prediction_outputs(self, visible_map, clustered_frontiers, gt_map):
        if not self.enable_prediction or not clustered_frontiers:
            return self._build_passthrough_outputs(visible_map)

        visible_map_np = self.occupancy_grid_to_numpy(visible_map)
        exp = self.prediction_expansion_cells
        # _expand_map returns a fresh array, so the windows are pasted into it.
        predicted_map_np, expanded_info = self._expand_map(visible_map_np, exp, visible_map)
        predicted_confidence_np = np.zeros(predicted_map_np.shape, dtype=np.float32)

        h, w = visible_map_np.shape
        known_mask_expanded = np.zeros(predicted_map_np.shape, dtype=bool)
        known_mask_expanded[exp:exp + h, exp:exp + w] = visible_map_np != UNKNOWN_VALUE
        predicted_confidence_np[known_mask_expanded] = 1.0

        self._check_resolution_compatibility(visible_map, gt_map)
        gt_map_np = self.occupancy_grid_to_numpy(gt_map)

        # The frontier is one world point; convert it twice — once into the
        # expanded predicted-map grid (where we paste) and once into the GT grid
        # (where we crop). The two grids generally have different origins.
        raw_frontier_points = self.world_to_grid(clustered_frontiers, visible_map)
        expanded_frontier_points = [(x + exp, y + exp) for x, y in raw_frontier_points]
        gt_frontier_points = self.world_to_grid(clustered_frontiers, gt_map)

        for pred_xy, gt_xy in zip(expanded_frontier_points, gt_frontier_points):
            self._paste_local_window(
                predicted_map_np,
                predicted_confidence_np,
                known_mask_expanded,
                gt_map_np,
                pred_xy,
                gt_xy,
                self.model_input_size,
            )

        return (
            self.numpy_to_occupancy_grid(predicted_map_np, expanded_info),
            self.numpy_to_confidence_grid(predicted_confidence_np, expanded_info),
        )

    @staticmethod
    def _paste_local_window(
        predicted_map,
        predicted_confidence,
        known_mask,
        gt_map_np,
        pred_xy,
        gt_xy,
        local_half_size,
    ):
        """Copy a 2*local_half_size square from GT (centred on gt_xy) into the
        predicted map (centred on pred_xy). Only cells that are currently unknown
        in the visible map are overwritten — visible cells always win."""
        pred_x, pred_y = pred_xy
        gt_x, gt_y = gt_xy
        pred_h, pred_w = predicted_map.shape
        gt_h, gt_w = gt_map_np.shape

        pred_y0 = max(0, pred_y - local_half_size)
        pred_y1 = min(pred_h, pred_y + local_half_size)
        pred_x0 = max(0, pred_x - local_half_size)
        pred_x1 = min(pred_w, pred_x + local_half_size)
        if pred_y1 <= pred_y0 or pred_x1 <= pred_x0:
            return

        # GT slice tracks the predicted slice via the same offsets from each
        # centre, so the two crops line up cell-for-cell.
        gt_y0 = gt_y + (pred_y0 - pred_y)
        gt_y1 = gt_y + (pred_y1 - pred_y)
        gt_x0 = gt_x + (pred_x0 - pred_x)
        gt_x1 = gt_x + (pred_x1 - pred_x)

        clip_y0 = max(0, -gt_y0)
        clip_y1 = max(0, gt_y1 - gt_h)
        clip_x0 = max(0, -gt_x0)
        clip_x1 = max(0, gt_x1 - gt_w)

        gt_y0 += clip_y0
        gt_y1 -= clip_y1
        gt_x0 += clip_x0
        gt_x1 -= clip_x1
        pred_y0 += clip_y0
        pred_y1 -= clip_y1
        pred_x0 += clip_x0
        pred_x1 -= clip_x1

        if gt_y1 <= gt_y0 or gt_x1 <= gt_x0:
            return

        gt_window = gt_map_np[gt_y0:gt_y1, gt_x0:gt_x1]
        pred_slice = predicted_map[pred_y0:pred_y1, pred_x0:pred_x1]
        conf_slice = predicted_confidence[pred_y0:pred_y1, pred_x0:pred_x1]
        known_slice = known_mask[pred_y0:pred_y1, pred_x0:pred_x1]

        update = (gt_window != UNKNOWN_VALUE) & ~known_slice
        pred_slice[update] = gt_window[update]
        conf_slice[update] = 1.0

    def _check_resolution_compatibility(self, visible_map, gt_map):
        # The window crop is in *cells*, so different resolutions silently change
        # the physical footprint of the oracle. Warn once instead of resampling —
        # the typical setup publishes GT at the visible-map resolution.
        if self._resolution_warned:
            return
        vr = visible_map.info.resolution
        gr = gt_map.info.resolution
        if abs(vr - gr) > 1e-4:
            self._resolution_warned = True
            self.get_logger().warn(
                f"GT resolution ({gr:.4f} m) differs from visible map "
                f"({vr:.4f} m); oracle window footprint will not match the "
                f"learned predictor. Republish GT at {vr:.4f} m to fix."
            )


def main(args=None):
    rclpy.init(args=args)
    node = OracleMapPredictionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
