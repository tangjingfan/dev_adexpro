"""Scores every /predicted_map against the ground-truth /gt_map.

For each prediction cycle, the cells the predictor filled in (unknown in the
visible map it was given, known in its output) are compared with GT. Per-cell
outcome counts accumulate on the GT grid, so the result is a map of where the
predictor tends to be right or wrong; per-cycle metrics go to a CSV.

Every `publish_period_s` (when something changed) the per-cell results are
published on the GT grid as prob_msgs/ProbGrid, all with the same stamp:
    /prediction_eval/accuracy       share of predictions that matched GT (NaN: never scored)
    /prediction_eval/error_balance  (phantom - missed) / predictions (NaN: never wrong)
    /prediction_eval/n_eval         times each cell was scored
prediction_heatmap draws them live.

Output, under $EXPRO_EXPERIMENTS_DIR (default ./experiments):
    runs/<EXPRO_RUN_ID>/prediction_eval/        when launched by the runner
    prediction_eval/<method>_<world>_<stamp>/   otherwise
holding accuracy.npz (rewritten every `save_period_s`) and cycles.csv.
"""

import array
import os
import signal
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from prob_msgs.msg import ProbGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from . import grids

RESULT_TOPICS = {
    "accuracy": "/prediction_eval/accuracy",
    "error_balance": "/prediction_eval/error_balance",
    "n_eval": "/prediction_eval/n_eval",
}


def _map_info(g):
    info = MapMetaData()
    info.resolution = float(g.resolution)
    info.height, info.width = (int(v) for v in g.shape)
    info.origin.position.x, info.origin.position.y = (float(v) for v in g.origin)
    info.origin.orientation.w = 1.0
    return info


class PredictionEvalNode(Node):

    def __init__(self):
        super().__init__("prediction_eval_node")
        self.method = self.declare_parameter("method", "unknown").value
        self.world = self.declare_parameter("world", "unknown").value
        self.save_period_s = self.declare_parameter("save_period_s", 5.0).value
        self.publish_period_s = self.declare_parameter("publish_period_s", 2.0).value
        self.trajectory_period_s = self.declare_parameter("trajectory_period_s", 1.0).value

        self.out_dir = self._resolve_out_dir()
        os.makedirs(self.out_dir, exist_ok=True)
        self.npz_path = os.path.join(self.out_dir, "accuracy.npz")
        self.csv = open(os.path.join(self.out_dir, "cycles.csv"), "w", buffering=1)
        self.csv.write("t_s,n_eval,acc,occ_precision,occ_recall,occ_iou,robot_x,robot_y\n")

        self.visible_history = deque(maxlen=30)
        self.canvas = None
        self.robot_xy = None
        self.trajectory = []
        self.n_cycles = 0
        self.n_unmatched = 0
        self.t0 = time.monotonic()
        self._last_traj_t = -np.inf
        self._unsaved = False
        self._unpublished = False

        self.create_subscription(OccupancyGrid, "/visible_map", self._visible_cb, 10)
        self.create_subscription(OccupancyGrid, "/gt_map", self._gt_cb, 10)
        self.create_subscription(OccupancyGrid, "/predicted_map", self._predicted_cb, 10)
        self.create_subscription(Odometry, "/state_estimation", self._odom_cb, 10)
        # Transient-local so a viewer started later still gets the latest result.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.result_pubs = {k: self.create_publisher(ProbGrid, t, latched)
                            for k, t in RESULT_TOPICS.items()}
        self.create_timer(self.save_period_s, self.save)
        self.create_timer(self.publish_period_s, self.publish_results)

        self.get_logger().info(f"Prediction eval ({self.method}, {self.world}) -> {self.out_dir}")

    def _resolve_out_dir(self):
        exp = os.environ.get("EXPRO_EXPERIMENTS_DIR", "").strip() or \
            os.path.join(os.getcwd(), "experiments")
        run_id = os.environ.get("EXPRO_RUN_ID", "").strip()
        if run_id:
            return os.path.join(exp, "runs", run_id, "prediction_eval")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(exp, "prediction_eval", f"{self.method}_{self.world}_{stamp}")

    # ------------------------------------------------------------------

    def _elapsed(self):
        return time.monotonic() - self.t0

    def _visible_cb(self, msg):
        if msg.info.width and msg.info.height:
            self.visible_history.append(grids.from_occupancy_grid(msg))

    def _gt_cb(self, msg):
        if not (msg.info.width and msg.info.height):
            return
        gt = grids.from_occupancy_grid(msg)
        if self.canvas is None:
            self.canvas = grids.AccuracyCanvas(gt)
            self.get_logger().info(f"GT map received: {gt.shape[1]}x{gt.shape[0]} cells")
            return
        try:
            if gt.shape != self.canvas.gt.shape or grids.cell_offset(gt, self.canvas.gt) != (0, 0):
                self.canvas.rebase(gt)
            else:
                self.canvas.gt = gt
        except ValueError as e:
            self.get_logger().warn(f"Ignoring GT map: {e}", throttle_duration_sec=10.0)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self.robot_xy = (p.x, p.y)
        t = self._elapsed()
        if t - self._last_traj_t >= self.trajectory_period_s:
            self._last_traj_t = t
            self.trajectory.append((t, p.x, p.y))

    def _predicted_cb(self, msg):
        if self.canvas is None or not self.visible_history:
            self.get_logger().info("Waiting for /gt_map and /visible_map", throttle_duration_sec=10.0)
            return
        predicted = grids.from_occupancy_grid(msg)
        visible = grids.match_visible(predicted, self.visible_history)
        if visible is None:
            # Not the plain "visible map padded on every side" layout; fall
            # back to the newest visible map, aligned by world coordinates.
            self.n_unmatched += 1
            visible = self.visible_history[-1]
            self.get_logger().warn(
                "No visible-map snapshot matches /predicted_map; using the latest one",
                throttle_duration_sec=30.0)
        try:
            mask, pred_cls, gt_cls = grids.score(predicted, visible, self.canvas.gt)
        except ValueError as e:
            self.get_logger().warn(f"Skipping cycle: {e}", throttle_duration_sec=10.0)
            return
        self.canvas.add(predicted, mask, pred_cls, gt_cls)
        self.n_cycles += 1
        self._unsaved = self._unpublished = True

        m = grids.cycle_metrics(mask, pred_cls, gt_cls)
        rx, ry = self.robot_xy if self.robot_xy else (float("nan"), float("nan"))
        self.csv.write(
            f"{self._elapsed():.2f},{m['n_eval']},{m['acc']:.4f},{m['occ_precision']:.4f},"
            f"{m['occ_recall']:.4f},{m['occ_iou']:.4f},{rx:.2f},{ry:.2f}\n")

    # ------------------------------------------------------------------

    def publish_results(self):
        if self.canvas is None or not self._unpublished:
            return
        c = self.canvas
        info = _map_info(c.gt)
        stamp = self.get_clock().now().to_msg()
        values = {
            "accuracy": grids.accuracy_grid(c.counts),
            "error_balance": grids.error_balance_grid(c.counts),
            "n_eval": grids.n_eval(c.counts),
        }
        for key, pub in self.result_pubs.items():
            msg = ProbGrid()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            msg.info = info
            msg.data = array.array("f", values[key].astype(np.float32).ravel())
            pub.publish(msg)
        self._unpublished = False

    def save(self):
        if self.canvas is None or not self._unsaved:
            return
        c = self.canvas
        visible = self.visible_history[-1] if self.visible_history else None
        explored = grids.resample(visible, c.gt) != grids.UNKNOWN if visible else \
            np.zeros(c.gt.shape, dtype=bool)
        tmp = self.npz_path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            counts=c.counts,
            count_names=np.array(grids.COUNT_NAMES),
            gt=c.gt.cls,
            explored=explored,
            origin=np.array(c.gt.origin, dtype=np.float64),
            resolution=np.float64(c.gt.resolution),
            trajectory=np.array(self.trajectory, dtype=np.float64).reshape(-1, 3),
            method=np.str_(self.method),
            world=np.str_(self.world),
            n_cycles=np.int64(self.n_cycles),
            n_unmatched=np.int64(self.n_unmatched),
        )
        os.replace(tmp, self.npz_path)
        self._unsaved = False

    def close(self):
        self.save()
        self.csv.close()


def main(args=None):
    rclpy.init(args=args)
    node = PredictionEvalNode()

    # tmux kill-session delivers SIGHUP; turn it into a clean shutdown so the
    # final counts are written.
    def _stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGHUP, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
