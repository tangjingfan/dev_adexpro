"""Live heatmaps of prediction accuracy, drawn from prediction_eval's output.

Subscribes to /prediction_eval/{accuracy,error_balance,n_eval} (ProbGrids on
the GT grid), /gt_map for the background and /state_estimation for the robot's
path, and redraws a matplotlib window every `refresh_period_s`:
  - accuracy: per cell, the share of its predictions that matched GT
  - error type: per cell, phantom obstacles (red) vs missed obstacles (blue)
Cells scored fewer than `min_count` times are hidden. Closing the window ends
the node; prediction_eval keeps recording.
"""

import threading

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from prob_msgs.msg import ProbGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from prediction_eval import grids
from prediction_eval.eval_node import RESULT_TOPICS


def _grid_of(msg):
    info = msg.info
    return grids.Grid(np.zeros((info.height, info.width), np.int8),
                      (info.origin.position.x, info.origin.position.y), info.resolution)


class PredictionHeatmapNode(Node):

    def __init__(self):
        super().__init__("prediction_heatmap_node")
        self.method = self.declare_parameter("method", "unknown").value
        self.world = self.declare_parameter("world", "unknown").value
        self.min_count = self.declare_parameter("min_count", 3).value
        self.refresh_period_s = self.declare_parameter("refresh_period_s", 2.0).value
        self.trajectory_period_s = self.declare_parameter("trajectory_period_s", 1.0).value

        # Callbacks run on the executor thread, the window on the main thread.
        self.lock = threading.Lock()
        self.results = {}
        self.result_seq = 0
        self.gt = None
        self.robot_xy = None
        self.trajectory = []
        self._last_traj_t = -np.inf

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for key, topic in RESULT_TOPICS.items():
            self.create_subscription(ProbGrid, topic, lambda m, k=key: self._result_cb(k, m), latched)
        self.create_subscription(OccupancyGrid, "/gt_map", self._gt_cb, 10)
        self.create_subscription(Odometry, "/state_estimation", self._odom_cb, 10)

    def _result_cb(self, key, msg):
        with self.lock:
            self.results[key] = msg
            self.result_seq += 1

    def _gt_cb(self, msg):
        if msg.info.width and msg.info.height:
            gt = grids.from_occupancy_grid(msg)
            with self.lock:
                self.gt = gt

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        t = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            self.robot_xy = (p.x, p.y)
            if t - self._last_traj_t >= self.trajectory_period_s:
                self._last_traj_t = t
                self.trajectory.append((p.x, p.y))

    def snapshot(self):
        """(seq, grid, gt_cls, acc, bal, n, trajectory, robot_xy) once all three
        results of one publish (same stamp) are in, else None."""
        with self.lock:
            msgs = [self.results.get(k) for k in RESULT_TOPICS]
            if None in msgs or len({(m.header.stamp.sec, m.header.stamp.nanosec) for m in msgs}) != 1:
                return None
            grid = _grid_of(msgs[0])
            acc, bal, n = (np.asarray(m.data, dtype=np.float32).reshape(grid.shape) for m in msgs)
            gt_cls = grids.resample(self.gt, grid) if self.gt is not None else grid.cls
            traj = np.array(self.trajectory, dtype=np.float64).reshape(-1, 2)
            return self.result_seq, grid, gt_cls, acc, bal, n, traj, self.robot_xy


def _run_window(node):
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    from . import heatmap

    fig, (ax_acc, ax_err) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=heatmap.SURFACE)
    fig.canvas.manager.set_window_title(f"Prediction accuracy: {node.method} / {node.world}")
    panels = heatmap.HeatmapPanels(fig, ax_acc, ax_err, f"{node.method} / {node.world}",
                                   min_count=node.min_count)
    waiting = ax_acc.text(0.5, 0.5, "waiting for prediction_eval results",
                          transform=ax_acc.transAxes, ha="center", va="center",
                          color=heatmap.TEXT_MUTED, fontsize=9)
    plt.show(block=False)
    drawn_seq = None
    while rclpy.ok() and plt.fignum_exists(fig.number):
        snap = node.snapshot()
        if snap is not None and snap[0] != drawn_seq:
            if drawn_seq is None:
                waiting.remove()
            drawn_seq = snap[0]
            panels.update(*snap[1:])
            fig.tight_layout()
            fig.canvas.draw_idle()
        plt.pause(node.refresh_period_s)


def main(args=None):
    rclpy.init(args=args)
    node = PredictionHeatmapNode()
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    try:
        _run_window(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
