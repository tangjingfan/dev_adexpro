"""Records the occupancy grid over a run, for comparing runs afterwards.

Started with the mapping stack (simulator's mapping_<world>.launch). Every
`save_period_s` the latest /projected_map is saved as
<EXPRO_RUN_DIR>/occupancy_grid/<t_s>.npz, holding:
    data        the grid, int8 [row (y), col (x)]: -1 unknown, 0 free, > 0 occupied
    origin      world x, y of cell (0, 0)
    resolution  metres per cell
    t_s         seconds on /time_duration, the same clock as visualization_tools'
                metrics file
Each file appears whole, so killing the run loses at most the latest period.
Without EXPRO_RUN_DIR (mapping launched by hand) nothing is recorded.
"""

import os

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32


class GridRecorder(Node):

    def __init__(self):
        super().__init__("grid_recorder")
        self.save_period_s = self.declare_parameter("save_period_s", 1.0).value
        self.t_s = None
        self.last_saved_t = None
        run_dir = os.environ.get("EXPRO_RUN_DIR", "").strip()
        if not run_dir:
            self.get_logger().info("EXPRO_RUN_DIR not set; not recording the occupancy grid")
            return
        self.out_dir = os.path.join(run_dir, "occupancy_grid")
        os.makedirs(self.out_dir, exist_ok=True)
        self.create_subscription(Float32, "/time_duration", self._time_cb, 10)
        self.create_subscription(OccupancyGrid, "/projected_map", self._map_cb, 10)
        self.get_logger().info(f"Recording the occupancy grid to {self.out_dir}")

    def _time_cb(self, msg):
        self.t_s = msg.data

    def _map_cb(self, msg):
        # Before /time_duration starts there is no clock to stamp the grid with.
        if self.t_s is None:
            return
        if self.last_saved_t is not None and self.t_s - self.last_saved_t < self.save_period_s:
            return
        self.last_saved_t = self.t_s
        info = msg.info
        tmp = os.path.join(self.out_dir, "latest.tmp.npz")
        np.savez_compressed(
            tmp,
            data=np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width),
            origin=np.array([info.origin.position.x, info.origin.position.y]),
            resolution=np.float64(info.resolution),
            t_s=np.float64(self.t_s),
        )
        os.replace(tmp, os.path.join(self.out_dir, f"{self.t_s:09.3f}.npz"))


def main(args=None):
    rclpy.init(args=args)
    node = GridRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
