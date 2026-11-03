"""Records the size of the occupancy grid over a run.

Started with the mapping stack (simulator's mapping_<world>.launch). Each
/projected_map adds a row to <EXPRO_RUN_DIR>/grid_metrics.csv:
    t_s, known_cells, free_cells, occupied_cells, known_area_m2
t_s comes from /time_duration, the same clock as visualization_tools' metrics
file. Rows are flushed as they are written, so killing the run loses nothing.
Without EXPRO_RUN_DIR (mapping launched by hand) nothing is recorded.
"""

import os

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

CSV_HEADER = "t_s,known_cells,free_cells,occupied_cells,known_area_m2\n"


class GridRecorder(Node):

    def __init__(self):
        super().__init__("grid_recorder")
        self.t_s = None
        self.csv = None
        run_dir = os.environ.get("EXPRO_RUN_DIR", "").strip()
        if not run_dir:
            self.get_logger().info("EXPRO_RUN_DIR not set; not recording the grid size")
            return
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "grid_metrics.csv")
        self.csv = open(path, "w", buffering=1)
        self.csv.write(CSV_HEADER)
        self.create_subscription(Float32, "/time_duration", self._time_cb, 10)
        self.create_subscription(OccupancyGrid, "/projected_map", self._map_cb, 10)
        self.get_logger().info(f"Recording the grid size to {path}")

    def _time_cb(self, msg):
        self.t_s = msg.data

    def _map_cb(self, msg):
        # Before /time_duration starts there is no clock to stamp the row with.
        if self.t_s is None:
            return
        grid = np.asarray(msg.data, dtype=np.int8)
        free = int(np.count_nonzero(grid == 0))
        occupied = int(np.count_nonzero(grid > 0))
        known = free + occupied
        area = known * msg.info.resolution ** 2
        self.csv.write(f"{self.t_s:.3f},{known},{free},{occupied},{area:.2f}\n")

    def close(self):
        if self.csv is not None:
            self.csv.close()


def main(args=None):
    rclpy.init(args=args)
    node = GridRecorder()
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
