"""Latches a world's ground-truth reachable free space on /reference_free_map.

For oracle experiments only. Reads <EXPRO_EXPERIMENTS_DIR>/../ground_truth/<world>.npz:

    ros2 run grid_recorder reference_publisher indoor
"""

import os
import sys

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def to_occupancy_data(free):
    """Occupancy values of a [row, col] boolean grid, in ROS row-major order."""
    return np.where(free, 0, 100).astype(np.int8).ravel().tolist()


class ReferencePublisher(Node):
    def __init__(self, world):
        super().__init__("reference_publisher")
        path = os.path.join(
            os.environ["EXPRO_EXPERIMENTS_DIR"], "..", "ground_truth", f"{world}.npz")
        reference = np.load(path)
        free = reference["free"]

        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = float(reference["resolution"])
        grid.info.height, grid.info.width = (int(n) for n in free.shape)
        grid.info.origin.position.x = float(reference["origin"][0])
        grid.info.origin.position.y = float(reference["origin"][1])
        grid.info.origin.orientation.w = 1.0
        grid.data = to_occupancy_data(free)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = self.create_publisher(OccupancyGrid, "/reference_free_map", latched)
        self._publisher.publish(grid)
        self.get_logger().info(f"{path}: {int(free.sum())} free cells")


def main():
    rclpy.init()
    node = ReferencePublisher(sys.argv[1])
    rclpy.spin(node)


if __name__ == "__main__":
    main()
