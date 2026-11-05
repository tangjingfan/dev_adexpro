"""Builds a world's occupancy-grid ground truth from its point cloud and mapping config.

    ros2 run grid_recorder make_reference indoor --output dev_results/ground_truth/indoor.npz
"""

import argparse
import os

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from .reference import build_reference_layers, read_ply_xyz


def _mapping_params(world):
    """resolution, z_min, z_max from the world's mapping config, whichever mapper it uses."""
    config_dir = os.path.join(get_package_share_directory("simulator"), "config")
    for name in (f"occupancy_{world}.yaml", f"mapping_{world}.yaml"):
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                params = next(iter(yaml.safe_load(f).values()))["ros__parameters"]
            if "obstacle_min_z" in params:
                return params["resolution"], params["obstacle_min_z"], params["obstacle_max_z"]
            return params["resolution"], params["occupancy_min_z"], params["occupancy_max_z"]
    raise FileNotFoundError(f"no mapping config for {world} in {config_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    args = parser.parse_args()

    resolution, z_min, z_max = _mapping_params(args.world)
    ply = os.path.join(get_package_share_directory("vehicle_simulator"),
                       "mesh", args.world, "preview", "pointcloud.ply")
    free, mask, origin = build_reference_layers(read_ply_xyz(ply), resolution, z_min, z_max,
                                                (args.start_x, args.start_y))
    np.savez_compressed(args.output, mask=mask, free=free, origin=np.array(origin),
                        resolution=np.float64(resolution))
    print(f"{args.world}: {int(mask.sum())} cells, {mask.shape[1]}x{mask.shape[0]} grid, "
          f"z {z_min}..{z_max}, resolution {resolution} -> {args.output}")


if __name__ == "__main__":
    main()
