"""The region of a world an occupancy grid should end up covering (ROS-free).

Built from the world's point cloud, so it depends neither on a run nor on any
method: the scene's points in the mapping height band become obstacle cells, a
flood fill from the robot's start gives the reachable free cells, and the
obstacle cells touching those are the walls the robot can see.
"""

from collections import deque

import numpy as np


def read_ply_xyz(path):
    """x, y, z columns of a binary little-endian PLY of float properties."""
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: expected a binary little-endian PLY")
    properties = [line.split() for line in header if line.startswith("property")]
    if any(p[1] != "float" for p in properties):
        raise ValueError(f"{path}: expected float properties only, got {properties}")
    names = [p[2] for p in properties]
    points = np.frombuffer(data, dtype="<f4", offset=end).reshape(-1, len(names))
    return points[:, [names.index(axis) for axis in "xyz"]]


def _touching(mask):
    """Cells that are in mask or 8-adjacent to a cell in it."""
    grown = mask.copy()
    grown[1:] |= mask[:-1]
    grown[:-1] |= mask[1:]
    across = grown.copy()
    grown[:, 1:] |= across[:, :-1]
    grown[:, :-1] |= across[:, 1:]
    return grown


def build_reference(points, resolution, z_min, z_max, start_xy=(0.0, 0.0)):
    """(mask, origin): reachable free cells plus the walls touching them.

    mask is indexed [row (y), col (x)]; origin is the world x, y of cell (0, 0).
    Raises ValueError as build_reference_layers does.
    """
    _, mask, origin = build_reference_layers(points, resolution, z_min, z_max, start_xy)
    return mask, origin


def build_reference_layers(points, resolution, z_min, z_max, start_xy=(0.0, 0.0)):
    """(free, mask, origin): the reachable free cells alone, and with their walls.

    Raises ValueError if the start is not free or the flood fill reaches the
    edge of the scene, i.e. the world has no closed boundary to measure against.
    """
    band = points[(points[:, 2] >= z_min) & (points[:, 2] < z_max)]
    if len(band) == 0:
        raise ValueError(f"no points between z = {z_min} and {z_max}")
    cells = np.floor(band[:, :2] / resolution).astype(int)
    low = cells.min(axis=0) - 2
    high = cells.max(axis=0) + 3
    occupied = np.zeros((high[1] - low[1], high[0] - low[0]), dtype=bool)
    occupied[cells[:, 1] - low[1], cells[:, 0] - low[0]] = True

    start = (int(np.floor(start_xy[1] / resolution)) - low[1],
             int(np.floor(start_xy[0] / resolution)) - low[0])
    height, width = occupied.shape
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        raise ValueError(f"start {start_xy} is outside the scene")
    if occupied[start]:
        raise ValueError(f"start {start_xy} is an obstacle cell")

    free = np.zeros_like(occupied)
    free[start] = True
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        if row in (0, height - 1) or col in (0, width - 1):
            raise ValueError("flood fill reached the edge of the scene: no closed boundary")
        for next_cell in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            if not occupied[next_cell] and not free[next_cell]:
                free[next_cell] = True
                queue.append(next_cell)

    mask = free | (_touching(free) & occupied)
    return free, mask, (float(low[0] * resolution), float(low[1] * resolution))


def covered_cells(known, map_origin, mask, mask_origin, resolution):
    """How many cells of mask are known. known and mask are boolean grids on the
    same lattice with different origins; each is clipped to the overlap."""
    off_x = int(round((map_origin[0] - mask_origin[0]) / resolution))
    off_y = int(round((map_origin[1] - mask_origin[1]) / resolution))
    y0, y1 = max(0, off_y), min(mask.shape[0], off_y + known.shape[0])
    x0, x1 = max(0, off_x), min(mask.shape[1], off_x + known.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(np.count_nonzero(
        mask[y0:y1, x0:x1] & known[y0 - off_y:y1 - off_y, x0 - off_x:x1 - off_x]))
