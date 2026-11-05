import numpy as np
import pytest

from grid_recorder.reference import build_reference, build_reference_layers, covered_cells

RES = 0.5


def wall_points(cells, z=0.5):
    """One point in the middle of each (col, row) cell."""
    return np.array([[(c + 0.5) * RES, (r + 0.5) * RES, z] for c, r in cells], dtype=np.float32)


def ring(x0, y0, x1, y1):
    return [(c, r) for c in range(x0, x1 + 1) for r in range(y0, y1 + 1)
            if c in (x0, x1) or r in (y0, y1)]


def test_keeps_reachable_free_and_its_walls_only():
    # A 3x3 room around the start, and a sealed room beside it.
    points = wall_points(ring(-2, -2, 2, 2) + ring(4, -2, 8, 2))
    mask, origin = build_reference(points, RES, 0.0, 1.0)
    # 3x3 free cells inside the ring; the walls are the ring itself.
    assert mask.sum() == 3 * 3 + len(ring(-2, -2, 2, 2))
    col = int(round((6.5 * RES - origin[0]) / RES - 0.5))
    row = int(round((0.5 * RES - origin[1]) / RES - 0.5))
    assert not mask[row, col]   # inside the sealed room


def test_layers_separate_free_from_walls():
    points = wall_points(ring(-2, -2, 2, 2))
    free, mask, _ = build_reference_layers(points, RES, 0.0, 1.0)
    assert free.sum() == 3 * 3
    assert mask[free].all()
    assert (mask & ~free).sum() == len(ring(-2, -2, 2, 2))


def test_height_band_selects_the_obstacles():
    points = wall_points(ring(-2, -2, 2, 2), z=2.0)
    with pytest.raises(ValueError):
        build_reference(points, RES, 0.0, 1.0)     # nothing in the band
    mask, _ = build_reference(points, RES, 1.5, 2.5)
    assert mask.sum() == 3 * 3 + len(ring(-2, -2, 2, 2))


def test_rejects_an_open_scene_and_a_blocked_start():
    gap = [c for c in ring(-2, -2, 2, 2) if c != (2, 0)]
    with pytest.raises(ValueError, match="edge of the scene"):
        build_reference(wall_points(gap), RES, 0.0, 1.0)
    with pytest.raises(ValueError, match="obstacle"):
        build_reference(wall_points(ring(-2, -2, 2, 2) + [(0, 0)]), RES, 0.0, 1.0)


def test_covered_cells_aligns_by_origin_and_clips():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = mask[3, 3] = True
    known = np.ones((2, 2), dtype=bool)
    # known's cell (0, 0) is mask's cell (1, 1); (3, 3) is out of its reach.
    assert covered_cells(known, (1.5, 1.5), mask, (1.0, 1.0), RES) == 1
    assert covered_cells(known, (0.0, 0.0), mask, (1.0, 1.0), RES) == 0
    known = np.ones((10, 10), dtype=bool)
    assert covered_cells(known, (-1.0, -1.0), mask, (1.0, 1.0), RES) == 2
