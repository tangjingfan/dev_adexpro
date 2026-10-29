import numpy as np
import pytest

from prediction_eval import grids
from prediction_eval.grids import FREE, OCCUPIED, UNKNOWN, Grid

RES = 0.5


def grid(cls, ox=0.0, oy=0.0):
    return Grid(np.asarray(cls, dtype=np.int8), (ox, oy), RES)


def test_classify():
    cls = grids.classify([-1, 0, 49, 50, 100, -1], 2, 3)
    assert cls.tolist() == [[UNKNOWN, FREE, FREE], [OCCUPIED, OCCUPIED, UNKNOWN]]


def test_cell_offset_and_resample():
    dst = grid(np.full((4, 4), UNKNOWN))
    src = grid([[FREE, OCCUPIED]], ox=1.0, oy=0.5)   # 2 cols right, 1 row up
    assert grids.cell_offset(src, dst) == (1, 2)
    out = grids.resample(src, dst)
    assert out[1, 2] == FREE and out[1, 3] == OCCUPIED
    assert (out != UNKNOWN).sum() == 2


def test_resample_clips_partial_overlap():
    dst = grid(np.full((2, 2), UNKNOWN))
    src = grid(np.full((3, 3), FREE), ox=-0.5, oy=-0.5)
    assert (grids.resample(src, dst) == FREE).all()


def test_cell_offset_rejects_mismatch():
    with pytest.raises(ValueError):
        grids.cell_offset(Grid(np.zeros((1, 1), np.int8), (0, 0), 0.25), grid([[0]]))
    with pytest.raises(ValueError):
        grids.cell_offset(grid([[0]], ox=0.25), grid([[0]]))


def test_match_visible_picks_the_snapshot_that_was_expanded():
    pad = 2
    small = grid(np.full((3, 3), FREE), ox=0.0)
    shifted = grid(np.full((3, 3), FREE), ox=0.5)          # same size, other origin
    bigger = grid(np.full((4, 4), FREE), ox=0.0)
    predicted = grid(np.full((3 + 2 * pad, 3 + 2 * pad), UNKNOWN), ox=-pad * RES, oy=-pad * RES)
    assert grids.match_visible(predicted, [small, shifted, bigger]) is small
    assert grids.match_visible(predicted, [shifted, bigger]) is None


def test_score_only_counts_newly_predicted_cells_known_to_gt():
    visible = grid([[FREE, UNKNOWN, UNKNOWN, UNKNOWN]])
    predicted = grid([[FREE, OCCUPIED, FREE, UNKNOWN]])
    gt = grid([[FREE, OCCUPIED, OCCUPIED, FREE]])
    mask, p, g = grids.score(predicted, visible, gt)
    # col 0 was already visible, col 3 was not predicted
    assert mask.tolist() == [[False, True, True, False]]
    m = grids.cycle_metrics(mask, p, g)
    assert m["n_eval"] == 2 and m["acc"] == 0.5
    assert m["occ_precision"] == 1.0 and m["occ_recall"] == 0.5


def test_canvas_accumulates_and_rebases():
    gt = grid(np.array([[FREE, OCCUPIED]]))
    canvas = grids.AccuracyCanvas(gt)
    frame = grid([[OCCUPIED, OCCUPIED]])
    mask = np.array([[True, True]])
    for _ in range(3):
        canvas.add(frame, mask, frame.cls, gt.cls)
    assert canvas.counts[grids.PO_GF, 0, 0] == 3
    assert canvas.counts[grids.PO_GO, 0, 1] == 3
    assert canvas.counts.sum() == 6

    # GT grows by one column on the left: counts move with their cells.
    canvas.rebase(grid(np.array([[UNKNOWN, FREE, OCCUPIED]]), ox=-RES))
    assert canvas.counts[grids.PO_GF, 0, 1] == 3
    assert canvas.counts[grids.PO_GO, 0, 2] == 3
    assert canvas.counts.sum() == 6


def test_from_occupancy_grid_reads_attributes():
    from types import SimpleNamespace as NS
    msg = NS(data=[0, 100, -1, 0], info=NS(width=2, height=2, resolution=RES,
                                            origin=NS(position=NS(x=1.0, y=-2.0))))
    g = grids.from_occupancy_grid(msg)
    assert g.cls.tolist() == [[FREE, OCCUPIED], [UNKNOWN, FREE]]
    assert g.origin == (1.0, -2.0) and g.resolution == RES


def test_per_cell_grids():
    counts = np.zeros((4, 1, 3), dtype=np.uint32)
    counts[grids.PF_GF, 0, 0] = 3                                    # always right
    counts[grids.PO_GF, 0, 1], counts[grids.PO_GO, 0, 1] = 1, 3      # one phantom obstacle
    n = grids.n_eval(counts)
    acc, bal = grids.accuracy_grid(counts), grids.error_balance_grid(counts)
    assert n.tolist() == [[3, 4, 0]]
    assert acc[0, 0] == 1.0 and acc[0, 1] == 0.75 and np.isnan(acc[0, 2])
    assert np.isnan(bal[0, 0]) and bal[0, 1] == 0.25 and np.isnan(bal[0, 2])
