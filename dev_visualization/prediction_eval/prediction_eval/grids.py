"""Grid alignment and per-cell accuracy bookkeeping, free of ROS.

Every grid here is a `Grid`: a (rows, cols) int8 class array (row = y, col = x,
as in nav_msgs/OccupancyGrid) plus its world origin and resolution. Grids that
share a resolution line up by an integer cell offset, which is all the scoring
needs: no resampling.
"""

from dataclasses import dataclass

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 1

# counts[k] for k = 2 * predicted + truth: what was predicted vs what GT says.
PF_GF, PF_GO, PO_GF, PO_GO = range(4)
COUNT_NAMES = ("pred_free_gt_free", "pred_free_gt_occ", "pred_occ_gt_free", "pred_occ_gt_occ")


@dataclass
class Grid:
    cls: np.ndarray      # (rows, cols) int8: UNKNOWN / FREE / OCCUPIED
    origin: tuple        # world (x, y) of the corner of cell (0, 0)
    resolution: float

    @property
    def shape(self):
        return self.cls.shape


def classify(data, rows, cols):
    """OccupancyGrid.data (-1 unknown, 0..100 occupancy) -> class array."""
    raw = np.asarray(data, dtype=np.int16).reshape(rows, cols)
    cls = np.full((rows, cols), UNKNOWN, dtype=np.int8)
    cls[(raw >= 0) & (raw < 50)] = FREE
    cls[raw >= 50] = OCCUPIED
    return cls


def from_occupancy_grid(msg):
    """A Grid from a nav_msgs/OccupancyGrid (read by attribute, no ROS import)."""
    info = msg.info
    return Grid(
        cls=classify(msg.data, info.height, info.width),
        origin=(info.origin.position.x, info.origin.position.y),
        resolution=info.resolution,
    )


def cell_offset(src, dst):
    """(drow, dcol) such that src cell (r, c) is dst cell (r + drow, c + dcol).

    Raises ValueError if the resolutions differ or the origins are not on a
    common lattice (more than a tenth of a cell off), since an integer offset
    would then silently misalign the grids.
    """
    if abs(src.resolution - dst.resolution) > 1e-6:
        raise ValueError(f"resolution mismatch: {src.resolution} vs {dst.resolution}")
    res = dst.resolution
    fx = (src.origin[0] - dst.origin[0]) / res
    fy = (src.origin[1] - dst.origin[1]) / res
    dcol, drow = int(round(fx)), int(round(fy))
    if abs(fx - dcol) > 0.1 or abs(fy - drow) > 0.1:
        raise ValueError(f"origins off-lattice by ({fx - dcol:.2f}, {fy - drow:.2f}) cells")
    return drow, dcol


def overlap(src_shape, dst_shape, drow, dcol):
    """Slices (src_sl, dst_sl) of the overlapping region, or None if disjoint."""
    r0, c0 = max(0, -drow), max(0, -dcol)
    r1 = min(src_shape[0], dst_shape[0] - drow)
    c1 = min(src_shape[1], dst_shape[1] - dcol)
    if r1 <= r0 or c1 <= c0:
        return None
    src_sl = (slice(r0, r1), slice(c0, c1))
    dst_sl = (slice(r0 + drow, r1 + drow), slice(c0 + dcol, c1 + dcol))
    return src_sl, dst_sl


def resample(src, dst, fill=UNKNOWN):
    """src's classes laid onto dst's cells; cells src does not cover get `fill`."""
    out = np.full(dst.shape, fill, dtype=np.int8)
    sl = overlap(src.shape, dst.shape, *cell_offset(src, dst))
    if sl is not None:
        out[sl[1]] = src.cls[sl[0]]
    return out


def match_visible(predicted, history):
    """The visible-map snapshot the predictor expanded into `predicted`.

    The predicted map is the visible map padded by the same number of unknown
    cells on every side, so its origin and size pin down the exact snapshot.
    Returns the newest match in `history` (oldest first), or None.
    """
    rows, cols = predicted.shape
    for vis in reversed(history):
        pad_r, pad_c = rows - vis.shape[0], cols - vis.shape[1]
        if pad_r < 0 or pad_r != pad_c or pad_r % 2:
            continue
        if abs(vis.resolution - predicted.resolution) > 1e-6:
            continue
        pad = pad_r // 2
        if cell_offset(vis, predicted) == (pad, pad):
            return vis
    return None


def score(predicted, visible, gt):
    """Per-cell outcome of one prediction cycle, in `predicted`'s frame.

    Only cells that were unknown in the visible map the predictor saw, that it
    filled in, and that GT knows are scored. Returns (eval_mask, pred_cls,
    gt_cls) on predicted's grid.
    """
    vis_cls = resample(visible, predicted)
    gt_cls = resample(gt, predicted)
    pred_cls = predicted.cls
    mask = (vis_cls == UNKNOWN) & (pred_cls != UNKNOWN) & (gt_cls != UNKNOWN)
    return mask, pred_cls, gt_cls


def cycle_metrics(mask, pred_cls, gt_cls):
    """Accuracy and occupied-class precision / recall / IoU for one cycle."""
    p, g = pred_cls[mask], gt_cls[mask]
    n = int(mask.sum())
    tp = int(((p == OCCUPIED) & (g == OCCUPIED)).sum())
    fp = int(((p == OCCUPIED) & (g == FREE)).sum())
    fn = int(((p == FREE) & (g == OCCUPIED)).sum())
    nan = float("nan")
    return {
        "n_eval": n,
        "acc": float((p == g).sum()) / n if n else nan,
        "occ_precision": tp / (tp + fp) if tp + fp else nan,
        "occ_recall": tp / (tp + fn) if tp + fn else nan,
        "occ_iou": tp / (tp + fp + fn) if tp + fp + fn else nan,
    }


class AccuracyCanvas:
    """Per-cell outcome counts accumulated over cycles, on the GT's grid.

    If the GT map's geometry changes, `rebase` moves the counts onto the new
    grid so nothing scored so far is lost (cells outside it are dropped: GT
    does not know them, so they could never have been scored).
    """

    def __init__(self, gt):
        self.gt = gt
        self.counts = np.zeros((4,) + gt.shape, dtype=np.uint32)

    def rebase(self, gt):
        counts = np.zeros((4,) + gt.shape, dtype=np.uint32)
        sl = overlap(self.gt.shape, gt.shape, *cell_offset(self.gt, gt))
        if sl is not None:
            counts[(slice(None),) + sl[1]] = self.counts[(slice(None),) + sl[0]]
        self.gt, self.counts = gt, counts

    def add(self, frame, mask, pred_cls, gt_cls):
        """Add one cycle's scored cells (all arrays on `frame`'s grid)."""
        sl = overlap(frame.shape, self.gt.shape, *cell_offset(frame, self.gt))
        if sl is None:
            return
        m = mask[sl[0]]
        k = (2 * pred_cls[sl[0]].astype(np.int16) + gt_cls[sl[0]]).astype(np.int16)
        dst = self.counts[(slice(None),) + sl[1]]
        for idx in range(4):
            dst[idx] += (m & (k == idx)).astype(np.uint32)


def n_eval(counts):
    """Per cell: how many times it was scored."""
    return counts.sum(axis=0)


def accuracy_grid(counts):
    """Per cell: share of its predictions that matched GT; NaN if never scored."""
    n = n_eval(counts).astype(np.float64)
    correct = counts[PF_GF].astype(np.float64) + counts[PO_GO]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, correct / n, np.nan)


def error_balance_grid(counts):
    """Per cell: (phantom - missed) / predictions, +1 all phantom obstacles
    (predicted occupied, GT free), -1 all missed obstacles (predicted free,
    GT occupied); NaN if never scored or never wrong."""
    n = n_eval(counts).astype(np.float64)
    phantom = counts[PO_GF].astype(np.float64)
    missed = counts[PF_GO].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((n > 0) & (phantom + missed > 0), (phantom - missed) / n, np.nan)
