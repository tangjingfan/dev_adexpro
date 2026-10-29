"""Accuracy / error-type heatmap panels drawn by heatmap_node.

Only imports matplotlib's object API: the caller picks the backend. Inputs are
the per-cell grids from prediction_eval.grids (accuracy_grid,
error_balance_grid, n_eval), NaN where a cell has nothing to show.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from prediction_eval import grids

# Light-mode palette (dataviz reference instance).
SURFACE = "#fcfcfb"
GT_UNKNOWN = "#e6e5e0"
GT_OCCUPIED = "#52514e"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"
ROBOT = "#eb6834"
ACCURACY_CMAP = LinearSegmentedColormap.from_list(
    "accuracy", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "diverging", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2423"])
BACKGROUND_CMAP = ListedColormap([SURFACE, GT_UNKNOWN, GT_OCCUPIED])


def masked(values, n, min_count):
    """values with NaNs and cells scored fewer than min_count times masked."""
    return np.ma.masked_where(~np.isfinite(values) | (n < min_count), values)


def weighted_accuracy(acc, n, min_count):
    keep = np.isfinite(acc) & (n >= min_count)
    return float((acc[keep] * n[keep]).sum() / n[keep].sum()) if keep.any() else float("nan")


def extent(g):
    x0, y0 = g.origin
    return (x0, x0 + g.shape[1] * g.resolution, y0, y0 + g.shape[0] * g.resolution)


def background(cls):
    """GT classes as context: 0 free (surface), 1 unknown, 2 obstacle."""
    bg = np.zeros(cls.shape, dtype=np.int8)
    bg[cls == grids.UNKNOWN] = 1
    bg[cls == grids.OCCUPIED] = 2
    return bg


def style_axes(ax):
    ax.set_aspect("equal")
    ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=2)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xlabel("x (m)", color=TEXT_MUTED, fontsize=8)
    ax.set_ylabel("y (m)", color=TEXT_MUTED, fontsize=8)


def add_colorbar(fig, im, ax, label, ticks=None, ticklabels=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=2)
    if ticklabels:
        cb.ax.set_yticklabels(ticklabels)
    cb.set_label(label, color=TEXT_MUTED, fontsize=8)


class HeatmapPanels:
    """The accuracy and error-type panels on a pair of axes, redrawn in place.

    Images are created on the first `update` and afterwards only get new data
    (and a new extent if the grid moved), so a live window stays cheap.
    """

    def __init__(self, fig, ax_acc, ax_err, title, min_count=3):
        self.fig, self.ax_acc, self.ax_err = fig, ax_acc, ax_err
        self.title, self.min_count = title, min_count
        self._images = None
        for ax in (ax_acc, ax_err):
            style_axes(ax)
        self.ax_err.set_title("Error type (cells with at least one mistake)",
                              color=TEXT, fontsize=10, loc="left")
        self._set_acc_title(float("nan"))

    def _set_acc_title(self, mean):
        self.ax_acc.set_title(f"{self.title}\nPrediction accuracy vs GT (weighted mean {mean:.2f})",
                              color=TEXT, fontsize=10, loc="left")

    def _create(self, ext, bg, acc, bal):
        common = dict(origin="lower", extent=ext, interpolation="nearest")
        bg_acc = self.ax_acc.imshow(bg, cmap=BACKGROUND_CMAP, vmin=0, vmax=2, **common)
        bg_err = self.ax_err.imshow(bg, cmap=BACKGROUND_CMAP, vmin=0, vmax=2, **common)
        im_acc = self.ax_acc.imshow(acc, cmap=ACCURACY_CMAP, vmin=0, vmax=1, **common)
        im_err = self.ax_err.imshow(bal, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, **common)
        (traj,) = self.ax_acc.plot([], [], color=TEXT, lw=1.0, alpha=0.6)
        (start,) = self.ax_acc.plot([], [], "o", ms=8, color=TEXT, mec=SURFACE, mew=2)
        (robot,) = self.ax_acc.plot([], [], "o", ms=8, color=ROBOT, mec=SURFACE, mew=2)
        add_colorbar(self.fig, im_acc, self.ax_acc, f"accuracy (cells predicted ≥ {self.min_count}×)")
        add_colorbar(self.fig, im_err, self.ax_err, "(phantom − missed) / predictions",
                     ticks=[-1, 0, 1], ticklabels=["missed obstacle", "balanced", "phantom obstacle"])
        self._images = (bg_acc, bg_err, im_acc, im_err)
        self._lines = (traj, start, robot)
        self._extent = ext

    def update(self, grid, gt_cls, acc, bal, n, trajectory=None, robot_xy=None):
        """Redraw. grid: a grids.Grid giving the geometry; gt_cls, acc, bal, n:
        per-cell arrays on it; trajectory: (N, 2+) array whose last two
        columns are x, y; robot_xy: current position."""
        ext = extent(grid)
        bg = background(gt_cls)
        acc_m = masked(acc, n, self.min_count)
        bal_m = masked(bal, n, self.min_count)
        if self._images is None:
            self._create(ext, bg, acc_m, bal_m)
        else:
            bg_acc, bg_err, im_acc, im_err = self._images
            for im, data in ((bg_acc, bg), (bg_err, bg), (im_acc, acc_m), (im_err, bal_m)):
                im.set_data(data)
            if ext != self._extent:
                for im in self._images:
                    im.set_extent(ext)
                for ax in (self.ax_acc, self.ax_err):
                    ax.set_xlim(ext[0], ext[1])
                    ax.set_ylim(ext[2], ext[3])
                self._extent = ext
        traj_line, start, robot = self._lines
        if trajectory is not None and len(trajectory):
            traj_line.set_data(trajectory[:, -2], trajectory[:, -1])
            start.set_data([trajectory[0, -2]], [trajectory[0, -1]])
        if robot_xy is not None:
            robot.set_data([robot_xy[0]], [robot_xy[1]])
        self._set_acc_title(weighted_accuracy(acc, n, self.min_count))
