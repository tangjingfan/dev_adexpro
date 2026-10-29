#!/usr/bin/env python3
"""Offline plotter for experiment_runner results.

Reads <output_dir>/summary.csv plus the per-run CSVs and plots, per (world, metric),
one curve per baseline averaged across repeats with a shaded ±1σ band.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("explored_volume_m3", "Explored Volume (m³)"),
    ("traveled_m", "Traveled Distance (m)"),
]


def _read_run_csv(path: Path) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[float]] = defaultdict(list)
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = float(row["t_s"])
            except (ValueError, KeyError):
                continue
            cols["t_s"].append(t)
            for k in ("traveled_m", "explored_volume_m3", "runtime_s"):
                v = row.get(k, "")
                cols[k].append(float(v) if v not in ("", None) else np.nan)
    return {k: np.asarray(v) for k, v in cols.items()}


def _resample(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step interpolation onto common time grid (carry-forward last value)."""
    if len(t) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.where(idx >= 0, y[np.clip(idx, 0, len(y) - 1)], np.nan)
    out[grid > t[-1]] = np.nan       # don't extrapolate past end of run
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path,
                        help="output_dir from runner (contains summary.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory for figures (default: results_dir/plots)")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Resampling step in seconds (default: 1.0)")
    args = parser.parse_args()

    summary_path = args.results_dir / "summary.csv"
    if not summary_path.exists():
        print(f"summary.csv not found at {summary_path}")
        return 1
    out_dir = args.out or (args.results_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # group rows by (world, baseline)
    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    with open(summary_path, "r") as f:
        for row in csv.DictReader(f):
            p = Path(row["csv_path"])
            if not p.is_absolute():
                p = (args.results_dir / p).resolve()
            if p.exists():
                groups[(row["world"], row["baseline"])].append(p)

    worlds = sorted({w for (w, _) in groups})
    for world in worlds:
        baselines = sorted({b for (w, b) in groups if w == world})
        for metric_key, metric_label in METRICS:
            fig, ax = plt.subplots(figsize=(8, 5))
            for baseline in baselines:
                runs = [_read_run_csv(p) for p in groups[(world, baseline)]]
                runs = [r for r in runs if "t_s" in r and len(r["t_s"]) > 1]
                if not runs:
                    continue
                t_max = max(r["t_s"][-1] for r in runs)
                grid = np.arange(0.0, t_max + args.dt, args.dt)
                stacked = np.vstack([
                    _resample(r["t_s"], r.get(metric_key, np.array([])), grid)
                    for r in runs
                ])
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                ax.plot(grid, mean, label=f"{baseline} (n={len(runs)})")
                ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} — {world}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig_path = out_dir / f"{world}__{metric_key}.png"
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)
            print(f"  wrote {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
