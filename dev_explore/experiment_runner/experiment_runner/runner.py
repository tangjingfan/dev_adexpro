"""Main loop: parse YAML, expand tasks, drive launcher + monitor per run."""

from __future__ import annotations

import argparse
import csv
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import rclpy
import yaml

from .baseline_registry import BASELINES
from .run_monitor import (
    END_MANUAL,
    RunMonitor,
    TerminationCfg,
    make_node_and_executor,
)
from .tmux_launcher import TmuxLauncher


SUMMARY_HEADER = [
    "world", "baseline", "run_idx", "end_reason", "elapsed_s",
    "total_traveled_m", "final_explored_volume_m3", "final_runtime_s",
    "sample_count", "csv_path",
]


@dataclass
class Task:
    world: str
    baseline: str
    run_idx: int
    max_time_s: float


def _expand_tasks(cfg: dict) -> List[Task]:
    """Order: (baseline x repeats) x worlds — group runs of the same baseline."""
    tasks: List[Task] = []
    for entry in cfg.get("runs", []):
        baseline = entry["baseline"]
        world = entry["world"]
        repeats = int(entry.get("repeats", 1))
        max_time_s = float(entry["max_time_s"])
        if baseline not in BASELINES:
            raise ValueError(
                f"Unknown baseline '{baseline}'. Known: {sorted(BASELINES)}")
        for idx in range(repeats):
            tasks.append(Task(world, baseline, idx, max_time_s))
    return tasks


def _load_cfg(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _termination_cfg(cfg: dict) -> TerminationCfg:
    t = cfg.get("termination", {}) or {}
    return TerminationCfg(
        stuck_window_s=float(t.get("stuck_window_s", 10.0)),
        stuck_dist_m=float(t.get("stuck_dist_m", 0.5)),
        warmup_s=float(t.get("warmup_s", 20.0)),
        odom_silence_s=float(t.get("odom_silence_s", 5.0)),
    )


def _run_tag(task: Task, timestamp: str) -> str:
    return f"{task.baseline}_{task.world}_r{task.run_idx}_{timestamp}"


def _open_summary(summary_path: Path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new = not summary_path.exists()
    f = open(summary_path, "a", newline="")
    writer = csv.writer(f)
    if new:
        writer.writerow(SUMMARY_HEADER)
        f.flush()
    return f, writer


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                        help="Workspace root (where install/ lives)")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args.config)
    output_dir = Path(cfg.get("output_dir", "dev_visualization/experiments"))
    if not output_dir.is_absolute():
        output_dir = (args.workspace / output_dir).resolve()
    log_dir = output_dir / "logs"
    runs_dir = output_dir / "runs"
    summary_path = output_dir / "summary.csv"

    term_cfg = _termination_cfg(cfg)
    tasks = _expand_tasks(cfg)
    if not tasks:
        print("[runner] No tasks in config; nothing to do.", file=sys.stderr)
        return 1

    print(f"[runner] {len(tasks)} run(s) queued; results → {output_dir}")

    rclpy.init()
    monitor, executor = make_node_and_executor()
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    launcher = TmuxLauncher(workspace_root=args.workspace, log_dir=log_dir,
                            experiments_dir=output_dir)
    summary_file, summary_writer = _open_summary(summary_path)

    interrupted = threading.Event()

    def _sigint(_signum, _frame):
        if not interrupted.is_set():
            print("\n[runner] SIGINT — stopping after current run.")
            interrupted.set()
        else:
            print("\n[runner] SIGINT (2nd) — hard exit.")
            sys.exit(130)

    signal.signal(signal.SIGINT, _sigint)

    try:
        for task_idx, task in enumerate(tasks, start=1):
            if interrupted.is_set():
                break
            spec = BASELINES[task.baseline]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session = f"er_{spec.name}_{task.world}_r{task.run_idx}_{timestamp}"
            run_tag = _run_tag(task, timestamp)
            csv_path = (runs_dir / task.world / task.baseline
                        / f"run_{task.run_idx}_{timestamp}.csv")

            print(f"\n[runner] ── [{task_idx}/{len(tasks)}] "
                  f"{task.baseline} on {task.world} (run {task.run_idx}, "
                  f"max {task.max_time_s:.0f}s) ──")

            _run_one(
                monitor=monitor,
                launcher=launcher,
                spec=spec,
                task=task,
                session=session,
                run_tag=run_tag,
                csv_path=csv_path,
                term_cfg=term_cfg,
                summary_writer=summary_writer,
                summary_file=summary_file,
                interrupted=interrupted,
            )
    finally:
        summary_file.close()
        executor.shutdown()
        monitor.destroy_node()
        rclpy.shutdown()

    print(f"\n[runner] Done. Summary: {summary_path}")
    return 0


def _run_one(*, monitor: RunMonitor, launcher: TmuxLauncher, spec, task: Task,
             session: str, run_tag: str, csv_path: Path,
             term_cfg: TerminationCfg, summary_writer, summary_file,
             interrupted: threading.Event) -> None:
    end_reason = None
    result = None

    try:
        launcher.start(spec, task.world, session, run_tag)
        print(f"[runner]   tmux session: {session}  (attach to watch)")

        print("[runner]   waiting for /state_estimation ...")
        if not monitor.wait_for_odom(timeout_s=60.0):
            print("[runner]   no odom in 60s — marking crash, killing session.")
            end_reason = "launch_failed"
            return

        print("[runner]   odom detected — start run")
        monitor.begin_run(csv_path=csv_path,
                          max_time_s=task.max_time_s,
                          cfg=term_cfg)

        # Block until monitor decides the run is over (or SIGINT)
        while True:
            reason = monitor.get_end_reason()
            if reason is not None:
                end_reason = reason
                break
            if interrupted.is_set():
                end_reason = END_MANUAL
                break
            time.sleep(0.25)

        print(f"[runner]   end_reason = {end_reason}")

    finally:
        result = monitor.end_run(reason=end_reason)
        launcher.kill(session)
        # Wait for ROS graph to clear before next run
        if monitor.wait_for_no_odom(timeout_s=20.0, quiet_s=3.0):
            print("[runner]   stack torn down cleanly.")
        else:
            print("[runner]   warning: odom still flowing after 20s — "
                  "continuing anyway.")

        if result is not None:
            summary_writer.writerow([
                task.world,
                task.baseline,
                task.run_idx,
                result.end_reason or "unknown",
                f"{result.elapsed_s:.2f}",
                f"{result.total_traveled_m:.3f}",
                f"{result.final_explored_volume_m3:.3f}",
                (f"{result.final_runtime_s:.6f}"
                 if result.final_runtime_s == result.final_runtime_s else ""),
                result.sample_count,
                str(csv_path),
            ])
            summary_file.flush()


if __name__ == "__main__":
    sys.exit(main())
