"""Per-run monitor: subscribes to robot state + visualization_tools metrics,
detects termination conditions, and writes per-run CSV.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


END_STUCK = "stuck"
END_TIMEOUT = "timeout"
END_CRASH = "crash"
END_MANUAL = "manual"

CSV_HEADER = ["t_s", "x", "y", "z", "traveled_m", "explored_volume_m3", "runtime_s"]

# Mirror visualization_tools' transInterval (0.2m): only record a sample
# once the robot has moved at least this far since the previous sample.
TRANS_INTERVAL_M = 0.2

# Displacement from the initial pose that counts as "motion started" → early
# exit from warmup. Slightly above odom noise floor.
MOTION_START_M = 0.1


@dataclass
class TerminationCfg:
    stuck_window_s: float = 10.0
    stuck_dist_m: float = 0.5
    warmup_s: float = 20.0
    odom_silence_s: float = 5.0     # crash if no odom for this long after first msg


@dataclass
class RunResult:
    end_reason: Optional[str] = None
    elapsed_s: float = 0.0
    total_traveled_m: float = 0.0
    final_explored_volume_m3: float = 0.0
    final_runtime_s: float = float("nan")
    sample_count: int = 0


@dataclass
class _RunState:
    csv_path: Path
    max_time_s: float
    cfg: TerminationCfg
    run_start_wall: float
    result: RunResult = field(default_factory=RunResult)
    # pose_window holds (odom_stamp_s, x, y) — sim/odom clock, not wall
    pose_window: Deque[Tuple[float, float, float]] = field(default_factory=deque)
    last_odom_wall: float = 0.0
    seen_odom: bool = False
    motion_started: bool = False                          # set once robot moves > MOTION_START_M from initial
    initial_pose_xy: Optional[Tuple[float, float]] = None  # first odom (x, y)
    last_sample_xy: Optional[Tuple[float, float]] = None
    last_time_duration: Optional[float] = None
    last_traveling_distance: Optional[float] = None
    last_explored_volume: Optional[float] = None
    last_runtime: Optional[float] = None
    csv_file: Optional[object] = None
    csv_writer: Optional[csv.writer] = None
    last_diag_wall: float = 0.0
    odom_msg_count: int = 0


class RunMonitor(Node):
    def __init__(self):
        super().__init__("experiment_run_monitor")
        self._lock = threading.Lock()
        self._state: Optional[_RunState] = None
        # always-on (independent of a run): last wall time any odom msg arrived
        self._last_odom_wall_any: Optional[float] = None

        # BEST_EFFORT is compatible with both reliable and best-effort publishers;
        # safe default for sim odom even though vehicle_simulator publishes RELIABLE.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Odometry, "/state_estimation",
                                 self._on_odom, odom_qos)
        self.create_subscription(Float32, "/traveling_distance",
                                 self._on_traveling_distance, 10)
        self.create_subscription(Float32, "/explored_volume",
                                 self._on_explored_volume, 10)
        self.create_subscription(Float32, "/runtime",
                                 self._on_runtime, 10)
        self.create_subscription(Float32, "/time_duration",
                                 self._on_time_duration, 10)

        self.create_timer(0.5, self._on_timer)

    # ── public API ─────────────────────────────────────────────────────────
    def begin_run(self, csv_path: Path, max_time_s: float, cfg: TerminationCfg) -> None:
        with self._lock:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(csv_path, "w", newline="")
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            self._state = _RunState(
                csv_path=csv_path,
                max_time_s=max_time_s,
                cfg=cfg,
                run_start_wall=time.monotonic(),
                csv_file=f,
                csv_writer=writer,
                last_diag_wall=time.monotonic(),
            )
        self.get_logger().info(
            f"begin_run: csv={csv_path.name} max_time={max_time_s:.0f}s "
            f"warmup={cfg.warmup_s:.0f}s stuck=({cfg.stuck_window_s:.0f}s/"
            f"{cfg.stuck_dist_m:.2f}m)"
        )

    def end_run(self, reason: Optional[str] = None) -> RunResult:
        with self._lock:
            st = self._state
            if st is None:
                return RunResult(end_reason=reason or END_MANUAL)
            if st.result.end_reason is None:
                st.result.end_reason = reason or END_MANUAL
            st.result.elapsed_s = time.monotonic() - st.run_start_wall
            if st.last_traveling_distance is not None:
                st.result.total_traveled_m = st.last_traveling_distance
            if st.last_explored_volume is not None:
                st.result.final_explored_volume_m3 = st.last_explored_volume
            if st.last_runtime is not None:
                st.result.final_runtime_s = st.last_runtime
            if st.csv_file is not None:
                st.csv_file.flush()
                st.csv_file.close()
            result = st.result
            self._state = None
            return result

    def get_end_reason(self) -> Optional[str]:
        with self._lock:
            return None if self._state is None else self._state.result.end_reason

    def odom_seen_within(self, seconds: float) -> bool:
        """True if any /state_estimation msg arrived in the last `seconds`."""
        with self._lock:
            t = self._last_odom_wall_any
        return t is not None and (time.monotonic() - t) < seconds

    def wait_for_odom(self, timeout_s: float, poll_s: float = 0.2) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.odom_seen_within(1.0):
                return True
            time.sleep(poll_s)
        return False

    def wait_for_no_odom(self, timeout_s: float, quiet_s: float = 3.0,
                         poll_s: float = 0.5) -> bool:
        """Wait until no odom message has arrived for `quiet_s` seconds."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.odom_seen_within(quiet_s):
                return True
            time.sleep(poll_s)
        return False

    # ── callbacks ──────────────────────────────────────────────────────────
    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            now_wall = time.monotonic()
            self._last_odom_wall_any = now_wall
            st = self._state
            if st is None:
                return
            stamp_s = (msg.header.stamp.sec
                       + msg.header.stamp.nanosec * 1e-9)
            st.seen_odom = True
            st.last_odom_wall = now_wall
            st.odom_msg_count += 1
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z

            # stuck-detection sliding window keyed on odom (sim) timestamp.
            # Keep the oldest entry that is *just before* cutoff so that
            # `window_span >= stuck_window_s` is actually reachable (otherwise
            # floating-point + dense sampling keeps window_span strictly < it).
            st.pose_window.append((stamp_s, x, y))
            cutoff = stamp_s - st.cfg.stuck_window_s
            while len(st.pose_window) >= 2 and st.pose_window[1][0] < cutoff:
                st.pose_window.popleft()

            # Detect first real motion → early-exit warmup
            if st.initial_pose_xy is None:
                st.initial_pose_xy = (x, y)
            elif not st.motion_started:
                dx0 = x - st.initial_pose_xy[0]
                dy0 = y - st.initial_pose_xy[1]
                if dx0 * dx0 + dy0 * dy0 >= MOTION_START_M ** 2:
                    st.motion_started = True

            # CSV row only when robot has moved >= TRANS_INTERVAL_M
            # (mirrors visualization_tools.transInterval)
            if st.last_sample_xy is None:
                should_sample = True
            else:
                dx = x - st.last_sample_xy[0]
                dy = y - st.last_sample_xy[1]
                should_sample = (dx * dx + dy * dy) >= TRANS_INTERVAL_M ** 2

            # Skip writes until visualization_tools' /time_duration starts
            # flowing (i.e., its `systemInited` has triggered) — otherwise t_s
            # uses a wall-time fallback that's discontinuous with later rows
            # and the metric columns are still empty.
            if (should_sample and st.csv_writer is not None
                    and st.last_time_duration is not None):
                st.csv_writer.writerow([
                    f"{st.last_time_duration:.3f}",
                    f"{x:.4f}",
                    f"{y:.4f}",
                    f"{z:.4f}",
                    "" if st.last_traveling_distance is None
                       else f"{st.last_traveling_distance:.4f}",
                    "" if st.last_explored_volume is None
                       else f"{st.last_explored_volume:.4f}",
                    "" if st.last_runtime is None else f"{st.last_runtime:.6f}",
                ])
                st.last_sample_xy = (x, y)
                st.result.sample_count += 1

    def _on_traveling_distance(self, msg: Float32) -> None:
        with self._lock:
            if self._state is not None:
                self._state.last_traveling_distance = float(msg.data)

    def _on_explored_volume(self, msg: Float32) -> None:
        with self._lock:
            if self._state is not None:
                self._state.last_explored_volume = float(msg.data)

    def _on_runtime(self, msg: Float32) -> None:
        with self._lock:
            if self._state is not None:
                self._state.last_runtime = float(msg.data)

    def _on_time_duration(self, msg: Float32) -> None:
        with self._lock:
            if self._state is not None:
                self._state.last_time_duration = float(msg.data)

    # ── periodic termination check ─────────────────────────────────────────
    def _on_timer(self) -> None:
        log_msg: Optional[str] = None
        with self._lock:
            st = self._state
            if st is None or st.result.end_reason is not None:
                return
            now = time.monotonic()
            elapsed = now - st.run_start_wall

            window_span = (st.pose_window[-1][0] - st.pose_window[0][0]
                           if len(st.pose_window) >= 2 else 0.0)
            max_d = 0.0
            if len(st.pose_window) >= 2:
                xs = [p[1] for p in st.pose_window]
                ys = [p[2] for p in st.pose_window]
                max_d = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

            # Warmup ends at the earlier of: elapsed >= warmup_s, or first motion
            warmup_active = (elapsed < st.cfg.warmup_s) and not st.motion_started

            # diagnostic snapshot every 2.5s wall time
            if now - st.last_diag_wall >= 2.5:
                st.last_diag_wall = now
                odom_age = (now - st.last_odom_wall) if st.seen_odom else -1.0
                log_msg = (
                    f"[INFO] elapsed={elapsed:5.1f}s "
                    f"warmup={'y' if warmup_active else 'N'} "
                    f"motion={'y' if st.motion_started else 'n'} "
                    f"odom_msgs={st.odom_msg_count} "
                    f"odom_age={odom_age:.2f}s "
                    f"win=({len(st.pose_window)} pts, {window_span:.2f}s) "
                    f"bbox_diag={max_d:.3f}m"
                )

            if elapsed > st.max_time_s:
                st.result.end_reason = END_TIMEOUT
            elif warmup_active:
                pass
            elif (st.seen_odom and
                    now - st.last_odom_wall > st.cfg.odom_silence_s):
                st.result.end_reason = END_CRASH
            elif (window_span >= st.cfg.stuck_window_s
                    and max_d < st.cfg.stuck_dist_m):
                st.result.end_reason = END_STUCK

        if log_msg is not None:
            self.get_logger().info(log_msg)


def make_node_and_executor() -> Tuple[RunMonitor, "rclpy.executors.SingleThreadedExecutor"]:
    """Helper: create the node + a dedicated executor (spin in a thread)."""
    from rclpy.executors import SingleThreadedExecutor
    node = RunMonitor()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    return node, executor
