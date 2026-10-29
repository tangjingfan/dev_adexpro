"""Spawn and tear down a tmux session that runs a baseline stack."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .baseline_registry import BaselineSpec, LaunchPane


SIMULATOR_CMD_TEMPLATE = "ros2 launch vehicle_simulator system_{world}.launch"


def _setup_ext() -> str:
    """Pick ROS setup script extension matching the user's login shell.

    tmux panes open with $SHELL, so we must source the matching variant —
    setup.zsh is not valid bash and vice versa.
    """
    shell = os.environ.get("SHELL", "")
    base = os.path.basename(shell)
    if base == "zsh":
        return "zsh"
    if base == "bash":
        return "bash"
    return "sh"  # POSIX fallback


def _run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _build_pane_cmd(
    pane: LaunchPane,
    world: str,
    workspace_root: Path,
    log_path: Optional[Path],
    experiments_dir: Optional[Path] = None,
) -> str:
    """Build the shell command sent to a tmux pane.

    Mirrors the structure of start_<baseline>.sh: optional sleep, deactivate
    conda, source ROS + workspace, optional conda activate, then the launch.
    """
    ext = _setup_ext()
    parts: List[str] = []
    if pane.delay_s > 0:
        parts.append(f"sleep {pane.delay_s:.1f}")
    parts.append("conda deactivate 2>/dev/null; conda deactivate 2>/dev/null")
    parts.append(f"source /opt/ros/humble/setup.{ext}")
    parts.append(
        f"source {shlex.quote(str(workspace_root / 'install' / f'setup.{ext}'))}"
    )
    if pane.conda_env:
        parts.append(f"conda activate {pane.conda_env}")
    # Nodes that write their own logs (self_adapt's online_train/) resolve
    # relative paths against this, so they land beside the runner's output.
    if experiments_dir is not None:
        parts.append(f"export EXPRO_EXPERIMENTS_DIR={shlex.quote(str(experiments_dir))}")

    launch_cmd = pane.cmd_template.format(world=world)
    if log_path is not None:
        launch_cmd = f"{launch_cmd} > {shlex.quote(str(log_path))} 2>&1"
    parts.append(launch_cmd)
    return "; ".join(parts)


class TmuxLauncher:
    def __init__(self, workspace_root: Path, log_dir: Path,
                 experiments_dir: Optional[Path] = None):
        self.workspace_root = workspace_root.resolve()
        self.log_dir = log_dir.resolve()
        self.experiments_dir = experiments_dir.resolve() if experiments_dir else None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def session_exists(self, session: str) -> bool:
        return _run(["tmux", "has-session", "-t", session]).returncode == 0

    def start(
        self,
        spec: BaselineSpec,
        world: str,
        session: str,
        run_tag: str,
    ) -> List[Path]:
        """Create the tmux session and dispatch all panes. Returns log paths."""
        if self.session_exists(session):
            self.kill(session)

        _run(["tmux", "new-session", "-d", "-s", session, "-x", "220", "-y", "50"],
             check=True)

        panes: List[LaunchPane] = []
        panes.append(LaunchPane(
            label="simulator",
            cmd_template=SIMULATOR_CMD_TEMPLATE,
            delay_s=0.0,
        ))
        if spec.mapping is not None:
            panes.append(spec.mapping)
        panes.extend(spec.extras)
        panes.append(spec.planner)

        # Create N-1 additional panes by tiled split.
        for _ in range(len(panes) - 1):
            _run(["tmux", "split-window", "-t", session], check=True)
            _run(["tmux", "select-layout", "-t", session, "tiled"], check=True)

        log_paths: List[Path] = []
        for idx, pane in enumerate(panes):
            log_file = self.log_dir / f"{run_tag}__{pane.label}.txt"
            log_paths.append(log_file)
            cmd_str = _build_pane_cmd(
                pane=pane,
                world=world,
                workspace_root=self.workspace_root,
                log_path=log_file,
                experiments_dir=self.experiments_dir,
            )
            _run([
                "tmux", "send-keys", "-t", f"{session}:0.{idx}",
                cmd_str, "Enter",
            ], check=True)

        return log_paths

    def kill(self, session: str) -> None:
        """Tear down session + best-effort process cleanup."""
        _run(["tmux", "kill-session", "-t", session])
        # belt-and-suspenders cleanup for processes that survive session kill
        for pattern in (
            "ros2 launch vehicle_simulator",
            "ros2 launch simulator",
            "ros2 launch tare_planner",
            "ros2 launch nbvplanner",
            "ros2 launch greedy",
            "ros2 launch ariadne",
            "ros2 launch cogniplan",
            "ros2 launch mapex",
            "ros2 launch gbplanner",
            "vehicle_simulator",
            "visualizationTools",
            "octomap_server",
            "gzserver",
            "gzclient",
        ):
            _run(["pkill", "-f", pattern])
        time.sleep(5.0)
