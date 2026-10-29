#!/bin/bash
# Start MapEx in a 4-pane tmux session.
# Usage: ./scripts/start_mapex.sh [world]  (world defaults to garage)

WORLD=${1:-garage}
source "$(dirname "$0")/_common.sh"
start_session mapex 2 2

# ── Pane 0 (top-left): Simulator + autonomy stack ──────────────────────────
run_pane 0 "$SIM_CMD"

# ── Pane 1 (bottom-left): Mapping / octomap_server ─────────────────────────
run_pane 1 "$MAPPING_CMD"

# ── Pane 2 (top-right): MapEx planner + RViz ────────────────────────────────
run_pane 2 "sleep 10; $DEACTIVATE; $SETUP; ros2 launch mapex $(world_launch_py mapex) $(log_to log_mapex.txt)"

# ── Pane 3 (bottom-right): Free terminal ───────────────────────────────────
run_pane 3 "$FREE_CMD"

finish_session
