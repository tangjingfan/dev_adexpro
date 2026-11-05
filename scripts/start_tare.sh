#!/bin/bash
# Start TARE in a 4-pane tmux session.
# Usage: ./scripts/start_tare.sh [world]  (world defaults to garage)

WORLD=${1:-garage}
source "$(dirname "$0")/_common.sh"
start_session tare 2 2

# ── Pane 0 (top-left): Simulator + autonomy stack ──────────────────────────
run_pane 0 "$SIM_CMD"

# ── Pane 1 (bottom-left): Mapping / octomap_server ─────────────────────────
# TARE plans without this map; it runs only to record the occupancy grid.
run_pane 1 "$MAPPING_CMD"

# ── Pane 2 (top-right): TARE planner ────────────────────────────────────────
run_pane 2 "sleep 8; $DEACTIVATE; $SETUP; ros2 launch tare_planner explore_${WORLD}.launch $(log_to log_tare.txt)"

# ── Pane 3 (bottom-right): Free terminal ───────────────────────────────────
run_pane 3 "$FREE_CMD"

finish_session
