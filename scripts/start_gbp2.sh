#!/bin/bash
# Start gbplanner2 in a 4-pane tmux session.
# Usage: ./scripts/start_gbp2.sh [world]  (world defaults to garage)

WORLD=${1:-garage}
source "$(dirname "$0")/_common.sh"
start_session gbp2 2 2

# ── Pane 0 (top-left): Simulator + autonomy stack ──────────────────────────
run_pane 0 "$SIM_CMD"

# ── Pane 1 (bottom-left): Mapping / octomap_server ─────────────────────────
run_pane 1 "$MAPPING_CMD"

# ── Pane 2 (top-right): gbplanner2 + RViz ───────────────────────────────────
run_pane 2 "sleep 10; $DEACTIVATE; $SETUP; ros2 launch gbplanner $(world_launch_py gbplanner) $(log_to log_gbp2.txt)"

# ── Pane 3 (bottom-right): Free terminal ───────────────────────────────────
run_pane 3 "$FREE_CMD"

finish_session
