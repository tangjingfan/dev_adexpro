#!/bin/bash
# Start the greedy planner in a 4-pane tmux session.
# Usage: ./scripts/start_greedy.sh [world]  (world defaults to garage)

WORLD=${1:-garage}
source "$(dirname "$0")/_common.sh"
start_session greedy 2 2

# ── Pane 0 (top-left): Simulator + autonomy stack ──────────────────────────
run_pane 0 "$SIM_CMD"

# ── Pane 1 (bottom-left): Mapping / octomap_server ─────────────────────────
run_pane 1 "$MAPPING_CMD"

# ── Pane 2 (top-right): Greedy planner + RViz ───────────────────────────────
run_pane 2 "sleep 10; $DEACTIVATE; $SETUP; ros2 launch greedy greedy_${WORLD}.launch $(log_to log_greedy.txt)"

# ── Pane 3 (bottom-right): Free terminal ───────────────────────────────────
run_pane 3 "$FREE_CMD"

finish_session
