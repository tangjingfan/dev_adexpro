#!/bin/bash
# Shared implementation behind start_adexpro.sh / start_expro.sh / start_oracle.sh:
# the full stack in a tmux session named after the prediction METHOD:
#   adexpro learned FPUnet prediction + online update  (map_prediction_online_update.launch)
#   expro   learned FPUnet prediction, no online update (map_prediction.launch)
#   oracle  GT-filled frontier windows                  (oracle_map_prediction.launch)
#
# Every method also gets a GT octomap (/gt_map), prediction_eval, which scores
# each /predicted_map against it (dev_results/experiments/prediction_eval/), and
# prediction_heatmap's live accuracy window.
#
# Usage: ./scripts/_start_prediction_stack.sh <world> <method> [conda_env]
#   conda_env defaults to adexpro. Set ATTACH=0 to leave the session detached,
#   EVAL=0 to skip prediction_eval and the window (the oracle still gets its GT octomap).

WORLD=${1:?world required}
METHOD=${2:?method required}
CONDA_ENV=${3:-adexpro}

# Per-method prediction launch file. Everything else is shared, so the methods
# can never drift apart the way near-identical copies did.
case "$METHOD" in
  adexpro) PRED_LAUNCH="map_prediction_online_update.launch" ;;
  expro)   PRED_LAUNCH="map_prediction.launch" ;;
  oracle)  PRED_LAUNCH="oracle_map_prediction.launch" ;;
  *)
    echo "Unknown method '$METHOD' (expected: adexpro | expro | oracle)" >&2
    exit 1
    ;;
esac
source "$(dirname "$0")/_common.sh"
start_session "$METHOD" 3 3

# ── Pane 0 (top-left): Simulator + autonomy stack ──────────────────────────
run_pane 0 "$SIM_CMD"

# ── Pane 1 (mid-left): Mapping / octomap_server ────────────────────────────
run_pane 1 "$MAPPING_CMD"

# ── Pane 2 (bottom-left): Map prediction ($METHOD, conda: $CONDA_ENV) ──
# The oracle backend needs /gt_map (started in pane 4) to be up first, so it
# waits a little longer than the learned backends.
if [ "$METHOD" = oracle ]; then PRED_SLEEP=12; else PRED_SLEEP=10; fi
run_pane 2 "sleep $PRED_SLEEP; conda activate $CONDA_ENV; $SETUP; ros2 launch simulator $PRED_LAUNCH $(log_to "log_prediction_${METHOD}.txt")"

# ── Pane 3 (top-right): Planner + RViz ─────────────────────────────────────
run_pane 3 "sleep 10; $DEACTIVATE; $SETUP; ros2 launch simulator planner_${WORLD}.launch $(log_to "log_${METHOD}.txt")"

# ── Pane 4 (mid-right): GT octomap (/overall_map → /gt_map) + prediction_eval
#    + the live heatmap window.
# The oracle backend also fills its windows from this /gt_map.
if [ "${EVAL:-1}" != 0 ]; then
  run_pane 4 "sleep 8; $DEACTIVATE; $SETUP; ros2 launch prediction_heatmap prediction_heatmap.launch world:=$WORLD method:=$METHOD $(log_to "log_prediction_eval_${METHOD}.txt")"
elif [ "$METHOD" = oracle ]; then
  run_pane 4 "sleep 8; $DEACTIVATE; $SETUP; ros2 launch simulator gt_mapping_${WORLD}.launch"
else
  run_pane 4 "$FREE_CMD"
fi

# ── Pane 5 (bottom-right): Free terminal ───────────────────────────────────
run_pane 5 "$FREE_CMD"

finish_session
