# Shared plumbing for scripts/start_*.sh. Source it after setting WORLD:
#
#   WORLD=${1:-garage}
#   source "$(dirname "$0")/_common.sh"
#   start_session <name> <left panes> <right panes>
#   run_pane 0 "$SIM_CMD"
#   ...
#   finish_session
#
# Every stack shares the same panes for the simulator, the mapper and a free
# terminal; only the planner pane differs.

# REPO = the repo root (where these scripts live, under scripts/), which is
# also the colcon workspace (WS). Run output (node logs, online-train logs) goes to
# dev_results/experiments/.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$REPO"
EXPERIMENTS_DIR="$REPO/dev_results/experiments"
LOG_DIR="$EXPERIMENTS_DIR/node_logs"
mkdir -p "$LOG_DIR"
DEACTIVATE="conda deactivate 2>/dev/null; conda deactivate 2>/dev/null"
case "$(basename "$SHELL")" in zsh) EXT=zsh;; bash) EXT=bash;; *) EXT=sh;; esac
SETUP="source /opt/ros/humble/setup.$EXT; source $WS/install/setup.$EXT; export EXPRO_EXPERIMENTS_DIR=$EXPERIMENTS_DIR"

SIM_CMD="$DEACTIVATE; $SETUP; ros2 launch vehicle_simulator system_${WORLD}.launch"
MAPPING_CMD="sleep 8; $DEACTIVATE; $SETUP; ros2 launch simulator mapping_${WORLD}.launch"
FREE_CMD="$DEACTIVATE; $SETUP"

# Redirect for a pane command: log_to log_x.txt -> > "<LOG_DIR>/log_x.txt" 2>&1
log_to() { echo "> \"$LOG_DIR/$1\" 2>&1"; }

# The <base>.launch.py launch file, or <base>.matterport.launch.py on matterport.
world_launch_py() {
  if [ "$WORLD" = "matterport" ]; then echo "$1.matterport.launch.py"; else echo "$1.launch.py"; fi
}

# Fresh tmux session SESSION=<name>: two columns, the left split into <left>
# panes and the right into <right>. Panes are numbered top to bottom, left
# column first.
start_session() {
  SESSION=$1
  local left=$2 right=$3 i
  tmux kill-session -t "$SESSION" 2>/dev/null
  tmux new-session -d -s "$SESSION" -x 220 -y 50
  tmux split-window -h -t "$SESSION"
  for ((i = 1; i < left; i++)); do
    tmux select-pane -t "$SESSION:0.$((i - 1))"
    tmux split-window -v -t "$SESSION"
  done
  for ((i = 1; i < right; i++)); do
    tmux select-pane -t "$SESSION:0.$((left + i - 1))"
    tmux split-window -v -t "$SESSION"
  done
}

# Type a command into pane <index> of the current session.
run_pane() { tmux send-keys -t "$SESSION:0.$1" "$2" Enter; }

# Focus the top-left pane and attach (ATTACH=0 leaves the session detached).
finish_session() {
  tmux select-pane -t "$SESSION:0.0"
  if [ "${ATTACH:-1}" = 1 ]; then
    tmux attach-session -t "$SESSION"
  else
    echo "Session '$SESSION' started detached. Attach: tmux attach -t $SESSION"
  fi
}
