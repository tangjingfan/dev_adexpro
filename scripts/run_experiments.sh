#!/bin/bash
# Batch experiments with experiment_runner.
# Usage: ./scripts/run_experiments.sh <config>
#   config: a YAML path, or a name under dev_explore/experiment_runner/config/
#           (e.g. indoor_self_adapt)
# Results go to dev_results/experiments/ (the configs' output_dir).

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="$REPO"

CFG=${1:?config required (path or name under dev_explore/experiment_runner/config/)}
if [ ! -f "$CFG" ]; then
  CFG="$WS/dev_explore/experiment_runner/config/${CFG%.yaml}.yaml"
fi
if [ ! -f "$CFG" ]; then
  echo "Config not found: $1" >&2
  exit 1
fi
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"

# The runner is a plain rclpy node: system Python, no conda.
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
# --workspace: install/ for the panes it launches, and the base the configs'
# relative output_dir (dev_results/experiments) resolves against.
exec ros2 run experiment_runner runner "$CFG" --workspace "$WS"
