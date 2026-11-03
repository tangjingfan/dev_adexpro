#!/bin/bash
# Start oexp: GT predictor (fills frontier windows from a GT octomap).
# Usage: ./scripts/start_oexp.sh [world] [conda_env]  (world defaults to garage)
# Stop:  tmux kill-session -t oexp
exec "$(dirname "$0")/_start_prediction_stack.sh" "${1:-garage}" oexp "${2:-adexpro}"
