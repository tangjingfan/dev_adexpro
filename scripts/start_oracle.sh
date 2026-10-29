#!/bin/bash
# Start oracle: oracle predictor (fills frontier windows from a GT octomap).
# Usage: ./scripts/start_oracle.sh [world] [conda_env]  (world defaults to garage)
# Stop:  tmux kill-session -t oracle
exec "$(dirname "$0")/_start_prediction_stack.sh" "${1:-garage}" oracle "${2:-adexpro}"
