#!/bin/bash
# Start ExPro: FPUnet predictor, no online update.
# Usage: ./scripts/start_expro.sh [world] [conda_env]  (world defaults to garage)
# Stop:  tmux kill-session -t expro
exec "$(dirname "$0")/_start_prediction_stack.sh" "${1:-garage}" expro "${2:-adexpro}"
