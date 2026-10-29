#!/bin/bash
# Start AdExPro: ExPro + online self-supervised update of the FPUnet predictor.
# Usage: ./scripts/start_adexpro.sh [world] [conda_env]  (world defaults to garage)
# Stop:  tmux kill-session -t adexpro
exec "$(dirname "$0")/_start_prediction_stack.sh" "${1:-garage}" adexpro "${2:-adexpro}"
