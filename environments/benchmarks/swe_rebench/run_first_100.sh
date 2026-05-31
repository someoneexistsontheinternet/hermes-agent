#!/bin/bash

set -euo pipefail

ROOT="/home/ggb/tinker_hermes_agent_test"
REPO="$ROOT/hermes-agent"
PYTHON="$ROOT/.venv/bin/python"
RUNNER="$REPO/environments/benchmarks/swe_rebench/run_mvp.py"
DATASET="$ROOT/datasets/SWE-rebench-V2_train.jsonl"
MODE="${SWE_REBENCH_MODE:-no_web}"

mkdir -p "$ROOT/logs/swe_rebench_eval"
LOG_FILE="$ROOT/logs/swe_rebench_eval/swe_rebench_first100_$(date +%Y%m%d_%H%M%S).log"

export PYTHONUNBUFFERED=1

"$PYTHON" "$RUNNER" \
  --dataset-path "$DATASET" \
  --max-samples 100 \
  --mode "$MODE" \
  "$@" 2>&1 | tee "$LOG_FILE"

echo
echo "Log saved to: $LOG_FILE"
