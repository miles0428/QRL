#!/usr/bin/env bash
# DQN with full observation PLUS an action/observation history stack.
#
# Motivation: the noise is AR(1) with phi=0.95 and a 20-step correlation time,
# and the greedy controllers ignore it entirely (their lookahead assumes zero
# noise). A window of (obs, action-taken) is what an agent needs to estimate the
# current noise field from how the state responded to its own past drives, which
# is the one edge available over greedy.
#
#   PY=/path/to/python bash scripts/run_full_hist.sh [steps]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
STEPS="${1:-150000}"

mkdir -p logs results

pids=()
for nr in 0.5 0.35; do
  for seed in 0 1 2; do
    log="logs/dqn_full_hist_nr${nr}_seed${seed}.log"
    "$PY" -u scripts/train_dqn.py --obs full_hist --seed "$seed" \
        --noise-rabi "$nr" --steps "$STEPS" > "$log" 2>&1 &
    pids+=($!)
    echo "launched full_hist nr=$nr seed$seed -> $log"
  done
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "FULL_HIST DONE (failures: $fail)"
exit $fail
