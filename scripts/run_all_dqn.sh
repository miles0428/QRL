#!/usr/bin/env bash
# Train DQN on all three observation modes x 3 seeds, in parallel.
#
#   PY=/path/to/python bash scripts/run_all_dqn.sh [noise_rabi] [steps]
#
# Each run is single-threaded (torch.set_num_threads(1) in train_dqn.py), so
# 9 concurrent runs sit comfortably on a many-core box.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
NR="${1:-0.5}"
STEPS="${2:-150000}"

mkdir -p logs results

pids=()
for obs in full masked masked_hist; do
  for seed in 0 1 2; do
    log="logs/dqn_${obs}_nr${NR}_seed${seed}.log"
    "$PY" scripts/train_dqn.py --obs "$obs" --seed "$seed" \
        --noise-rabi "$NR" --steps "$STEPS" > "$log" 2>&1 &
    pids+=($!)
    echo "launched $obs seed$seed -> $log"
  done
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail+1))
done

echo "ALL RUNS DONE (failures: $fail)"
exit $fail
