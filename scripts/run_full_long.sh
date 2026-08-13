#!/usr/bin/env bash
# Extend the full-observation DQN to 400k steps, 3 seeds.
#
# Every hyperparameter is identical to the 150k runs (same seeds, same epsilon
# schedule), so the first 150k steps reproduce those runs exactly and any change
# in the final number is attributable to the extra training alone.
#
#   PY=/path/to/python bash scripts/run_full_long.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
STEPS="${1:-400000}"

mkdir -p logs results

pids=()
for seed in 0 1 2; do
  log="logs/dqn_fullLong_nr0.5_seed${seed}.log"
  "$PY" -u scripts/train_dqn.py --obs full --seed "$seed" \
      --noise-rabi 0.5 --steps "$STEPS" \
      --out "results/dqn_fullLong_nr0.5_seed${seed}.json" > "$log" 2>&1 &
  pids+=($!)
  echo "launched full-long seed$seed -> $log"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail+1))
done
echo "FULL-LONG DONE (failures: $fail)"
exit $fail
