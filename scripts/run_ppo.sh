#!/usr/bin/env bash
# PPO across observation modes and noise levels, 3 seeds each.
#
#   PY=/path/to/python bash scripts/run_ppo.sh [steps]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
STEPS="${1:-150000}"

mkdir -p logs results

pids=()
for nr in 0.5 0.35; do
  for obs in full full_hist; do
    for seed in 0 1 2; do
      log="logs/ppo_${obs}_nr${nr}_seed${seed}.log"
      "$PY" -u scripts/train_ppo.py --obs "$obs" --seed "$seed" \
          --noise-rabi "$nr" --steps "$STEPS" > "$log" 2>&1 &
      pids+=($!)
      echo "launched PPO $obs nr=$nr seed$seed"
    done
  done
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "PPO DONE (failures: $fail)"
exit $fail
