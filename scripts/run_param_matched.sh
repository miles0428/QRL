#!/usr/bin/env bash
# Parameter-matched classical control for the QDQN comparison.
#
# QDQN on sxy_prevact has 140 trainable parameters (90 circuit, 45 lam, 5 w).
# The classical runs used an MLP of width 128 = 18,437 parameters, so "QDQN uses
# 132x fewer parameters" says nothing on its own about circuit expressivity --
# the two models were never matched. An MLP of width 6 has 137 parameters, which
# is the honest control: same task, same loop, same evaluation seeds, same
# budget, ~same parameter count.
#
#   PY=/path/to/python bash scripts/run_param_matched.sh [width] [steps]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
WIDTH="${1:-6}"
STEPS="${2:-150000}"
NR=0.35

mkdir -p logs results

pids=()
for seed in 0 1 2; do
  log="logs/dqn_w${WIDTH}_sxy_prevact_nr${NR}_seed${seed}.log"
  "$PY" -u scripts/train_dqn.py --obs sxy_prevact --width "$WIDTH" --seed "$seed" \
      --noise-rabi "$NR" --steps "$STEPS" \
      --out "results/dqn_w${WIDTH}_sxy_prevact_nr${NR}_seed${seed}.json" > "$log" 2>&1 &
  pids+=($!)
  echo "launched width=$WIDTH seed$seed"
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "PARAM-MATCHED DONE (failures: $fail)"
exit $fail
