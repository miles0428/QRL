#!/usr/bin/env bash
# QDQN (VQC Q-function from the qdqn-cartpole branch) on the sxy game at 0.35.
#
# Observation is sxy_prevact: [sx, sy, dsx, dsy] + one-hot of the action that
# produced them -- 9 dims, so 9 qubits, one per dimension, matching how the
# CartPole config maps 4 observation dims onto 4 qubits. Five observables, one
# per action, ZZ on consecutive pairs.
#
# Trainer settings come from configs/qdqn.yaml: batch 16, a gradient step every
# 10 environment steps, target sync every 3 gradient steps, and three learning
# rates (w at 0.1, circuit weights and lam at 0.001).
#
#   PY=/path/to/python bash scripts/run_qdqn.sh [steps]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
STEPS="${1:-150000}"
NR=0.35

mkdir -p logs results

pids=()
for seed in 0 1 2; do
  log="logs/qdqn_sxy_prevact_nr${NR}_seed${seed}.log"
  "$PY" -u scripts/train_dqn.py --model vqc --obs sxy_prevact --seed "$seed" \
      --noise-rabi "$NR" --steps "$STEPS" \
      --out "results/qdqn_sxy_prevact_nr${NR}_seed${seed}.json" > "$log" 2>&1 &
  pids+=($!)
  echo "launched qdqn sxy_prevact seed$seed"
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "QDQN DONE (failures: $fail)"
exit $fail
