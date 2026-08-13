#!/usr/bin/env bash
# The sxy game at noise/Rabi = 0.35: observation is [sx, sy, dsx, dsy].
#
# sz is information-theoretically recoverable while the episode is live
# (sz = +sqrt(1-sx^2-sy^2), verified to 4e-14), so this mask hides nothing in
# the Shannon sense. What it breaks is model class: the update of (sx, sy)
# depends on sz, so on this projection the dynamics are no longer linear.
# A linear fitted model, exact on the full Bloch vector, is misspecified here --
# measured residual 0.206 vs 0.138, 1.5x worse.
#
# So this run asks: can a network recover what a linear model cannot?
#
#   PY=/path/to/python bash scripts/run_sxy.sh [steps]
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
  for obs in sxy sxy_prevact; do
    for algo in dqn ppo; do
      log="logs/${algo}_${obs}_nr${NR}_seed${seed}.log"
      "$PY" -u "scripts/train_${algo}.py" --obs "$obs" --seed "$seed" \
          --noise-rabi "$NR" --steps "$STEPS" > "$log" 2>&1 &
      pids+=($!)
      echo "launched $algo $obs seed$seed"
    done
  done
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "SXY DONE (failures: $fail)"
exit $fail
