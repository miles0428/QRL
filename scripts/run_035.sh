#!/usr/bin/env bash
# Everything at noise/Rabi = 0.35 only.
#
# Three configurations, 3 seeds each:
#   dqn full_prevact  -- Markov obs + one-hot of the action that produced it (11D).
#                        The spin state is already Markov, so stacking old states
#                        adds nothing; the one missing piece is which action caused
#                        the observed delta, since delta = own action + noise.
#   ppo full          -- policy-gradient baseline. On-policy, so no replay buffer
#                        and none of the stale-action-history pathology.
#   ppo full_prevact  -- both changes together.
#
# Compare against, at the same noise level and the same eval seeds:
#   greedy (physics)        399.56
#   greedy (learned model)  388.94
#   dqn full                359.79 (best-ckpt, inflated; unbiased is lower)
#   dqn full_hist           319.98
#
#   PY=/path/to/python bash scripts/run_035.sh [steps]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY="${PY:-python}"
STEPS="${1:-150000}"
NR=0.35

mkdir -p logs results

pids=()
launch() {  # $1=script $2=tag $3=obs $4=seed
  local log="logs/$2_$3_nr${NR}_seed$4.log"
  "$PY" -u "scripts/$1" --obs "$3" --seed "$4" --noise-rabi "$NR" --steps "$STEPS" \
      > "$log" 2>&1 &
  pids+=($!)
  echo "launched $2 $3 seed$4"
}

for seed in 0 1 2; do
  launch train_dqn.py dqn full_prevact "$seed"
  launch train_ppo.py ppo full         "$seed"
  launch train_ppo.py ppo full_prevact "$seed"
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "RUN_035 DONE (failures: $fail)"
exit $fail
