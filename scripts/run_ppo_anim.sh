#!/usr/bin/env bash
# One PPO run with checkpointing, to have a trained policy available to animate.
#   PY=/path/to/python bash scripts/run_ppo_anim.sh [obs] [steps]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PY:-python}"
OBS="${1:-full_prevact}"
STEPS="${2:-150000}"
mkdir -p logs results
"$PY" -u scripts/train_ppo.py --obs "$OBS" --seed 1 --noise-rabi 0.35 \
    --steps "$STEPS" --save-model \
    --out "results/ppo_anim_${OBS}_nr0.35.json" > "logs/ppo_anim_${OBS}.log" 2>&1
echo "PPO ANIM DONE"
