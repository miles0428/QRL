#!/usr/bin/env bash
# Render every policy's episode as MP4 at the same settings, for side-by-side use.
#   PY=/path/to/python bash scripts/render_all_mp4.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PY:-python}"
DPI="${1:-140}"

# QDQN carries the amplitude histogram, so it gets the taller layout.
"$PY" scripts/animate_episode.py --policy checkpoint \
    --ckpt results/qdqn_anim_nr0.35_seed0.pt --seeds 1000-1005 \
    --stride 4 --fps 20 --dpi "$DPI" --out figures/episode_qdqn.mp4

"$PY" scripts/animate_episode.py --policy checkpoint \
    --ckpt results/ppo_anim_full_prevact_nr0.35.pt --seeds 1000-1005 \
    --stride 4 --fps 20 --dpi "$DPI" --out figures/episode_ppo.mp4

"$PY" scripts/animate_episode.py --policy greedy --seeds 1000-1005 \
    --stride 4 --fps 20 --dpi "$DPI" --out figures/episode_greedy.mp4

# IDLE dies in a handful of steps; every step, slowly, or there is nothing to see.
"$PY" scripts/animate_episode.py --policy idle --seeds 1000-1005 \
    --stride 1 --fps 4 --dpi "$DPI" --out figures/episode_idle.mp4

echo "ALL MP4 DONE"
