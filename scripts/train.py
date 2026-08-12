"""Single-run entrypoint.  [WIRED AT CHECKPOINT 3/4]

    python scripts/train.py --config configs/qdqn.yaml --seed 0

Loads a YAML config, prints resolved versions, seeds everything, builds the model via
the factory, and runs the model-agnostic trainer, writing results/{name}_{seed}.csv.
"""
from __future__ import annotations

import sys

if __name__ == "__main__":
    raise SystemExit("scripts/train.py is wired at Checkpoint 3/4.")
