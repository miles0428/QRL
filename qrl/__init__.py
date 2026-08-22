"""QRL — Quantum Reinforcement Learning framework.

Top-level package exports the two primary factory APIs:
  - build_model  : from qrl.models
  - build_trainer: from qrl.trainers
"""
from __future__ import annotations

from qrl.models import build_model
from qrl.trainers import build_trainer

__all__ = ["build_model", "build_trainer"]
