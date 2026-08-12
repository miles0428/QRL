"""Model factory. The ONLY place that maps a config `type` string to a concrete
QFunction subclass, and the only import path through which qiskit is pulled in.

`trainer.py` imports `build_model` (or receives an already-built model) but never a
concrete model class, so it stays quantum-agnostic. Adding an ansatz = new subclass +
one `elif` here.
"""
from __future__ import annotations

from .base import QFunction


def build_model(config: dict) -> QFunction:
    """Construct a QFunction from a parsed config dict.

    Expects `config["model"]` (with a `type`) and `config["normalization"]`.
    Imports of concrete models are lazy so an MLP run never imports qiskit.
    """
    model_cfg = config["model"]
    norm_cfg = config["normalization"]
    mtype = model_cfg["type"].lower()

    if mtype == "vqc":
        from .vqc import VQCQFunction
        return VQCQFunction(model_cfg, norm_cfg)
    if mtype == "mlp":
        from .mlp import MLPQFunction
        return MLPQFunction(model_cfg, norm_cfg)

    raise ValueError(f"Unknown model type {mtype!r}. Expected 'vqc' or 'mlp'.")


__all__ = ["QFunction", "build_model"]
