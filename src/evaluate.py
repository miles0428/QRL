"""Greedy evaluation + the single solve criterion + finite-shot hook.  [CHECKPOINT 3/5]

The ONE solve definition used everywhere: mean reward >= 475 over 100 consecutive
episodes. Reports episodes-to-solve and env-steps-to-solve, not just final reward.

`evaluate_finite_shot` is left as a stub hook (per spec): it will swap in a
qiskit-aer estimator with a given shot count for the later robustness sweep. The
sweep itself is intentionally NOT implemented yet.
"""
from __future__ import annotations


def greedy_eval(*args, **kwargs):
    raise NotImplementedError("greedy_eval() is implemented at Checkpoint 3.")


def evaluate_finite_shot(model, shots: int):
    """Hook: evaluate `model` under finite sampling with an Aer estimator.

    Deliberately unimplemented -- only the signature exists so the later shot sweep
    plugs in without touching the trainer. Aer is used ONLY here, never in training.
    """
    raise NotImplementedError(
        "Finite-shot evaluation is a stub hook (see spec). Implement in the shot-sweep phase."
    )
