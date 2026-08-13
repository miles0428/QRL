"""Quantum-classical hybrid agent playing a headless Chrome-Dino clone.

Demo (NOT a quantum-advantage claim): a CNN compresses the screen into a few features and a
variational quantum circuit (the backbone's single ``build_circuit``) is the Q-function head.
All training runs on the backbone ``torch_sv`` autograd backend so CNN+VQC form one graph.
"""
import os as _os
import sys as _sys

# Put the QRL repo root (its `src` package: build_circuit, torch_sv, trainer, replay, and the
# sibling `experiments.common`) on the import path so `import src.*` / `import experiments.common`
# resolve from anywhere under experiments/dino/. This file lives at QRL/experiments/dino/__init__.py,
# so the repo root is three directories up. READ-ONLY on src.
_QRL_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _QRL_ROOT not in _sys.path:
    _sys.path.insert(0, _QRL_ROOT)
