"""VQC correctness tests (Checkpoint 2). Both target SILENT-failure bugs.

  test_forward_and_grads_smoke
      random batch of 8 states -> output shape [8, 2]; gradients non-None AND non-zero
      for ALL THREE param groups (lam, vqc, w). A dead `lam` gradient (scalings in the
      circuit, or input_gradients=False) is exactly what this catches.

  test_reverse_vs_paramshift_agreement
      gradients for one small batch from BOTH gradient methods must agree to ~1e-6.
      Both are exact, so any mismatch means the binding, observables, or circuit is wrong.

Runnable two ways:
    pytest tests/test_vqc.py
    python tests/test_vqc.py        # standalone (no pytest needed), prints PASS/FAIL
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.vqc import VQCQFunction  # noqa: E402

# Small ansatz (n_layers=3) to keep the param-shift path fast in tests.
MODEL_CFG = dict(
    type="vqc", n_qubits=4, n_layers=3, reuploading=False,
    gradient="reverse", observable_qubits=[0, 1],
    lr=dict(input_scaling=1e-3, variational=1e-3, output_scaling=1e-1),
)
NORM_CFG = dict(cart_position_bound=2.4, pole_angle_bound=0.2095, velocity_transform="arctan")


def _make(gradient: str) -> VQCQFunction:
    cfg = dict(MODEL_CFG)
    cfg["gradient"] = gradient
    return VQCQFunction(cfg, NORM_CFG)


def test_forward_and_grads_smoke():
    torch.manual_seed(0)
    m = _make("reverse")
    states = torch.randn(8, 4)

    q = m(states)
    assert q.shape == (8, 2), f"expected [8, 2], got {tuple(q.shape)}"

    m.zero_grad()
    q.pow(2).mean().backward()

    vqc_w = next(iter(m.vqc.parameters()))
    for name, p in (("lam", m.lam), ("vqc", vqc_w), ("w", m.w)):
        assert p.grad is not None, f"{name} gradient is None"
        assert p.grad.abs().sum().item() > 0, f"{name} gradient is all-zero (silent dead grad)"


def test_reverse_vs_paramshift_agreement():
    torch.manual_seed(0)
    mr = _make("reverse")
    mp = _make("paramshift")
    # Force byte-identical parameters so we compare gradients, not initialisations.
    with torch.no_grad():
        for pr, pp in zip(mr.parameters(), mp.parameters()):
            pp.copy_(pr)

    states = torch.randn(5, 4)

    def run(m):
        m.zero_grad()
        out = m(states)
        out.sum().backward()
        vqc_w = next(iter(m.vqc.parameters()))
        return dict(out=out.detach(), lam=m.lam.grad.clone(),
                    w=m.w.grad.clone(), vqc=vqc_w.grad.clone())

    a, b = run(mr), run(mp)
    tol = 1e-5  # float32 through TorchConnector; both paths are exact so diff ~1e-7
    diffs = {k: (a[k] - b[k]).abs().max().item() for k in a}
    for k, d in diffs.items():
        assert d < tol, f"{k} disagrees between reverse and paramshift: max|d|={d:.2e}"
    return diffs


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print("running test_forward_and_grads_smoke ...", flush=True)
    test_forward_and_grads_smoke()
    print("  PASS: shape [8,2]; lam/vqc/w gradients all non-zero")
    print("running test_reverse_vs_paramshift_agreement ...", flush=True)
    d = test_reverse_vs_paramshift_agreement()
    print("  max|reverse - paramshift|:", {k: f"{v:.2e}" for k, v in d.items()})
    print("  PASS: reverse and param-shift agree to ~1e-6")
    print("ALL VQC TESTS PASSED")
