"""VQC correctness tests.  [IMPLEMENTED AT CHECKPOINT 2 -- must pass before proceeding]

Two tests, both targeting silent-failure bugs:

  test_forward_and_grads_smoke
      random batch of 8 states -> assert output shape [8, 2]; assert gradients are
      non-None AND non-zero for ALL THREE param groups (lam, vqc, w). A dead `lam`
      gradient (the exact bug from putting scalings inside the circuit, or forgetting
      input_gradients=True) is what this catches.

  test_reverse_vs_paramshift_agreement
      compute gradients for one small batch with BOTH ReverseEstimatorGradient and
      ParamShiftEstimatorGradient; assert they agree to ~1e-6. Both are exact, so any
      mismatch means the parameter binding, observables, or circuit structure is wrong.
"""
import pytest

pytest.skip("Implemented at Checkpoint 2.", allow_module_level=True)
