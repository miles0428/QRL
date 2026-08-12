"""QDQN-on-CartPole research package.

Subpackage map:
  models/   QFunction interface (base), VQC (quantum) and MLP (classical) approximators.
  replay    uniform experience replay (stores raw observations).
  trainer   model-agnostic DQN loop -- MUST NOT import qiskit.
  evaluate  greedy rollouts + the single solve criterion + finite-shot hook.
  seeds     global seeding for byte-identical runs.
  plots     figure functions.
  versions  resolved-dependency logging (printed at the top of every run log).
"""
