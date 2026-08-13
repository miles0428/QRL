"""Classical MLP baseline, parameter-count matched to the VQC.

VQCQFunction's default config (4 qubits, n_layers=5) has 40 circuit weights +
4 lam + 2 w = 46 trainable parameters total. A 4 -> hidden -> 2 MLP with bias
has 7*hidden + 2 parameters; hidden=6 gives 44 (within the required +-20% of
46 -- see README for the derivation). This is deliberately tiny: the point of
the comparison in benchmark/plotting code is parameter-efficiency, and a
large MLP (e.g. 2x128 ~ 17k params) would make that comparison meaningless.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import PolicyFunction, QFunction, ValueFunction, normalize_observation

DEFAULT_HIDDEN = 6  # 7*6 + 2 = 44 params, vs VQC's 46 (default config) -- see README
# A critic has one output unit instead of two, so it needs one more hidden unit
# to land at a comparable parameter count: 5*7 + 1 = 43 vs VQCValue's 45.
DEFAULT_CRITIC_HIDDEN = 7


class MLPQFunction(QFunction):
    def __init__(self, n_inputs: int = 4, hidden: int = DEFAULT_HIDDEN, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)
        return self.net(normalized)


class MLPPolicy(PolicyFunction):
    """Classical policy-gradient baseline, parameter-matched to VQCPolicy.

    Same body as MLPQFunction -- only the output contract differs (logits, not
    Q-values), so hidden=6 gives 44 parameters against VQCPolicy's 45 (40
    circuit + 4 lam + 1 beta). The match is even closer than on the DQN side.

    No explicit inverse temperature. A linear output layer can already scale its
    own logits without bound, so a beta here would be redundant -- unlike the
    quantum head, whose expectation values are hard-bounded to [-1, 1] and
    therefore cannot sharpen the policy on their own. That asymmetry is a real
    property of the two models, not an inconsistency in the comparison.
    """

    def __init__(self, n_inputs: int = 4, hidden: int = DEFAULT_HIDDEN, n_actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)
        return self.net(normalized)


class MLPValue(ValueFunction):
    """Classical critic, parameter-matched to VQCValue (45: 40 circuit + 4 lam + 1 w).

    Uses hidden=7 where the actor uses 6, which is not an oversight. A critic has
    ONE output unit instead of two, so it loses parameters relative to the actor
    at equal width: 5*hidden + 1 gives 37 at hidden=6 (18% under the VQC's 45,
    inside the +-20% band but only just) and 43 at hidden=7 (4% under). The
    comparison this project makes is parameter-efficiency, so the closer match is
    the honest one, and the width is an implementation detail either way.

    THE HEAD IS VQCValue'S, DELIBERATELY: `w * sigmoid(.)`, a [0,1] core scaled
    by a trainable magnitude initialized to the return scale.

    A plain linear output would be the obvious choice and is the wrong one here.
    An A2C run has only ~60 gradient steps, and Adam moves a parameter by roughly
    its learning rate per step, so the output weights (initialized at ~0.4) get
    nowhere near the ~14 they would need to span [0, V*] with tanh features. The
    critic stays effectively constant, and a constant critic has explained
    variance of exactly 0 -- which turns GAE's baseline back into REINFORCE's
    batch mean, the same degeneration documented in src/models/vqc_value.py.

    Measured on a 3-gradient-step probe: explained_variance 0.004 for a plain
    linear head, 0.000 after merely setting its bias to V* (a constant predictor
    is still a constant predictor), against 0.466 for the quantum critic whose
    head already had range.

    Giving both critics the same head shape is also what makes the actor/critic
    grid measure what it claims to. The quantum head NEEDS a scale because
    expectation values are hard-bounded to [-1,1]; the classical one does not, in
    principle. But if only one of them starts with usable range, the grid
    compares initializations rather than function approximators.
    """

    def __init__(self, n_inputs: int = 4, hidden: int = DEFAULT_CRITIC_HIDDEN,
                 value_init: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.w = nn.Parameter(torch.full((1,), float(value_init)))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        normalized = normalize_observation(states)
        core = torch.sigmoid(self.net(normalized))  # [B, 1] in [0, 1]
        return (core * self.w).reshape(-1)  # [B], see ValueFunction docstring


if __name__ == "__main__":
    critic = MLPValue()
    n_critic = sum(p.numel() for p in critic.parameters())
    print(f"MLPValue trainable params (hidden={DEFAULT_CRITIC_HIDDEN}): {n_critic}  "
          f"vs VQCValue's 45")
    v = critic(torch.rand(5, 4, dtype=torch.float32))
    assert v.shape == (5,), v.shape
    assert abs(n_critic - 45) / 45 <= 0.20, f"{n_critic} is outside +-20% of VQCValue's 45"

    policy = MLPPolicy()
    print(f"MLPPolicy trainable params (hidden={DEFAULT_HIDDEN}): "
          f"{sum(p.numel() for p in policy.parameters())}")
    assert policy(torch.rand(5, 4, dtype=torch.float32)).shape == (5, 2)

    model = MLPQFunction()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP trainable params (hidden={DEFAULT_HIDDEN}): {n_params}")

    states = torch.rand(8, 4, dtype=torch.float32) * 0.2 - 0.1
    q_values = model(states)
    assert q_values.shape == (8, 2)

    loss = q_values.sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0
    print("mlp.py smoke test OK")
