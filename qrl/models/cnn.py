"""CNN → VQC hybrid models for image-based RL (DinoRun-v0).

Architecture::

    frames [B,4,84,84] ─► NatureCNNEncoder ─► tanh ─► f∈[-1,1]^n_qubits
                                                        │
                                  λ⊙f (input scaling, outside circuit)
                                                        │
                              build_circuit(n_qubits, n_layers, reuploading=True)
                              evaluated on torch_sv (exact autograd statevector)
                                                        │
                              ⟨ZZ⟩ per action (disjoint 2-qubit correlators)
                              Q = (o+1)/2 · w  (output scaling)  → Q-values [B,n_actions]

Three heads share the same CNN encoder and VQC body:
  - CNNVQCDQNActorCritic  → DQN  (Q-function interface)
  - CNNVQCA2CActorCritic  → A2C  (actor-critic interface)
  - CNNVQCPolicy          → PG   (policy interface)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qrl.models.base import PolicyFunction, QFunction, ValueFunction
from qrl.models.vqc import build_circuit
from qrl.backends.torch_statevector import TorchStatevectorQNN


# ---------------------------------------------------------------------------
# Helper: swap CNOT ring entangler (inlined from experiments.common._swap_entangler)
# ---------------------------------------------------------------------------

def _swap_entangler(circuit: QuantumCircuit, entangler: str) -> QuantumCircuit:
    """Replace the CNOT ring with CZ or no entangler.

    Parameters
    ----------
    circuit : QuantumCircuit
        The built circuit with CNOT ring.
    entangler : str
        ``"cx"``  — keep CNOTs (default)
        ``"cz"``  — replace each CNOT with CZ
        ``"none"`` — remove all entanglers

    Returns
    -------
    QuantumCircuit
        A shallow copy of the circuit with entanglers replaced.
    """
    if entangler == "cx":
        return circuit

    new_circuit = circuit.copy()
    new_circuit.data.clear()

    for instruction in circuit.data:
        if instruction.operation.name == "cx":
            q0, q1 = [circuit.find_bit(q).index for q in instruction.qubits]
            if entangler == "cz":
                new_circuit.cz(q0, q1)
            # entangler == "none": skip entirely
        else:
            new_circuit.append(instruction)

    return new_circuit


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class NatureCNNEncoder(nn.Module):
    """Nature-DQN conv stack → Linear → n_qubits → tanh (trained end-to-end).

    Input:  [B, 4, 84, 84] raw stacked frames (uint8)
    Output: [B, out_dim]    ∈ [-1, 1]^out_dim
    """

    def __init__(self, in_channels: int = 4, out_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),  # 84→20
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),            # 20→9
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),            # 9→7
        )
        # 64 * 7 * 7 = 3136
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.conv(x)))  # [B, out_dim] ∈ [-1, 1]


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def _disjoint_zz(n_qubits: int, n_actions: int) -> list[SparsePauliOp]:
    """One disjoint 2-qubit ZZ correlator per action: (0,1), (2,3), (4,5), ..."""
    assert 2 * n_actions <= n_qubits, f"need >= {2*n_actions} qubits for {n_actions} disjoint ZZ pairs"
    obs = []
    for a in range(n_actions):
        q0, q1 = 2 * a, 2 * a + 1
        obs.append(SparsePauliOp.from_sparse_list([("ZZ", [q0, q1], 1.0)], num_qubits=n_qubits))
    return obs


def _single_z(n_qubits: int, n_actions: int) -> list[SparsePauliOp]:
    """One single-qubit Z per action (ablation: single-Z vs ZZ observables)."""
    assert n_actions <= n_qubits
    return [SparsePauliOp.from_sparse_list([("Z", [a], 1.0)], num_qubits=n_qubits)
            for a in range(n_actions)]


# ---------------------------------------------------------------------------
# DQN head
# ---------------------------------------------------------------------------

class CNNVQCDQNActorCritic(QFunction):
    """DQN Q-function: NatureCNNEncoder → VQC head → Q-values.

    Implements the ``QFunction`` interface so ``dqn_update`` / ``select_action``
    treat it exactly like ``VQCQFunction``.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_actions: int = 3,
        n_layers: int = 3,
        reuploading: bool = True,
        observable: str = "zz",
        entangler: str = "cx",
        hidden: int = 256,
        w_init: float = 1.0,
        lam_init: float = 1.0,
        seed: int | None = None,
        lr: dict | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.n_qubits = int(n_qubits)
        self.n_actions = int(n_actions)
        self.n_layers = int(n_layers)
        self.reuploading = reuploading
        self.entangler = entangler
        self.observable_kind = observable

        # Default lr schedule (matching experiments/dino/model.py defaults)
        self._lr = dict(lr or dict(
            cnn=5e-4, input_scaling=3e-3, variational=1e-2, output_scaling=3e-2
        ))

        # CNN encoder
        self.encoder = NatureCNNEncoder(in_channels=4, out_dim=self.n_qubits, hidden=hidden)

        # VQC circuit
        circuit, input_params, weight_params = build_circuit(
            self.n_qubits, n_layers, reuploading
        )
        if entangler != "cx":
            circuit = _swap_entangler(circuit, entangler)
        if observable == "zz":
            self.observables = _disjoint_zz(self.n_qubits, self.n_actions)
        elif observable == "z":
            self.observables = _single_z(self.n_qubits, self.n_actions)
        else:
            raise ValueError(f"unknown observable {observable!r} (expected 'zz' or 'z')")

        init_w = (torch.rand(len(weight_params)) * 2.0 - 1.0) * np.pi
        self.vqc = TorchStatevectorQNN(
            circuit, self.observables, input_params, weight_params,
            initial_weights=init_w
        )

        # Trainable scalings (outside circuit — Failure Mode 2 defense)
        self.lam = nn.Parameter(torch.full((self.n_qubits,), float(lam_init)))
        self.w = nn.Parameter(torch.full((self.n_actions,), float(w_init)))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: [B, 4, 84, 84] uint8 → Q-values [B, n_actions]"""
        x = states.to(torch.float32) / 255.0          # [B, 4, 84, 84] ∈ [0, 1]
        f = self.encoder(x)                           # [B, n_qubits] ∈ [-1, 1]
        encoded = self.lam * f                        # trainable input scaling → RY angles
        o = self.vqc(encoded).to(self.w.dtype)       # [B, n_actions] ∈ [-1, 1]
        return (o + 1.0) * 0.5 * self.w             # → Q-values ∈ [0, w]

    def param_groups(self) -> list[dict]:
        return [
            {"params": list(self.encoder.parameters()), "lr": float(self._lr["cnn"])},
            {"params": [self.lam], "lr": float(self._lr["input_scaling"])},
            {"params": list(self.vqc.parameters()), "lr": float(self._lr["variational"])},
            {"params": [self.w], "lr": float(self._lr["output_scaling"])},
        ]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_counts(self) -> dict:
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        vqc = sum(p.numel() for p in self.vqc.parameters())
        return {
            "cnn_trainable": enc,
            "vqc_circuit": vqc,
            "lam": self.lam.numel(),
            "w": self.w.numel(),
            "total_trainable": self.num_trainable_params(),
        }

    def loggable_scalars(self) -> dict[str, float]:
        out = {f"w{i}": float(v) for i, v in enumerate(self.w.detach().cpu().numpy())}
        out["lam_mean"] = float(self.lam.detach().mean())
        return out


# ---------------------------------------------------------------------------
# A2C head
# ---------------------------------------------------------------------------

class CNNVQCA2CActorCritic(nn.Module):
    """A2C actor-critic: shared CNN encoder + two independent VQC heads.

    forward()  → (policy_logits, value) for A2C training
    get_policy() → policy_logits  (for PG, which only needs the actor)
    get_critic() → V(s)          (value estimate)
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_actions: int = 3,
        n_layers: int = 3,
        reuploading: bool = True,
        observable: str = "zz",
        entangler: str = "cx",
        hidden: int = 256,
        w_init: float = 1.0,
        lam_init: float = 1.0,
        seed: int | None = None,
        lr: dict | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.n_qubits = int(n_qubits)
        self.n_actions = int(n_actions)
        self.n_layers = int(n_layers)
        self.reuploading = reuploading
        self.entangler = entangler
        self.observable_kind = observable

        self._lr = dict(lr or dict(
            cnn=5e-4, input_scaling=3e-3, variational=1e-2,
            output_scaling=3e-2,
        ))

        # Shared CNN encoder
        self.encoder = NatureCNNEncoder(in_channels=4, out_dim=self.n_qubits, hidden=hidden)

        # --- Actor VQC head (produces policy logits) ---
        actor_circuit, actor_inp, actor_w = build_circuit(
            self.n_qubits, n_layers, reuploading
        )
        if entangler != "cx":
            actor_circuit = _swap_entangler(actor_circuit, entangler)
        if observable == "zz":
            actor_obs = _disjoint_zz(self.n_qubits, self.n_actions)
        else:
            actor_obs = _single_z(self.n_qubits, self.n_actions)
        init_w_actor = (torch.rand(len(actor_w)) * 2.0 - 1.0) * np.pi
        self.actor_vqc = TorchStatevectorQNN(
            actor_circuit, actor_obs, actor_inp, actor_w, initial_weights=init_w_actor
        )
        self.actor_lam = nn.Parameter(torch.full((self.n_qubits,), float(lam_init)))
        self.actor_w = nn.Parameter(torch.full((self.n_actions,), float(w_init)))

        # --- Critic VQC head (produces state value) ---
        critic_circuit, critic_inp, critic_w = build_circuit(
            self.n_qubits, n_layers, reuploading
        )
        if entangler != "cx":
            critic_circuit = _swap_entangler(critic_circuit, entangler)
        # Critic uses a single ZZZZ observable (whole-register correlator)
        critic_obs = [SparsePauliOp.from_sparse_list(
            [("Z", [q], 1.0) for q in range(self.n_qubits)], num_qubits=self.n_qubits
        )]
        init_w_critic = (torch.rand(len(critic_w)) * 2.0 - 1.0) * np.pi
        self.critic_vqc = TorchStatevectorQNN(
            critic_circuit, critic_obs, critic_inp, critic_w, initial_weights=init_w_critic
        )
        self.critic_lam = nn.Parameter(torch.full((self.n_qubits,), float(lam_init)))
        self.critic_w = nn.Parameter(torch.full((1,), float(w_init)))

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits, value) for A2C training.

        states: [B, 4, 84, 84] uint8
        policy_logits: [B, n_actions]
        value: [B]  (ValueFunction contract: shape [B], not [B,1])
        """
        x = states.to(torch.float32) / 255.0          # [B, 4, 84, 84] ∈ [0, 1]
        f = self.encoder(x)                           # [B, n_qubits] ∈ [-1, 1]

        # Actor
        enc_actor = self.actor_lam * f
        o_actor = self.actor_vqc(enc_actor).to(self.actor_w.dtype)  # [B, n_actions]
        policy_logits = (o_actor + 1.0) * 0.5 * self.actor_w       # [B, n_actions]

        # Critic
        enc_critic = self.critic_lam * f
        raw_critic = self.critic_vqc(enc_critic)  # [B, 1] ∈ [-1, 1]
        value = ((raw_critic + 1.0) * 0.5 * self.critic_w).reshape(-1)  # [B]

        return policy_logits, value

    def get_policy(self, states: torch.Tensor) -> torch.Tensor:
        """PG interface: return policy logits only."""
        x = states.to(torch.float32) / 255.0
        f = self.encoder(x)
        enc = self.actor_lam * f
        o = self.actor_vqc(enc).to(self.actor_w.dtype)
        return (o + 1.0) * 0.5 * self.actor_w

    def get_critic(self, states: torch.Tensor) -> torch.Tensor:
        """Return V(s) as a [B] tensor (ValueFunction contract)."""
        x = states.to(torch.float32) / 255.0
        f = self.encoder(x)
        enc = self.critic_lam * f
        raw = self.critic_vqc(enc)
        return ((raw + 1.0) * 0.5 * self.critic_w).reshape(-1)

    def param_groups(self) -> list[dict]:
        return [
            {"params": list(self.encoder.parameters()), "lr": float(self._lr["cnn"])},
            # Actor
            {"params": [self.actor_lam], "lr": float(self._lr["input_scaling"])},
            {"params": list(self.actor_vqc.parameters()), "lr": float(self._lr["variational"])},
            {"params": [self.actor_w], "lr": float(self._lr["output_scaling"])},
            # Critic
            {"params": [self.critic_lam], "lr": float(self._lr["input_scaling"])},
            {"params": list(self.critic_vqc.parameters()), "lr": float(self._lr["variational"])},
            {"params": [self.critic_w], "lr": float(self._lr["output_scaling"])},
        ]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# PG head
# ---------------------------------------------------------------------------

class CNNVQCPolicy(PolicyFunction):
    """REINFORCE policy: NatureCNNEncoder → VQC policy head → action distribution logits.

    Uses the same Softmax-VQC head as VQCPolicy (beta inverse-temperature),
    with the CNN as the perception encoder instead of arctan normalization.
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_actions: int = 3,
        n_layers: int = 3,
        reuploading: bool = True,
        observable: str = "zz",
        entangler: str = "cx",
        hidden: int = 256,
        lam_init: float = 1.0,
        seed: int | None = None,
        beta_init: float = 1.0,
        trainable_beta: bool = True,
        lr: dict | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        from qrl.models.vqc_policy import _inverse_softplus

        self.n_qubits = int(n_qubits)
        self.n_actions = int(n_actions)
        self.n_layers = int(n_layers)
        self.reuploading = reuploading
        self.entangler = entangler

        self._lr = dict(lr or dict(
            cnn=5e-4, input_scaling=3e-3, variational=1e-2, beta=1e-1,
        ))

        # CNN encoder
        self.encoder = NatureCNNEncoder(in_channels=4, out_dim=self.n_qubits, hidden=hidden)

        # VQC circuit
        circuit, input_params, weight_params = build_circuit(
            self.n_qubits, n_layers, reuploading
        )
        if entangler != "cx":
            circuit = _swap_entangler(circuit, entangler)
        if observable == "zz":
            observables = _disjoint_zz(self.n_qubits, self.n_actions)
        else:
            observables = _single_z(self.n_qubits, self.n_actions)
        init_w = (torch.rand(len(weight_params)) * 2.0 - 1.0) * np.pi
        self.vqc = TorchStatevectorQNN(
            circuit, observables, input_params, weight_params, initial_weights=init_w
        )

        # Scalings
        self.lam = nn.Parameter(torch.full((self.n_qubits,), float(lam_init)))
        self.beta_raw = nn.Parameter(
            torch.tensor([_inverse_softplus(beta_init)], dtype=torch.float32),
            requires_grad=trainable_beta,
        )

    @property
    def beta(self) -> torch.Tensor:
        return nn.functional.softplus(self.beta_raw)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """states: [B, 4, 84, 84] uint8 → logits [B, n_actions]"""
        x = states.to(torch.float32) / 255.0          # [B, 4, 84, 84] ∈ [0, 1]
        f = self.encoder(x)                           # [B, n_qubits] ∈ [-1, 1]
        scaled = self.lam * f                         # trainable input scaling
        expvals = self.vqc(scaled)                   # [B, n_actions] ∈ [-1, 1]
        return self.beta * expvals                    # logits; softmax applied by loss

    def param_groups(self) -> list[dict]:
        return [
            {"params": list(self.encoder.parameters()), "lr": float(self._lr["cnn"])},
            {"params": [self.lam], "lr": float(self._lr["input_scaling"])},
            {"params": list(self.vqc.parameters()), "lr": float(self._lr["variational"])},
            {"params": [self.beta_raw], "lr": float(self._lr["beta"])},
        ]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    torch.manual_seed(0)
    batch = 4

    print("=== CNNVQCDQNActorCritic ===")
    dqn = CNNVQCDQNActorCritic(n_qubits=6, n_actions=3, n_layers=3, seed=0)
    states = torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8)
    q = dqn(states)
    assert q.shape == (batch, 3), q.shape
    q.sum().backward()
    alive = {k: p.grad is not None and p.grad.abs().sum().item() > 0
             for k, p in [("cnn", next(dqn.encoder.parameters())),
                          ("lam", dqn.lam),
                          ("vqc", dqn.vqc.weights),
                          ("w", dqn.w)]}
    print(f"  out {tuple(q.shape)} | grads alive: {alive} | params: {dqn.param_counts()}")
    print(f"  Q range [{q.min():.3f}, {q.max():.3f}]")

    print("\n=== CNNVQCA2CActorCritic ===")
    a2c = CNNVQCA2CActorCritic(n_qubits=6, n_actions=3, n_layers=3, seed=0)
    logits, value = a2c(states)
    assert logits.shape == (batch, 3), logits.shape
    assert value.shape == (batch,), value.shape
    (logits.sum() + value.sum()).backward()
    print(f"  logits {tuple(logits.shape)} | value {tuple(value.shape)}")
    print(f"  params: {a2c.num_trainable_params()}")

    print("\n=== CNNVQCPolicy ===")
    pg = CNNVQCPolicy(n_qubits=6, n_actions=3, n_layers=3, seed=0)
    logits_pg = pg(states)
    assert logits_pg.shape == (batch, 3), logits_pg.shape
    probs = torch.softmax(logits_pg, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(batch)), probs.sum(-1)
    logits_pg.sum().backward()
    print(f"  logits {tuple(logits_pg.shape)} | probs sum {tuple(probs.sum(-1).shape)}")
    print(f"  params: {pg.num_trainable_params()}")

    print("\nAll smoke tests passed ✓")
