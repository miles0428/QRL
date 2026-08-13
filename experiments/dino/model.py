"""CNN → VQC hybrid Q-function for the dino game.

Architecture (honest framing: the CNN does the perception, the VQC is a *tiny* quantum
decision head — this is a hybrid demo, not a quantum-advantage claim):

    frames [B,4,84,84] ─► CNN encoder ─► tanh ─► f∈[-1,1]^n_qubits
                                                    │
                              λ⊙f (input scaling, OUTSIDE circuit, backbone convention)
                                                    │
                              build_circuit(n_qubits, n_layers, reuploading=True)   ← THE single
                              evaluated on backbone torch_sv (exact autograd statevector)  circuit
                                                    │
                              ⟨ZZ⟩ per action (disjoint 2-qubit correlators)  o∈[-1,1]^n_actions
                                                    │
                              Q = (o+1)/2 · w  (w trainable output scaling)  → Q-values [B,n_actions]

Because ``torch_sv`` is pure torch, ``model(frames).sum().backward()`` flows through the VQC
AND the CNN in one graph — verified by the smoke test below (live grads on CNN, λ, circuit
weights, and w).

Two encoders (both implemented; default B):
  A. ``pretrained``     — torchvision mobilenet_v3_small, ImageNet weights, FROZEN. The last 3
                          stacked frames become the 3 input channels (keeps motion; fair to B),
                          normalized with ImageNet stats. Only the projection Linear→n_qubits
                          trains. NOTE: ImageNet features are out-of-distribution for a
                          black-and-white line game, so this is expected to underperform B.
  B. ``trainable_cnn``  — Nature-DQN conv stack (32,8×8,s4 → 64,4×4,s2 → 64,3×3,s1) → Linear →
                          n_qubits → tanh, trained end-to-end with the VQC through torch_sv.

Reward is strictly non-negative (+1/frame), so mapping ⟨ZZ⟩∈[-1,1] → (o+1)/2∈[0,1] scaled by a
trainable per-action ``w`` (expected to climb toward the tens) is the natural value head — the
same Failure-Mode-1 output-scaling trick the backbone VQC uses for CartPole.
"""
from __future__ import annotations

# `python -m experiments.dino.model` runs the package __init__ first, which puts the QRL
# backbone `src` on sys.path. (Belt-and-suspenders for direct `python experiments/dino/model.py`.)
if __name__ == "__main__" and __package__ in (None, ""):
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
    __package__ = "experiments.dino"
    import experiments.dino  # noqa: F401  (triggers the sys.path bootstrap)

import numpy as np
import torch
import torch.nn as nn

from qiskit.quantum_info import SparsePauliOp
from experiments.common import _swap_entangler   # backbone ablation hook (CZ / none entangler)
from src.models.base import QFunction            # backbone interface (READ-ONLY)
from src.models.torch_sv import TorchStatevectorQNN
from src.models.vqc import build_circuit         # THE single circuit source of truth


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------
class NatureCNNEncoder(nn.Module):
    """Variant B: Nature-DQN conv stack → Linear → n_qubits → tanh (trained end-to-end)."""

    def __init__(self, in_channels: int = 4, out_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # 84 →20→9→7 spatial; 64*7*7 = 3136
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.conv(x)))       # [B, out_dim] in [-1, 1]


class PretrainedEncoder(nn.Module):
    """Variant A: FROZEN ImageNet mobilenet_v3_small features → trainable Linear → tanh.

    The 4-frame stack's last 3 frames are used as the 3 RGB channels (keeps motion info so the
    comparison to B is fair), normalized with ImageNet statistics. Only ``proj`` trains.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, out_dim: int = 6, backbone: str = "mobilenet_v3_small"):
        super().__init__()
        import torchvision
        if backbone == "mobilenet_v3_small":
            net = torchvision.models.mobilenet_v3_small(
                weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            self.features = net.features
            feat_dim = 576
        elif backbone == "resnet18":
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(net.children())[:-2])
            feat_dim = 512
        else:
            raise ValueError(f"unknown backbone {backbone!r}")
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.features.eval()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feat_dim, out_dim)          # the only trainable part
        self.register_buffer("_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()                              # frozen backbone: keep BN/dropout in eval
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,4,84,84] in [0,1]. Use the last 3 frames as RGB channels (motion preserved).
        rgb = x[:, 1:4, :, :] if x.shape[1] >= 3 else x[:, :1, :, :].repeat(1, 3, 1, 1)
        rgb = (rgb - self._mean) / self._std
        with torch.no_grad():
            feat = self.features(rgb)
        feat = self.pool(feat).flatten(1)                 # [B, feat_dim]
        return torch.tanh(self.proj(feat))                # [B, out_dim] in [-1, 1]


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------
def _disjoint_zz(n_qubits: int, n_actions: int) -> list[SparsePauliOp]:
    """One disjoint 2-qubit ZZ correlator per action: (0,1),(2,3),(4,5),... (Z-diagonal)."""
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
# the hybrid Q-function
# ---------------------------------------------------------------------------
class DinoQFunction(QFunction):
    """CNN encoder → VQC head Q-function. Obeys the backbone ``QFunction`` interface, so the
    reused ``dqn_update``/``select_action`` treat it exactly like the CartPole VQC."""

    def __init__(self, n_qubits: int = 6, n_actions: int = 3, n_layers: int = 5,
                 reuploading: bool = True, encoder: str = "trainable_cnn",
                 observable: str = "zz", hidden: int = 128, pretrained_backbone: str = "mobilenet_v3_small",
                 head: str = "vqc", lr: dict | None = None, w_init: float = 1.0,
                 lam_init: float = 1.0, seed: int | None = None):
        super().__init__()
        self.head_kind = head                    # "vqc" (quantum) | "classical" (diagnostic Linear head)
        self.n_qubits = int(n_qubits)
        self.n_actions = int(n_actions)
        self.n_layers = int(n_layers)
        self.reuploading = bool(reuploading)
        self.obs_dim = (4, 84, 84)               # image obs (informational; trainer stays agnostic)
        self.encoder_kind = encoder
        self.observable_kind = observable
        # output_scaling gets a large lr (backbone philosophy). variational lr is higher than the
        # backbone's CartPole value because the VQC head sits on top of a CNN and the diagnostic
        # showed it under-trains relative to a linear head — it needs to move faster.
        self._lr = dict(lr or dict(cnn=5e-4, input_scaling=3e-3, variational=1e-2, output_scaling=3e-2))

        if seed is not None:                     # reproducible weight init across encoder + circuit
            torch.manual_seed(seed)

        # --- encoder (perception) ---
        if encoder == "trainable_cnn":
            self.encoder: nn.Module = NatureCNNEncoder(in_channels=4, out_dim=self.n_qubits, hidden=hidden)
        elif encoder == "pretrained":
            self.encoder = PretrainedEncoder(out_dim=self.n_qubits, backbone=pretrained_backbone)
        else:
            raise ValueError(f"unknown encoder {encoder!r} (expected 'trainable_cnn' or 'pretrained')")

        if head == "vqc":
            # --- the ONE circuit, on torch_sv (exact-autograd statevector) ---
            self.circuit, input_params, weight_params = build_circuit(self.n_qubits, n_layers, reuploading)
            if observable == "zz":
                self.observables = _disjoint_zz(self.n_qubits, self.n_actions)
            elif observable == "z":
                self.observables = _single_z(self.n_qubits, self.n_actions)
            else:
                raise ValueError(f"unknown observable {observable!r} (expected 'zz' or 'z')")
            init_w = (torch.rand(len(weight_params)) * 2.0 - 1.0) * np.pi     # backbone init convention
            self.vqc = TorchStatevectorQNN(self.circuit, self.observables, input_params,
                                           weight_params, initial_weights=init_w)
            # --- torch-side scalings (outside the circuit), backbone convention ---
            self.lam = nn.Parameter(torch.full((self.n_qubits,), float(lam_init)))  # input scaling (FM 2)
            self.w = nn.Parameter(torch.full((self.n_actions,), float(w_init)))   # output scaling (FM 1)
        elif head == "classical":
            # DIAGNOSTIC head: same CNN + same n_qubits tanh bottleneck, but a plain Linear instead
            # of the VQC. If this learns and the VQC doesn't, the quantum head is the bottleneck.
            self.classical_head = nn.Linear(self.n_qubits, self.n_actions)
        else:
            raise ValueError(f"unknown head {head!r} (expected 'vqc' or 'classical')")

    # -- forward -------------------------------------------------------------
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = states.to(torch.float32) / 255.0                  # raw 0-255 pixels → [0,1]
        f = self.encoder(x)                                   # [B, n_qubits] in [-1,1]
        if self.head_kind == "classical":
            return self.classical_head(f)                     # [B, n_actions] raw Q-values
        encoded = self.lam * f                                # trainable input scaling → RY angles
        o = self.vqc(encoded).to(self.w.dtype)                # [B, n_actions] in [-1,1]
        return (o + 1.0) * 0.5 * self.w                       # → Q-values in [0, w] (returns are >=0)

    # -- optimizer grouping (owned by the model) -----------------------------
    def param_groups(self) -> list[dict]:
        if self.head_kind == "classical":
            return [
                {"params": list(self.encoder.parameters()), "lr": float(self._lr["cnn"])},
                {"params": list(self.classical_head.parameters()), "lr": float(self._lr["variational"])},
            ]
        return [
            {"params": list(self.encoder.parameters()), "lr": float(self._lr["cnn"])},
            {"params": [self.lam], "lr": float(self._lr["input_scaling"])},
            {"params": list(self.vqc.parameters()), "lr": float(self._lr["variational"])},
            {"params": [self.w], "lr": float(self._lr["output_scaling"])},
        ]

    # -- param accounting (be transparent: CNN does the heavy lifting) -------
    def param_counts(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        if self.head_kind == "classical":
            hp = sum(p.numel() for p in self.classical_head.parameters())
            return {"cnn_trainable": enc, "vqc_circuit": 0, "lam": 0, "w": 0,
                    "head_classical": hp, "total_trainable": self.num_trainable_params()}
        vqc = sum(p.numel() for p in self.vqc.parameters())
        return {"cnn_trainable": enc, "vqc_circuit": vqc,
                "lam": self.lam.numel(), "w": self.w.numel(),
                "total_trainable": self.num_trainable_params()}

    def loggable_scalars(self) -> dict[str, float]:
        if self.head_kind == "classical":
            return {"w0": 0.0, "w1": 0.0, "w2": 0.0, "lam_mean": 0.0}
        out = {f"w{i}": float(v) for i, v in enumerate(self.w.detach().cpu().numpy())}
        out["lam_mean"] = float(self.lam.detach().mean())
        return out


def load_agent(ckpt_path: str) -> "DinoQFunction":
    """Rebuild a :class:`DinoQFunction` from a backbone-neutral checkpoint and load its weights.

    Uses ``meta['constructor']`` (n_qubits/encoder/observable/...) to reconstruct the exact
    class, then restores the full ``state_dict``. Session B can also read the same file.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ctor = ckpt.get("meta", {}).get("constructor", {})
    model = DinoQFunction(**ctor)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Smoke test:  python -m experiments.dino.model [--encoder trainable_cnn|pretrained|both]
# Builds the model, does one forward + backward on a random [B,4,84,84] uint8 batch, and
# prints output shape + that grads are ALIVE on CNN, lam, circuit weights, and w.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser(description="CNN→VQC dino model smoke test.")
    ap.add_argument("--encoder", default="both", choices=["trainable_cnn", "pretrained", "both"])
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--observable", default="zz", choices=["zz", "z"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    encoders = ["trainable_cnn", "pretrained"] if args.encoder == "both" else [args.encoder]
    for enc in encoders:
        try:
            m = DinoQFunction(n_qubits=args.n_qubits, n_actions=3, encoder=enc,
                              observable=args.observable, seed=0)
        except Exception as e:
            print(f"[{enc:13s}] SKIPPED ({type(e).__name__}: {e})")
            continue
        m.train()
        states = torch.randint(0, 256, (args.batch, 4, 84, 84), dtype=torch.uint8)
        q = m(states)
        m.zero_grad()
        q.sum().backward()
        cnn_p = next(p for p in m.encoder.parameters() if p.requires_grad)
        checks = {"cnn": cnn_p, "lam": m.lam, "vqc": m.vqc.weights, "w": m.w}
        alive = {k: (p.grad is not None and p.grad.abs().sum().item() > 0) for k, p in checks.items()}
        pc = m.param_counts()
        print(f"[{enc:13s}] out {tuple(q.shape)} | grads alive {alive} | "
              f"params: CNN {pc['cnn_trainable']:,} vs VQC {pc['vqc_circuit']} "
              f"(+lam {pc['lam']}, w {pc['w']}) | Q range [{q.min():.3f},{q.max():.3f}]")
