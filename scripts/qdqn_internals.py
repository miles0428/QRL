"""
Look inside the QDQN while it plays: what the circuit is fed, and what it holds.

Three things are recorded at every step of one episode and drawn against time:

  encoding angles   what the circuit actually sees. Not the raw observation --
                    the observation goes through arctan, then is tiled once per
                    layer (per-layer encoding), then scaled elementwise by the
                    trainable `lam`. These 45 numbers are the RY angles the
                    9 qubits are rotated by, 9 per layer across 5 layers.

  |amplitude|^2     the 512 basis-state probabilities of the state the circuit
                    holds just before measurement. This is the register the
                    five ZZ observables are read from.

  Q-values          the five outputs, w * <O> after rescaling -- what the argmax
                    is taken over.

Together these answer "what does the quantum part actually do here": whether the
state stays near a product state or spreads, whether all five layers contribute,
and how sharply the Q-values separate the chosen action.

Usage:
    python scripts/qdqn_internals.py --ckpt results/qdqn_anim_nr0.35_seed0.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from animate_episode import load_checkpoint  # noqa: E402
from greedy_baseline import sigma_for_noise_rabi  # noqa: E402
from qdqn import torch_statevector as tsv  # noqa: E402
from qdqn.base import normalize_observation  # noqa: E402
from quantum_spin_cartpole import QuantumSpinCartPoleEnv  # noqa: E402

ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "IDLE"]


def circuit_internals(net, s: np.ndarray):
    """
    Encoding angles, statevector, measured observables and Q-values.

    `raw` is the five <O_i> the circuit is actually measured on, one per action,
    each in [-1, 1]. Those five numbers are the entire quantum output: the
    Q-values are just w * (raw+1)/2. The 512 amplitudes exist inside the
    simulation but are never read.
    """
    with torch.no_grad():
        x = torch.from_numpy(s).unsqueeze(0)
        normalized = normalize_observation(x)
        if net.per_layer_encoding and net.lam.numel() != normalized.shape[1]:
            normalized = normalized.tile(1, net.lam.numel() // normalized.shape[1])
        scaled = net.lam * normalized                      # the RY angles
        raw, psi = tsv.simulate(net.vqc._compiled, scaled, net.vqc.weight,
                                net.vqc._obs, return_state=True)
        out = (raw + 1.0) / 2.0 if net.output_rescaling else raw
        q = net.w * out
    return (scaled[0].numpy(), psi[0].numpy(), q[0].numpy(), raw[0].numpy())


def rollout(net, feat, sigma, seed):
    env = QuantumSpinCartPoleEnv(seed=0, sigma_ou=sigma)
    obs, _ = env.reset(seed=seed)
    env._ou.reset(seed=seed)
    s = feat.reset(obs)
    ANG, PSI, Q, SZ, ACT, R = [], [], [], [], [], []
    term = trunc = False
    while not (term or trunc):
        ang, psi, q, _raw = circuit_internals(net, s)
        a = int(np.argmax(q))
        obs, r, term, trunc, info = env.step(a)
        s = feat.step(obs, a)
        ANG.append(ang); PSI.append(psi); Q.append(q)
        SZ.append(info["sz"]); ACT.append(a); R.append(r)
    return dict(ang=np.array(ANG), psi=np.array(PSI), q=np.array(Q),
                sz=np.array(SZ), a=np.array(ACT), r=np.array(R),
                survived=bool(trunc and not term))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/qdqn_anim_nr0.35_seed0.pt")
    p.add_argument("--noise-rabi", type=float, default=0.35)
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--out", default="figures/qdqn_internals.png")
    args = p.parse_args()

    net, feat, ck = load_checkpoint(args.ckpt)
    sigma = sigma_for_noise_rabi(args.noise_rabi)
    lo, hi = (int(v) for v in args.seeds.split("-"))

    for sd in range(lo, hi + 1):
        ep = rollout(net, feat, sigma, sd)
        print(f"  seed {sd}: len {len(ep['a']):3}  return {ep['r'].sum():7.2f}  "
              f"{'SURVIVED' if ep['survived'] else 'died'}")
        if ep["survived"]:
            break

    n_qubits, n_layers = ck["input_dim"], ck["n_layers"]
    prob = np.abs(ep["psi"]) ** 2
    T = len(ep["a"])

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.5, 2.2, 1.3],
                                          "hspace": 0.16})

    # --- context -------------------------------------------------------------
    ax = axes[0]
    ax.axhline(0, color="#d62728", lw=0.9, ls="--")
    ax.plot(ep["sz"], lw=1.0, color="#1f77b4")
    ax.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"QDQN internals -- {n_qubits} qubits x {n_layers} layers, "
                 f"{ck['obs']}, {ck['steps']//1000}k steps   "
                 f"(episode return {ep['r'].sum():.1f}, {T} steps)", fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)

    # --- encoding angles -----------------------------------------------------
    ax = axes[1]
    v = np.abs(ep["ang"]).max()
    im = ax.imshow(ep["ang"].T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, extent=[0, T, 0, ep["ang"].shape[1]])
    for L in range(1, n_layers):
        ax.axhline(L * n_qubits, color="k", lw=0.6, alpha=0.4)
    ax.set_yticks([L * n_qubits + n_qubits / 2 for L in range(n_layers)])
    ax.set_yticklabels([f"layer {L}" for L in range(n_layers)], fontsize=8)
    ax.set_ylabel("RY encoding angle\n(lam x arctan(obs))", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025).set_label("rad", fontsize=8)

    # --- probability amplitudes ---------------------------------------------
    ax = axes[2]
    im = ax.imshow(prob.T, aspect="auto", origin="lower", cmap="magma",
                   norm=matplotlib.colors.LogNorm(vmin=max(prob.min(), 1e-6), vmax=prob.max()),
                   extent=[0, T, 0, prob.shape[1]])
    ax.set_ylabel(f"basis state (0..{prob.shape[1]-1})", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025).set_label(r"$|c_i|^2$", fontsize=8)
    part = 1.0 / np.sum(prob ** 2, axis=1)      # participation ratio
    ax2 = ax.twinx()
    ax2.plot(part, color="#39ff14", lw=0.9, alpha=0.85)
    ax2.set_ylabel("participation ratio", color="#2ca02c", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)

    # --- Q-values ------------------------------------------------------------
    ax = axes[3]
    for i, nm in enumerate(ACTION_NAMES):
        ax.plot(ep["q"][:, i], lw=0.9, label=nm)
    ax.set_ylabel("Q value", fontsize=9)
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=5, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    print()
    print(f"encoding angles: {ep['ang'].shape[1]} of them "
          f"({n_qubits} qubits x {n_layers} layers), range "
          f"[{ep['ang'].min():+.3f}, {ep['ang'].max():+.3f}] rad")
    lam = net.lam.detach().numpy()
    print(f"lam magnitude per layer: " +
          ", ".join(f"L{L} {np.abs(lam[L*n_qubits:(L+1)*n_qubits]).mean():.3f}"
                    for L in range(n_layers)))
    print(f"participation ratio: mean {part.mean():.1f} of {prob.shape[1]} basis states "
          f"(1 = single basis state, {prob.shape[1]} = uniform)")
    print(f"top basis state holds {prob.max(axis=1).mean()*100:.1f}% of the probability on average")
    print(f"Q spread (max-min) mean {np.ptp(ep['q'], axis=1).mean():.3f}, "
          f"w = {net.w.detach().numpy().round(2)}")


if __name__ == "__main__":
    main()
