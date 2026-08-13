"""Generate the figure/table suite for the dino-QRL writeup.

Produces (into figures/):
  * dino_score_distribution.png  — greedy (eps=0) eval score distribution: random vs classical vs VQC
  * dino_filmstrip.png           — a strip of frames from a VQC rollout (the agent clearing cacti)
  * dino_qvalue_trace.png        — the VQC head's 3 Q-values over a rollout + chosen action + clears
  * dino_param_bar.png           — CNN (perception) vs VQC (quantum head) trainable-param counts
  * dino_comparison_table.md     — the results table (also printed)

Run:  python -m experiments.dino.visualize
Evals are greedy (epsilon=0), the honest measure (training scores are depressed by the 5% eps floor).
"""
from __future__ import annotations

import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import experiments.dino  # noqa: F401
from experiments.dino.dino_game import DinoGame
from experiments.dino.env import FRAME_SKIP, N_STACK, make_dino_env
from experiments.dino.model import load_agent
from experiments.dino.record_demo import _annotate, _greedy

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = os.path.join(_ROOT, "figures")
RES = os.path.join(_ROOT, "results")
ACTIONS = ["RUN", "JUMP", "DUCK"]


# --------------------------------------------------------------------------- evals
def eval_scores(policy, n=15, seed0=30000, max_steps=1600):
    """policy(obs)->action; return list of greedy episode scores over fixed seeds."""
    scores = []
    for i in range(n):
        env = make_dino_env(max_steps=max_steps, seed=seed0 + i)
        obs, info = env.reset(seed=seed0 + i)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        scores.append(info["score"])
    return scores


def greedy_policy(model):
    def act(obs):
        with torch.no_grad():
            return int(torch.argmax(model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)), 1).item())
    return act


# --------------------------------------------------------------------------- rollout capture
def vqc_rollout(model, seed=20000, max_steps=340):
    """One greedy rollout; capture per-decision Q-values/action + annotated frames + clear steps."""
    g = DinoGame(seed=seed); g.reset(seed=seed)
    stack = deque([g.render_gray84()] * N_STACK, maxlen=N_STACK)
    qs, acts, frames, clears = [], [], [], []
    a, q = 0, [0, 0, 0]
    prev_cleared = 0
    for t in range(max_steps):
        if t % FRAME_SKIP == 0:
            a, q = _greedy(model, np.stack(stack))
            qs.append(q); acts.append(a)
        frames.append((t, g.render_rgb(), g.score, a, q))
        crashed = g.step(a)
        if g.obstacles_cleared > prev_cleared:
            clears.append(t); prev_cleared = g.obstacles_cleared
        stack.append(g.render_gray84())
        if crashed:
            break
    return np.array(qs), np.array(acts), frames, clears


# --------------------------------------------------------------------------- figures
def fig_score_distribution(dist, out):
    labels = list(dist.keys())
    data = [dist[k] for k in labels]
    means = [np.mean(d) for d in data]
    colors = ["#999999", "#1f77b4", "#d62728"][: len(labels)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (d, c) in enumerate(zip(data, colors)):
        x = np.random.default_rng(i).normal(i, 0.06, size=len(d))
        ax.scatter(x, d, color=c, alpha=0.7, s=40, zorder=3)
        ax.hlines(means[i], i - 0.25, i + 0.25, color=c, lw=3, zorder=4)
        ax.text(i, means[i], f" {means[i]:.0f}", va="center", fontweight="bold", color=c)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("greedy episode score (frames survived)")
    ax.set_title("Dino greedy (ε=0) score distribution — identical CNN, different Q-head")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def fig_filmstrip(frames, clears, out, n=6):
    # pick n frames spread across the rollout, preferring jump moments
    idx = np.linspace(0, len(frames) - 1, num=n).astype(int)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.7 * n))
    for ax, i in zip(axes, idx):
        t, rgb, score, a, q = frames[i]
        ax.imshow(rgb)
        ax.set_title(f"t={t}  score={score}  VQC head → {ACTIONS[a]}   Q={np.round(q,2).tolist()}",
                     fontsize=9, loc="left")
        ax.axis("off")
    fig.suptitle("VQC agent playing (a quantum circuit picks each action)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print("saved", out)


def fig_qtrace(qs, acts, clears, out):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    steps = np.arange(len(qs)) * FRAME_SKIP
    for j, name in enumerate(ACTIONS):
        ax.plot(steps, qs[:, j], label=f"Q({name})", lw=1.8)
    jump_dec = np.where(acts == 1)[0]
    ax.scatter(jump_dec * FRAME_SKIP, qs[jump_dec, 1], color="k", s=18, zorder=5, label="chose JUMP")
    for c in clears:
        ax.axvline(c, color="green", alpha=0.25, lw=1)
    ax.plot([], [], color="green", alpha=0.4, label="obstacle cleared")
    ax.set_xlabel("game frame"); ax.set_ylabel("Q-value from ⟨ZZ⟩ (VQC head)")
    ax.set_title("The quantum head deciding: Q-values over a rollout (JUMP spikes at each cactus)")
    ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def fig_param_bar(pc, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = ["CNN encoder\n(perception)", "VQC circuit\n(quantum head)", "λ + w\n(scalings)"]
    vals = [pc["cnn_trainable"], pc["vqc_circuit"], pc["lam"] + pc["w"]]
    bars = ax.bar(names, vals, color=["#1f77b4", "#d62728", "#2ca02c"])
    ax.set_yscale("log"); ax.set_ylabel("trainable parameters (log)")
    ax.set_title("Where the parameters live: the quantum head is tiny")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f" {v:,}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


def main():
    os.makedirs(FIG, exist_ok=True)
    vqc = load_agent(os.path.join(RES, "dino_vqc.pt"))
    print(f"VQC agent: head={vqc.head_kind} qubits={vqc.n_qubits} obs={vqc.observable_kind}")

    # --- score distributions (greedy, eps=0) ---
    dist = {}
    rng = np.random.default_rng(0)
    dist["random"] = eval_scores(lambda o: int(rng.integers(3)), n=15)
    dist["classical\nLinear head"] = eval_scores(greedy_policy(load_agent(os.path.join(RES, "dino_classical.pt"))), n=15) \
        if os.path.exists(os.path.join(RES, "dino_classical.pt")) else None
    dist["VQC head\n(quantum)"] = eval_scores(greedy_policy(vqc), n=15)
    dist = {k: v for k, v in dist.items() if v is not None}
    fig_score_distribution(dist, os.path.join(FIG, "dino_score_distribution.png"))

    # --- rollout-based figures ---
    qs, acts, frames, clears = vqc_rollout(vqc, seed=20000, max_steps=340)
    fig_filmstrip(frames, clears, os.path.join(FIG, "dino_filmstrip.png"))
    fig_qtrace(qs, acts, clears, os.path.join(FIG, "dino_qvalue_trace.png"))
    fig_param_bar(vqc.param_counts(), os.path.join(FIG, "dino_param_bar.png"))

    # --- comparison table ---
    lines = ["| agent (identical CNN + task) | mean | median | best | min | n |",
             "|---|---|---|---|---|---|"]
    for k, v in dist.items():
        v = np.array(v)
        lines.append(f"| {k.replace(chr(10),' ')} | {v.mean():.0f} | {np.median(v):.0f} | "
                     f"{v.max():.0f} | {v.min():.0f} | {len(v)} |")
    table = "\n".join(lines)
    with open(os.path.join(FIG, "dino_comparison_table.md"), "w", encoding="utf-8") as f:
        f.write("# Dino greedy (ε=0) eval — score = frames survived\n\n" + table + "\n")
    print("\n" + table)
    print("\nsaved", os.path.join(FIG, "dino_comparison_table.md"))


if __name__ == "__main__":
    main()
