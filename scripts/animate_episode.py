"""
Animate one successful episode: the spin on the Bloch sphere, plus the traces.

Renders a GIF with three panels:
  left   the Bloch sphere, with the trajectory trailing behind the spin and the
         northern hemisphere (the survival region) shaded
  top    <sigma_z> against time, with the sz=0 death line marked
  bottom the action taken at each step, idle greyed out

Policies:
  --policy greedy            physics-aware one-step lookahead
  --policy checkpoint --ckpt results/....pt    a trained DQN / QDQN

"Successful" means the episode ran the full 500 steps without terminating. The
script searches evaluation seeds until it finds one, so the animation shows what
the policy does when it works rather than a random draw that may die early.

Usage:
    python scripts/animate_episode.py --policy greedy
    python scripts/animate_episode.py --policy checkpoint --ckpt results/qdqn_anim_nr0.35_seed0.pt
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
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greedy_baseline import IDLE, GreedyPolicy, sigma_for_noise_rabi  # noqa: E402
from quantum_spin_cartpole import QuantumSpinCartPoleEnv  # noqa: E402
from train_dqn import Featurizer, build_qnet  # noqa: E402

ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "IDLE"]
ACTION_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#bbbbbb"]


def load_checkpoint(path):
    """Rebuild whichever network the checkpoint came from -- DQN, QDQN or PPO."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    algo = ck.get("algo", "dqn")

    if algo == "ppo":
        from train_ppo import ActorCritic

        net = ActorCritic(ck["input_dim"], ck["width"])
    else:
        class A:
            pass

        a = A()
        for k in ("model", "width", "n_layers", "reuploading", "per_layer_encoding", "seed"):
            setattr(a, k, ck[k])
        net = build_qnet(a, ck["input_dim"])

    net.load_state_dict(ck["state_dict"])
    net.eval()
    feat = Featurizer(ck["obs"], hist=ck.get("hist", 8))
    return net, feat, ck


def greedy_action(net, s, algo):
    """PPO returns (logits, value); DQN/QDQN return Q-values directly."""
    with torch.no_grad():
        out = net(torch.from_numpy(s).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.argmax(1).item())


def rollout(policy, sigma, seed, feat=None, net=None, algo="dqn"):
    """One episode. Returns traces and whether it survived to truncation."""
    env = QuantumSpinCartPoleEnv(seed=0, sigma_ou=sigma)
    obs, _ = env.reset(seed=seed)
    env._ou.reset(seed=seed)
    s = feat.reset(obs) if feat is not None else None
    greedy = GreedyPolicy(env) if policy == "greedy" else None

    sx, sy, sz, acts, rew = [], [], [], [], []
    term = trunc = False
    while not (term or trunc):
        if policy == "greedy":
            a = greedy.act(env._state.full().ravel())
        else:
            a = greedy_action(net, s, algo)
        obs, r, term, trunc, info = env.step(a)
        if feat is not None:
            s = feat.step(obs, a)
        sx.append(info["sx"]); sy.append(info["sy"]); sz.append(info["sz"])
        acts.append(a); rew.append(r)
    return {
        "sx": np.array(sx), "sy": np.array(sy), "sz": np.array(sz),
        "a": np.array(acts), "r": np.array(rew),
        "survived": bool(trunc and not term), "ret": float(np.sum(rew)),
    }


def find_successful(policy, sigma, feat, net, seeds, algo="dqn", verbose=True):
    best = None
    for sd in seeds:
        ep = rollout(policy, sigma, sd, feat, net, algo)
        if verbose:
            print(f"  seed {sd}: return {ep['ret']:7.2f}  len {len(ep['a']):3}  "
                  f"{'SURVIVED' if ep['survived'] else 'died'}")
        if ep["survived"]:
            return sd, ep
        if best is None or ep["ret"] > best[1]["ret"]:
            best = (sd, ep)
    return best  # nothing survived; animate the best attempt and say so


def animate(ep, seed, label, out, stride=2, fps=25, dpi=70, figsize=(9.5, 4.3)):
    sx, sy, sz, acts = ep["sx"], ep["sy"], ep["sz"], ep["a"]
    n = len(sz)
    frames = list(range(0, n, stride)) + [n - 1]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[3, 1],
                          hspace=0.12, wspace=0.18)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    axz = fig.add_subplot(gs[0, 1])
    axa = fig.add_subplot(gs[1, 1], sharex=axz)

    # --- Bloch sphere scaffolding -------------------------------------------
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax3d.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                        color="0.85", lw=0.4)
    th = np.linspace(0, 2 * np.pi, 120)
    ax3d.plot(np.cos(th), np.sin(th), 0, color="#d62728", lw=1.2, alpha=0.7)
    ax3d.scatter([0], [0], [1], color="#2ca02c", s=45, depthshade=False)
    ax3d.text(0, 0, 1.22, "|0>  target", color="#2ca02c", fontsize=8, ha="center")
    ax3d.text(1.15, 0, -0.15, "sz = 0  death", color="#d62728", fontsize=8)
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("<sx>"); ax3d.set_ylabel("<sy>"); ax3d.set_zlabel("<sz>")
    ax3d.view_init(elev=22, azim=-60)

    trail, = ax3d.plot([], [], [], lw=1.1, color="#1f77b4", alpha=0.75)
    head = ax3d.scatter([], [], [], color="#1f77b4", s=60, depthshade=False)
    arrow = ax3d.plot([], [], [], lw=2.0, color="#1f77b4")[0]

    # --- sz trace ------------------------------------------------------------
    axz.axhline(0, color="#d62728", lw=1.0, ls="--")
    axz.plot(sz, color="0.85", lw=0.8)                 # full path, ghosted
    zline, = axz.plot([], [], color="#1f77b4", lw=1.3)
    zdot, = axz.plot([], [], "o", color="#1f77b4", ms=5)
    axz.set_ylim(-1.05, 1.05); axz.set_ylabel(r"$\langle\sigma_z\rangle$")
    axz.tick_params(labelbottom=False)
    axz.grid(alpha=0.25, lw=0.5)

    # --- action strip --------------------------------------------------------
    axa.scatter(np.arange(n), acts, s=6,
                c=[ACTION_COLORS[a] for a in acts], alpha=0.25)
    adot = axa.scatter([], [], s=42, c="k", zorder=5)
    axa.set_yticks(range(5)); axa.set_yticklabels(ACTION_NAMES, fontsize=8)
    axa.set_xlabel("step"); axa.set_ylabel("action", fontsize=9)
    axa.set_xlim(0, n); axa.grid(alpha=0.25, lw=0.5)

    title = fig.suptitle("", fontsize=11)

    def update(i):
        k = max(1, i + 1)
        lo = max(0, k - 120)                            # trailing window
        trail.set_data(sx[lo:k], sy[lo:k]); trail.set_3d_properties(sz[lo:k])
        head._offsets3d = ([sx[k - 1]], [sy[k - 1]], [sz[k - 1]])
        arrow.set_data([0, sx[k - 1]], [0, sy[k - 1]])
        arrow.set_3d_properties([0, sz[k - 1]])
        t = np.arange(k)
        zline.set_data(t, sz[:k])
        zdot.set_data([k - 1], [sz[k - 1]])
        adot.set_offsets([[k - 1, acts[k - 1]]])
        idle_frac = float(np.mean(acts[:k] == IDLE))
        title.set_text(f"{label}  |  seed {seed}  |  step {k}/{n}  "
                       f"|  return {np.sum(ep['r'][:k]):.1f}  "
                       f"|  <sz> {sz[k-1]:+.3f}  |  idle {idle_frac*100:.0f}%")
        return trail, head, arrow, zline, zdot, adot, title

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["greedy", "checkpoint"], default="greedy")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--noise-rabi", type=float, default=0.35)
    p.add_argument("--seeds", default="1000-1029", help="eval seeds to search, e.g. 1000-1029")
    p.add_argument("--stride", type=int, default=2, help="animate every Nth step")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--dpi", type=int, default=70)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sigma = sigma_for_noise_rabi(args.noise_rabi)
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = range(lo, hi + 1)

    feat = net = None
    algo = "dqn"
    if args.policy == "checkpoint":
        if not args.ckpt:
            p.error("--policy checkpoint needs --ckpt")
        net, feat, ck = load_checkpoint(args.ckpt)
        algo = ck.get("algo", "dqn")
        if algo == "ppo":
            label = f"PPO (MLP-{ck['width']})"
        elif ck.get("model") == "vqc":
            label = f"QDQN ({ck['input_dim']}q x {ck['n_layers']}L)"
        else:
            label = f"DQN MLP-{ck['width']}"
        label += f"  {ck['obs']}  {ck['steps']//1000}k steps"
        print(f"loaded {args.ckpt}: {label}")
        print(f"  its recorded final eval: {ck['final']['return_mean']:.2f} "
              f"({ck['final']['survival_rate']*100:.0f}% survive)")
    else:
        label = "greedy (one-step lookahead)"

    print(f"searching {lo}-{hi} for a successful episode at noise/Rabi {args.noise_rabi}:")
    seed, ep = find_successful(args.policy, sigma, feat, net, seeds, algo)
    if not ep["survived"]:
        print(f"  !! none of these seeds survived; animating the best "
              f"(seed {seed}, return {ep['ret']:.1f}) and labelling it as such")
        label += "  [best attempt, did not survive]"

    out = args.out or f"figures/episode_{args.policy}.gif"
    animate(ep, seed, label, out, stride=args.stride, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
