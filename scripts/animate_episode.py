"""
Animate one episode, keeping straight what the agent sees and what it does not.

Panels:
  left    the Bloch sphere: spin vector, trailing path, |0> target, sz=0 death
          circle, and a readout of sx, sy, sz.
  top-r   (QDQN only) the five measured observables <O_i>, one bar per action.
          These five numbers are the entire quantum output -- the Q-values are
          just w * (<O>+1)/2 -- so this is what the argmax actually decides on.
          The chosen action's bar is highlighted.
  mid-r   OBSERVATION: the components the agent is given.
  bot-r   ENV INTERNAL STATE: the components it is not. Under the sxy observation
          sz is hidden, yet sz alone sets the reward and the sz<0 termination, so
          these two panels are deliberately kept apart -- putting them on one
          axis would suggest the agent can see its own score.
  strip   the action taken at each step.

Panel titles adapt to the checkpoint's observation mode, so a `full` policy shows
sz as observed rather than hidden.

Policies:
  --policy greedy                              physics-aware one-step lookahead
  --policy idle                                no control, for contrast
  --policy checkpoint --ckpt results/....pt    a trained DQN, QDQN or PPO

The script searches evaluation seeds for an episode that survives all 500 steps;
if none do it animates the best attempt and labels it as such.

Output format follows the extension: .mp4 (ffmpeg from imageio-ffmpeg) or .gif.

Usage:
    python scripts/animate_episode.py --policy greedy --out figures/ep.mp4
    python scripts/animate_episode.py --policy checkpoint \
        --ckpt results/qdqn_anim_nr0.35_seed0.pt --out figures/ep_qdqn.mp4
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

# indices into the raw 6D observation [sx, sy, sz, dsx, dsy, dsz]
COMPONENTS = {0: r"$\langle\sigma_x\rangle$", 1: r"$\langle\sigma_y\rangle$",
              2: r"$\langle\sigma_z\rangle$"}
COMP_COLORS = {0: "#ff7f0e", 1: "#2ca02c", 2: "#1f77b4"}


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


def rollout(policy, sigma, seed, feat=None, net=None, algo="dqn", want_obs=False):
    """One episode. Records the measured observables too when asked."""
    env = QuantumSpinCartPoleEnv(seed=0, sigma_ou=sigma)
    obs, _ = env.reset(seed=seed)
    env._ou.reset(seed=seed)
    s = feat.reset(obs) if feat is not None else None
    greedy = GreedyPolicy(env) if policy == "greedy" else None
    if want_obs:
        from qdqn_internals import circuit_internals

    sx, sy, sz, acts, rew, raws, qs = [], [], [], [], [], [], []
    term = trunc = False
    while not (term or trunc):
        if policy == "greedy":
            a = greedy.act(env._state.full().ravel())
        elif policy == "idle":
            a = IDLE
        elif want_obs:
            _, _psi, q, raw = circuit_internals(net, s)
            raws.append(raw); qs.append(q)
            a = int(np.argmax(q))
        else:
            a = greedy_action(net, s, algo)
        obs, r, term, trunc, info = env.step(a)
        if feat is not None:
            s = feat.step(obs, a)
        sx.append(info["sx"]); sy.append(info["sy"]); sz.append(info["sz"])
        acts.append(a); rew.append(r)

    out = {
        "sx": np.array(sx), "sy": np.array(sy), "sz": np.array(sz),
        "a": np.array(acts), "r": np.array(rew),
        "survived": bool(trunc and not term), "ret": float(np.sum(rew)),
    }
    if raws:
        out["obs_exp"] = np.array(raws)      # [T, 5] measured <O_i>
        out["q"] = np.array(qs)              # [T, 5] w * (<O>+1)/2, the decision
    return out


def find_successful(policy, sigma, feat, net, seeds, algo="dqn", want_obs=False):
    best = None
    for sd in seeds:
        ep = rollout(policy, sigma, sd, feat, net, algo, want_obs)
        print(f"  seed {sd}: return {ep['ret']:7.2f}  len {len(ep['a']):3}  "
              f"{'SURVIVED' if ep['survived'] else 'died'}")
        if ep["survived"]:
            return sd, ep
        if best is None or ep["ret"] > best[1]["ret"]:
            best = (sd, ep)
    return best


def _writer(out: Path, fps: int):
    """MP4 through the ffmpeg binary imageio-ffmpeg ships; GIF through Pillow."""
    if out.suffix.lower() == ".mp4":
        import imageio_ffmpeg
        from matplotlib.animation import FFMpegWriter

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return FFMpegWriter(fps=fps, bitrate=4000,
                            extra_args=["-pix_fmt", "yuv420p", "-vcodec", "libx264"])
    return PillowWriter(fps=fps)


def animate(ep, seed, label, out, seen=(0, 1, 2), stride=2, fps=25, dpi=130):
    """`seen` lists which of sx(0), sy(1), sz(2) the agent actually receives."""
    sx, sy, sz, acts = ep["sx"], ep["sy"], ep["sz"], ep["a"]
    comp = {0: sx, 1: sy, 2: sz}
    obs_exp = ep.get("obs_exp")
    hidden = [c for c in (0, 1, 2) if c not in seen]
    n = len(sz)
    frames = list(range(0, n, stride)) + [n - 1]

    rows = (1 if obs_exp is not None else 0) + (1 if seen else 0) + \
           (1 if hidden else 0) + 1
    heights = ([1.15] if obs_exp is not None else []) + \
              ([1.0] if seen else []) + ([1.0] if hidden else []) + [0.55]

    fig = plt.figure(figsize=(13.5, 2.05 * rows + 1.4))
    gs = fig.add_gridspec(rows, 2, width_ratios=[1.0, 1.2],
                          height_ratios=heights, hspace=0.34, wspace=0.20)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")

    r = 0
    axo = None
    if obs_exp is not None:
        axo = fig.add_subplot(gs[r, 1]); r += 1
    axseen = fig.add_subplot(gs[r, 1]) if seen else None
    if seen:
        r += 1
    axhid = fig.add_subplot(gs[r, 1]) if hidden else None
    if hidden:
        r += 1
    axa = fig.add_subplot(gs[r, 1])

    # --- Bloch sphere --------------------------------------------------------
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax3d.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                        color="0.86", lw=0.4)
    th = np.linspace(0, 2 * np.pi, 160)
    ax3d.plot(np.cos(th), np.sin(th), 0, color="#d62728", lw=1.3, alpha=0.75)
    ax3d.scatter([0], [0], [1], color="#2ca02c", s=55, depthshade=False)
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel(r"$\langle\sigma_x\rangle$", fontsize=9, labelpad=-2)
    ax3d.set_ylabel(r"$\langle\sigma_y\rangle$", fontsize=9, labelpad=-2)
    ax3d.set_zlabel(r"$\langle\sigma_z\rangle$", fontsize=9, labelpad=-2)
    # sparse ticks: the default density collides with the axis labels
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_ticks([-1, 0, 1])
    ax3d.tick_params(labelsize=7, pad=-1)
    ax3d.view_init(elev=20, azim=-62)
    # Legend as 2D text in a corner. In-scene 3D labels ride on top of the
    # sphere and the trajectory as the view fills up.
    # under the readout box: the bottom-left corner collides with the x label
    ax3d.text2D(0.0, 0.845,
                "$\\bullet$ |0> target\n$-$ sz = 0 death",
                transform=ax3d.transAxes, fontsize=8.5, va="top", color="0.3")

    trail, = ax3d.plot([], [], [], lw=1.2, color="#1f77b4", alpha=0.8)
    head = ax3d.scatter([], [], [], color="#1f77b4", s=70, depthshade=False)
    arrow = ax3d.plot([], [], [], lw=2.2, color="#1f77b4")[0]
    readout = ax3d.text2D(0.0, 1.0, "", transform=ax3d.transAxes, fontsize=10,
                          family="monospace", va="top",
                          bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                    ec="0.8", alpha=0.85))

    # --- the five measured observables ---------------------------------------
    bars = None
    if obs_exp is not None:
        bars = axo.bar(np.arange(5), np.zeros(5), width=0.62,
                       color=ACTION_COLORS, edgecolor="none")
        axo.axhline(0, color="0.5", lw=0.8)
        # headroom above +1 so the annotation never sits on top of a bar
        axo.set_ylim(-1.08, 1.62)
        axo.set_yticks([-1, 0, 1])
        axo.set_xticks(range(5)); axo.set_xticklabels(ACTION_NAMES, fontsize=9)
        axo.set_ylabel(r"$\langle O_i\rangle$", fontsize=10)
        axo.set_title("measured observables  (one ZZ per action; the whole quantum output)",
                      fontsize=9.5, pad=4)
        axo.grid(alpha=0.2, lw=0.5, axis="y")
        pick = axo.text(0.5, 0.985, "", transform=axo.transAxes, fontsize=9,
                        ha="center", va="top", family="monospace")

    # --- observation vs hidden internal state --------------------------------
    def _trace_axes(ax, comps, title, color_title):
        ax.axhline(0, color="#d62728", lw=0.9, ls="--")
        lines = {}
        for c in comps:
            lines[c], = ax.plot([], [], lw=1.4, color=COMP_COLORS[c],
                                label=COMPONENTS[c])
            ax.plot(comp[c], color="0.90", lw=0.6, zorder=0)
        ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, n)
        ax.set_title(title, fontsize=9.5, color=color_title, pad=3)
        ax.legend(fontsize=8, ncol=len(comps), loc="lower left")
        ax.tick_params(labelbottom=False, labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        return lines

    seen_lines = hidden_lines = {}
    if seen:
        seen_lines = _trace_axes(
            axseen, seen, "OBSERVATION  -- given to the agent", "#1a1a1a")
    if hidden:
        hidden_lines = _trace_axes(
            axhid, hidden,
            "ENV INTERNAL STATE  -- hidden from the agent, but sets reward and death",
            "#b32d2d")

    # --- action strip --------------------------------------------------------
    axa.scatter(np.arange(n), acts, s=7,
                c=[ACTION_COLORS[a] for a in acts], alpha=0.25)
    adot = axa.scatter([], [], s=48, c="k", zorder=5)
    axa.set_yticks(range(5)); axa.set_yticklabels(ACTION_NAMES, fontsize=8)
    axa.set_xlabel("step"); axa.set_ylabel("action", fontsize=9)
    axa.set_xlim(0, n); axa.grid(alpha=0.25, lw=0.5)
    axa.tick_params(labelsize=8)

    title = fig.suptitle("", fontsize=12)

    def update(i):
        k = max(1, i + 1)
        lo = max(0, k - 120)
        trail.set_data(sx[lo:k], sy[lo:k]); trail.set_3d_properties(sz[lo:k])
        head._offsets3d = ([sx[k - 1]], [sy[k - 1]], [sz[k - 1]])
        arrow.set_data([0, sx[k - 1]], [0, sy[k - 1]])
        arrow.set_3d_properties([0, sz[k - 1]])
        readout.set_text(f"sx {sx[k-1]:+.3f}\nsy {sy[k-1]:+.3f}\nsz {sz[k-1]:+.3f}")

        t = np.arange(k)
        for c, ln in {**seen_lines, **hidden_lines}.items():
            ln.set_data(t, comp[c][:k])
        adot.set_offsets([[k - 1, acts[k - 1]]])

        if bars is not None:
            o = obs_exp[k - 1]
            raw_best = int(np.argmax(o))
            for j, (rect, h) in enumerate(zip(bars, o)):
                rect.set_height(h)
                rect.set_alpha(1.0 if j == acts[k - 1] else 0.35)
            # The decision is argmax of w*(<O>+1)/2, not of <O>. w differs per
            # action (about 3% spread here), so when the observables are this
            # close the output scaling alone can pick a different action.
            note = ("" if raw_best == acts[k - 1]
                    else f"  (max <O> is {ACTION_NAMES[raw_best]}; w flips it)")
            pick.set_text(f"chosen -> {ACTION_NAMES[acts[k-1]]}{note}\n"
                          f"spread of <O>  {np.ptp(o):.4f}")

        idle = float(np.mean(acts[:k] == IDLE))
        title.set_text(f"{label}   |   seed {seed}   |   step {k}/{n}   "
                       f"|   return {np.sum(ep['r'][:k]):.1f}   |   idle {idle*100:.0f}%")
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=_writer(out, fps), dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/fps:.1f}s, "
          f"{out.stat().st_size/1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["greedy", "idle", "checkpoint"], default="greedy")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--noise-rabi", type=float, default=0.35)
    p.add_argument("--seeds", default="1000-1029")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--observables", action=argparse.BooleanOptionalAction, default=None,
                   help="show the five measured <O_i>; defaults on for a VQC")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sigma = sigma_for_noise_rabi(args.noise_rabi)
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = range(lo, hi + 1)

    feat = net = None
    algo, want_obs = "dqn", False
    seen = (0, 1, 2)      # greedy and idle are described by the full state
    if args.policy == "checkpoint":
        if not args.ckpt:
            p.error("--policy checkpoint needs --ckpt")
        net, feat, ck = load_checkpoint(args.ckpt)
        algo = ck.get("algo", "dqn")
        is_vqc = ck.get("model") == "vqc"
        want_obs = is_vqc if args.observables is None else args.observables
        if want_obs and not is_vqc:
            p.error("--observables only applies to a VQC checkpoint")
        # which of sx, sy, sz this observation mode actually hands over
        seen = tuple(c for c in (0, 1, 2) if c in feat.idx)
        if algo == "ppo":
            label = f"PPO (MLP-{ck['width']})"
        elif is_vqc:
            label = f"QDQN ({ck['input_dim']}q x {ck['n_layers']}L)"
        else:
            label = f"DQN MLP-{ck['width']}"
        label += f"  {ck['obs']}  {ck['steps']//1000}k steps"
        print(f"loaded {args.ckpt}: {label}")
        print(f"  final eval {ck['final']['return_mean']:.2f} "
              f"({ck['final']['survival_rate']*100:.0f}% survive)")
        names = ["sx", "sy", "sz"]
        print(f"  agent observes: {', '.join(names[c] for c in seen) or 'none'}"
              f"   hidden: {', '.join(names[c] for c in (0, 1, 2) if c not in seen) or 'none'}")
    elif args.policy == "idle":
        label = "always IDLE (no control)"
    else:
        label = "greedy (one-step lookahead)"

    print(f"searching {lo}-{hi} for a successful episode at noise/Rabi {args.noise_rabi}:")
    seed, ep = find_successful(args.policy, sigma, feat, net, seeds, algo, want_obs)
    if not ep["survived"]:
        print(f"  !! none of these seeds survived; animating the best "
              f"(seed {seed}, return {ep['ret']:.1f}) and labelling it as such")
        label += "  [best attempt, did not survive]"

    out = args.out or f"figures/episode_{args.policy}.mp4"
    animate(ep, seed, label, out, seen=seen, stride=args.stride, fps=args.fps, dpi=args.dpi)


if __name__ == "__main__":
    main()
