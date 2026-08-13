"""
DQN on QuantumSpinCartPole, with and without the sx/sy mask.

Three observation modes:

  full         [sx, sy, sz, dsx, dsy, dsz]           -- 6D, Markov in the spin state
  masked       [sz, dsz]                             -- 2D, azimuth removed
  masked_hist  masked + one-hot action, stacked k    -- 2D*k + 5*k

masked_hist exists because a memoryless agent provably cannot do better than
IDLE under the mask: IDLE is the identity operator, so sz alone never reveals
the azimuth phi, and the drive that would reveal it has negative one-step value.
Giving the net a window of (sz, dsz, action-taken) lets it in principle infer
phi from how sz responded to its own past drives. Without the action history the
window is useless, so the actions are stacked too.

Evaluation uses epsilon=0 on the same fixed seeds (1000..1000+N) that
greedy_baseline.py and greedy_masked.py use, so the numbers are directly
comparable to the greedy controllers.

Usage:
    python scripts/train_dqn.py --obs full        --noise-rabi 0.35 --seed 0
    python scripts/train_dqn.py --obs masked      --noise-rabi 0.35 --seed 0
    python scripts/train_dqn.py --obs masked_hist --noise-rabi 0.35 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# these runs are small MLPs; one thread each so many can run in parallel
torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantum_spin_cartpole.constants import NOISE_RABI_RATIO  # noqa: E402
from greedy_baseline import sigma_for_noise_rabi  # noqa: E402
from quantum_spin_cartpole import QuantumSpinCartPoleEnv  # noqa: E402

N_ACTIONS = 5

# Which components of the 6D observation [sx, sy, sz, dsx, dsy, dsz] each family
# of modes keeps. Suffixes _prevact / _hist are handled separately.
#
#   full    everything.
#   masked  sz and dsz only. Reward information is intact -- fidelity is a
#           function of sz -- but the azimuth is gone, so the agent cannot tell
#           +X from +Y.
#   sx      sx and dsx only. The harder mirror image: the agent can no longer
#           see sz at all, so it cannot see its own reward or how close it is to
#           the sz<0 termination. Still solvable in principle, because an X drive
#           leaves sx fixed while a Y drive moves sx in proportion to sz -- so a
#           Y probe reads out sz, sign included.
#   sxy     sx, sy and their deltas. Information-theoretically this hides
#           nothing while the episode is live: reset samples the upper
#           hemisphere and sz<0 terminates, so sz = +sqrt(1-sx^2-sy^2) exactly
#           (verified to 4e-14). But sufficiency is not learnability. The true
#           update of (sx, sy) depends on sz, so on this 2D projection the
#           dynamics are NOT linear -- a linear fitted model, which is exact on
#           the full Bloch vector, is misspecified here. This mode separates
#           "the information is present" from "this model class can use it".
OBS_INDICES = {
    "full": [0, 1, 2, 3, 4, 5],
    "masked": [2, 5],
    "sx": [0, 3],
    "sxy": [0, 1, 3, 4],
}

# Derived so that adding a family above automatically exposes it on the CLI --
# forgetting to update a hand-written choices list silently kills every run.
OBS_MODES = [f"{fam}{suf}" for fam in OBS_INDICES for suf in ("", "_prevact", "_hist")]


# ----------------------------------------------------------------------------
# Observation handling
# ----------------------------------------------------------------------------
class Featurizer:
    """Turns raw 6D env observations into the network input for a given mode."""

    def __init__(self, mode: str, hist: int = 8):
        self.mode = mode
        self.hist = hist if mode.endswith("_hist") else 1
        # _prevact: keep the Markov observation and append only the one-hot of
        # the action that produced it. The spin state is already Markov, so
        # stacking old states adds nothing; the single missing piece is which
        # action caused the observed delta, since delta = own action + noise.
        # Knowing the action lets the net subtract the known part and read off
        # the noise. 11 dims instead of 88, and no start-of-episode padding.
        self.with_action = mode.endswith("_hist") or mode.endswith("_prevact")
        self.family = mode.split("_")[0]
        self.idx = OBS_INDICES[self.family]
        base = len(self.idx)
        self.base = base
        self.dim = (base + (N_ACTIONS if self.with_action else 0)) * self.hist
        self._buf: deque = deque(maxlen=self.hist)

    def _core(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)[self.idx]

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self._buf.clear()
        core = self._core(obs)
        for _ in range(self.hist):
            self._buf.append((core, np.zeros(N_ACTIONS, dtype=np.float32)))
        return self._flat()

    def step(self, obs: np.ndarray, prev_action: int) -> np.ndarray:
        onehot = np.zeros(N_ACTIONS, dtype=np.float32)
        onehot[prev_action] = 1.0
        self._buf.append((self._core(obs), onehot))
        return self._flat()

    def _flat(self) -> np.ndarray:
        if not self.with_action:
            return self._buf[-1][0]
        if self.hist == 1:
            return np.concatenate(self._buf[-1])
        return np.concatenate([np.concatenate(x) for x in self._buf])


# ----------------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, dim: int, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


def default_observables(n_qubits: int, n_actions: int) -> list[str]:
    """
    One Pauli string per action, ZZ on a consecutive pair, striding and wrapping.

    Follows the qdqn-cartpole config's choice of two-qubit tensor products over
    single-qubit Z ("performance depends critically on observable choice"), just
    generalized past CartPole's 4 qubits / 2 actions.
    """
    out = []
    for i in range(n_actions):
        s = ["I"] * n_qubits
        s[(2 * i) % n_qubits] = "Z"
        s[(2 * i + 1) % n_qubits] = "Z"
        out.append("".join(s))
    return out


def build_qnet(args, dim: int):
    """MLP head, or the VQC Q-function copied from the qdqn-cartpole branch."""
    if args.model == "mlp":
        return QNet(dim, args.width)
    from qdqn import VQCQFunction  # imported lazily so MLP runs need no qiskit

    return VQCQFunction(
        n_qubits=dim,                       # one observation dimension per qubit
        n_layers=args.n_layers,
        n_actions=N_ACTIONS,
        reuploading=args.reuploading,
        observables=default_observables(dim, N_ACTIONS),
        backend="torch_sv",
        seed=args.seed,
        output_rescaling=True,
        per_layer_encoding=args.per_layer_encoding,
    )


def build_optimizer(args, net):
    """VQC needs three learning rates; w must climb from 1 to ~tens fast."""
    if args.model == "mlp":
        return torch.optim.Adam(net.parameters(), lr=args.lr)
    circuit_w = [p for n, p in net.named_parameters() if n not in ("lam", "w")]
    return torch.optim.Adam([
        {"params": circuit_w, "lr": args.lr_variational},
        {"params": [net.lam], "lr": args.lr_input_scaling},
        {"params": [net.w], "lr": args.lr_output_scaling},
    ])


class Replay:
    def __init__(self, cap: int, dim: int):
        self.cap = cap
        self.s = np.zeros((cap, dim), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, dim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.n = 0
        self.i = 0

    def add(self, s, a, r, s2, d):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (j + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, size=batch)
        return (
            torch.from_numpy(self.s[idx]),
            torch.from_numpy(self.a[idx]),
            torch.from_numpy(self.r[idx]),
            torch.from_numpy(self.s2[idx]),
            torch.from_numpy(self.d[idx]),
        )


def make_env(sigma, seed=0):
    return QuantumSpinCartPoleEnv(seed=seed, sigma_ou=sigma)


@torch.no_grad()
def evaluate(net, feat: Featurizer, sigma, n_episodes=50, seed0=1000):
    """epsilon=0 rollouts on the baseline's fixed seeds."""
    env = make_env(sigma, seed=0)
    rets, lens, survived = [], [], []
    for k in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + k)
        env._ou.reset(seed=seed0 + k)  # baselines pin the noise the same way
        s = feat.reset(obs)
        total, steps = 0.0, 0
        term = trunc = False
        while not (term or trunc):
            q = net(torch.from_numpy(s).unsqueeze(0))
            a = int(q.argmax(1).item())
            obs, r, term, trunc, _ = env.step(a)
            s = feat.step(obs, a)
            total += r
            steps += 1
        rets.append(total)
        lens.append(steps)
        survived.append(bool(trunc and not term))
    rets = np.array(rets)
    return {
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "return_median": float(np.median(rets)),
        "length_mean": float(np.mean(lens)),
        "survival_rate": float(np.mean(survived)),
        "returns": rets.tolist(),
    }


def train(args):
    sigma = (
        args.sigma_ou if args.sigma_ou is not None else sigma_for_noise_rabi(args.noise_rabi)
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    feat = Featurizer(args.obs, hist=args.hist)
    eval_feat = Featurizer(args.obs, hist=args.hist)
    net = build_qnet(args, feat.dim)
    tgt = build_qnet(args, feat.dim)
    tgt.load_state_dict(net.state_dict())
    opt = build_optimizer(args, net)
    buf = Replay(args.replay, feat.dim)

    env = make_env(sigma, seed=args.seed)
    # Training episodes get their own deterministic OU seed stream, well clear of
    # the 1000..1049 band the evaluation protocol uses, so training is
    # reproducible without ever seeing an evaluation noise realization.
    ep_counter = 0
    train_seed0 = 500_000 + args.seed * 100_000

    obs, _ = env.reset(seed=train_seed0)
    env._ou.reset(seed=train_seed0)
    s = feat.reset(obs)

    ep_ret, ep_len = 0.0, 0
    train_rets: list[float] = []
    curve = []
    best = {"return_mean": -1e9}
    best_state = None
    t0 = time.time()

    tag = args.model if args.model == "mlp" else f"vqc({feat.dim}q x {args.n_layers}L)"
    print(f"[{tag} {args.obs} seed{args.seed}] input dim {feat.dim}, sigma_ou {sigma:.5f} "
          f"(noise/Rabi {args.noise_rabi}), {args.steps} env steps")
    if args.model == "vqc":
        print(f"  observables: {default_observables(feat.dim, N_ACTIONS)}")
        print(f"  batch {args.batch}, train_every {args.train_every}, "
              f"target_every {args.target_every}, lr w/lam/circuit "
              f"{args.lr_output_scaling}/{args.lr_input_scaling}/{args.lr_variational}")

    for step in range(1, args.steps + 1):
        eps = max(args.eps_end, 1.0 + (args.eps_end - 1.0) * step / args.eps_decay)
        if rng.random() < eps:
            a = int(rng.integers(0, N_ACTIONS))
        else:
            with torch.no_grad():
                a = int(net(torch.from_numpy(s).unsqueeze(0)).argmax(1).item())

        obs, r, term, trunc, _ = env.step(a)
        s2 = feat.step(obs, a)
        buf.add(s, a, r, s2, float(term))  # only true termination bootstraps to 0
        s = s2
        ep_ret += r
        ep_len += 1

        if term or trunc:
            train_rets.append(ep_ret)
            ep_counter += 1
            obs, _ = env.reset(seed=train_seed0 + ep_counter)
            env._ou.reset(seed=train_seed0 + ep_counter)
            s = feat.reset(obs)
            ep_ret, ep_len = 0.0, 0

        if buf.n >= args.learn_start and step % args.train_every == 0:
            bs, ba, br, bs2, bd = buf.sample(args.batch, rng)
            with torch.no_grad():
                if args.double:
                    a2 = net(bs2).argmax(1, keepdim=True)
                    q2 = tgt(bs2).gather(1, a2).squeeze(1)
                else:
                    q2 = tgt(bs2).max(1).values
                y = br + args.gamma * (1.0 - bd) * q2
            q = net(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()

        if step % args.target_every == 0:
            tgt.load_state_dict(net.state_dict())

        if step % args.eval_every == 0:
            # Checkpoint selection runs on its own seed band (2000..), disjoint
            # from the 1000.. band used to report. Selecting the best checkpoint
            # on the same seeds you then report inflates the result, and the
            # inflation grows with the number of evaluations: measured +10.7 at
            # 15 evals and +21.8 at 40, which is large enough to invent an
            # improvement that is not there.
            ev = evaluate(net, eval_feat, sigma, n_episodes=args.eval_episodes,
                          seed0=args.select_seed0)
            recent = float(np.mean(train_rets[-20:])) if train_rets else float("nan")
            curve.append({"step": step, "eps": eps, "train_recent20": recent, **{
                k: v for k, v in ev.items() if k != "returns"}})
            print(f"  step {step:>7}  eps {eps:.3f}  train20 {recent:>8.2f}  "
                  f"eval {ev['return_mean']:>8.2f} +- {ev['return_std']:<6.1f} "
                  f"len {ev['length_mean']:>6.1f}  surv {ev['survival_rate']*100:>3.0f}%")
            if ev["return_mean"] > best["return_mean"]:
                best = ev
                best_state = {k: v.clone() for k, v in net.state_dict().items()}

    # final evaluation on the full 50-seed protocol, using the best checkpoint
    if best_state is not None:
        net.load_state_dict(best_state)
    final = evaluate(net, eval_feat, sigma, n_episodes=args.final_episodes)
    elapsed = time.time() - t0

    print(f"  FINAL ({args.final_episodes} greedy eps): {final['return_mean']:.2f} "
          f"+- {final['return_std']:.2f}  median {final['return_median']:.2f}  "
          f"len {final['length_mean']:.1f}  survive {final['survival_rate']*100:.0f}%")
    print(f"  {elapsed/60:.1f} min")

    out = Path(args.out or f"results/dqn_{args.obs}_nr{args.noise_rabi}_seed{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        ckpt = out.with_suffix(".pt")
        torch.save({
            "state_dict": net.state_dict(),
            "obs": args.obs, "model": args.model, "input_dim": feat.dim,
            "hist": args.hist, "width": args.width, "n_layers": args.n_layers,
            "reuploading": args.reuploading, "per_layer_encoding": args.per_layer_encoding,
            "noise_rabi": args.noise_rabi, "sigma_ou": sigma, "seed": args.seed,
            "steps": args.steps, "final": {k: v for k, v in final.items() if k != "returns"},
        }, ckpt)
        print(f"  saved {ckpt}")

    out.write_text(json.dumps({
        "obs": args.obs,
        "seed": args.seed,
        "noise_rabi": args.noise_rabi,
        "sigma_ou": sigma,
        "input_dim": feat.dim,
        "steps": args.steps,
        "curve": curve,
        "final": final,
        "wall_clock_s": elapsed,
        "hparams": {k: v for k, v in vars(args).items() if k != "out"},
    }, indent=2))
    print(f"  wrote {out}")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", choices=OBS_MODES, default="full")
    p.add_argument("--hist", type=int, default=8)
    p.add_argument("--noise-rabi", type=float, default=NOISE_RABI_RATIO)
    p.add_argument("--sigma-ou", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--replay", type=int, default=50_000)
    p.add_argument("--learn-start", type=int, default=1_000)
    p.add_argument("--train-every", type=int, default=1)
    p.add_argument("--target-every", type=int, default=500)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=30_000)
    p.add_argument("--double", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--final-episodes", type=int, default=50)
    # --- quantum Q-function (ported from the qdqn-cartpole branch) -----------
    p.add_argument("--model", choices=["mlp", "vqc"], default="mlp")
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--reuploading", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--per-layer-encoding", action=argparse.BooleanOptionalAction,
                   default=True)
    # three separate rates, per configs/qdqn.yaml: w must climb from 1 to ~tens
    p.add_argument("--lr-variational", type=float, default=1e-3)
    p.add_argument("--lr-input-scaling", type=float, default=1e-3)
    p.add_argument("--lr-output-scaling", type=float, default=1e-1)
    p.add_argument("--save-model", action="store_true",
                   help="write the best checkpoint next to the results JSON, so a "
                        "trained policy can be replayed or animated later")
    p.add_argument("--select-seed0", type=int, default=2000,
                   help="seed band for checkpoint selection; must not overlap the "
                        "reporting band (1000..1000+final_episodes)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.model == "vqc":
        # Trainer settings from configs/qdqn.yaml, applied unless explicitly
        # overridden on the command line. A gradient step through a circuit
        # costs far more than an environment step, so steps_per_update=10 is
        # the single largest wall-clock lever; target_update_every there is 3
        # gradient steps, i.e. every 30 environment steps here.
        for name, val in (("batch", 16), ("train_every", 10), ("target_every", 30)):
            if getattr(args, name) == p.get_default(name):
                setattr(args, name, val)
    train(args)


if __name__ == "__main__":
    main()
