"""
Greedy with NO physics knowledge: fit the model from data, then look ahead.

The greedy in greedy_baseline.py is handed Omega_R, dt, the Hamiltonian form and
the reward formula. This one is handed none of that. It sees exactly what a
model-free RL agent sees -- (obs, action, reward, next_obs, terminated) tuples --
and fits its own model:

  dynamics  next_bloch ~ M_a @ bloch,  one 3x3 matrix per action, least squares.
            Nothing here assumes a Hamiltonian; it is a plain linear regression.
            (It happens to be well specified, because Bloch-vector evolution
            under a fixed rotation really is linear -- but the fitter is not told
            that, it just finds the least-squares map.)

  reward    r ~ MLP(next_bloch, one-hot action), a small network fit on observed
            scalar rewards. This has to discover on its own that the payoff
            depends on |sz| and that crossing sz<0 costs -5.

It then acts by argmax over actions of the predicted one-step reward, exactly as
the physics-aware greedy does.

Data comes from a random policy with terminate_on_violation=False, so the
exploration phase covers the whole sphere rather than only the states a dying
random policy reaches. Exploration budget is counted and reported.

Usage:
    python scripts/greedy_learned.py --noise-rabi 0.5 --fit-steps 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from greedy_baseline import IDLE, _bloch, build_action_propagators, sigma_for_noise_rabi  # noqa: E402
from quantum_spin_cartpole import QuantumSpinCartPoleEnv  # noqa: E402

N_ACTIONS = 5


# ----------------------------------------------------------------------------
# Data collection
# ----------------------------------------------------------------------------
def collect(sigma, n_steps, seed=0):
    """Random-policy transitions with terminations disabled, for coverage."""
    env = QuantumSpinCartPoleEnv(seed=seed, sigma_ou=sigma, terminate_on_violation=False)
    rng = np.random.default_rng(seed)
    B, A, B2, R = [], [], [], []
    obs, _ = env.reset(seed=seed)
    env._ou.reset(seed=seed)
    b = np.array(_bloch(env._state.full().ravel()))
    ep = 0
    for t in range(n_steps):
        a = int(rng.integers(0, N_ACTIONS))
        _, r, term, trunc, info = env.step(a)
        b2 = np.array([info["sx"], info["sy"], info["sz"]])
        B.append(b); A.append(a); B2.append(b2); R.append(r)
        b = b2
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            env._ou.reset(seed=seed + 10_000 + ep)
            b = np.array(_bloch(env._state.full().ravel()))
    return np.array(B), np.array(A), np.array(B2), np.array(R, dtype=np.float32)


# ----------------------------------------------------------------------------
# Model fitting
# ----------------------------------------------------------------------------
def fit_dynamics(B, A, B2):
    """Per-action least squares  B2 ~ B @ M.T  ->  returns list of 3x3 M_a."""
    Ms = []
    for a in range(N_ACTIONS):
        m = A == a
        X, Y = B[m], B2[m]
        # solve min ||X W - Y||, then M = W.T
        W, *_ = np.linalg.lstsq(X, Y, rcond=None)
        Ms.append(W.T)
    return Ms


class RewardNet(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(3 + N_ACTIONS, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit_reward(B2, A, R, epochs=300, seed=0):
    torch.manual_seed(seed)
    onehot = np.zeros((len(A), N_ACTIONS), dtype=np.float32)
    onehot[np.arange(len(A)), A] = 1.0
    X = torch.from_numpy(np.hstack([B2.astype(np.float32), onehot]))
    Y = torch.from_numpy(R)
    net = RewardNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n = len(Y)
    for e in range(epochs):
        idx = torch.randperm(n)[:4096]
        loss = nn.functional.mse_loss(net(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        full = float(nn.functional.mse_loss(net(X), Y))
    return net, full


class LearnedGreedy:
    def __init__(self, Ms, rnet):
        self.Ms = Ms
        self.rnet = rnet
        self._eye = np.eye(N_ACTIONS, dtype=np.float32)

    @torch.no_grad()
    def act(self, b: np.ndarray) -> int:
        nxt = np.stack([M @ b for M in self.Ms]).astype(np.float32)   # (5,3)
        x = torch.from_numpy(np.hstack([nxt, self._eye]))
        return int(self.rnet(x).argmax().item())


# ----------------------------------------------------------------------------
def evaluate(policy, sigma, n_episodes=50, seed0=1000):
    env = QuantumSpinCartPoleEnv(seed=0, sigma_ou=sigma)
    rets, lens, surv = [], [], []
    for k in range(n_episodes):
        env.reset(seed=seed0 + k)
        env._ou.reset(seed=seed0 + k)
        b = np.array(_bloch(env._state.full().ravel()))
        total, steps = 0.0, 0
        term = trunc = False
        while not (term or trunc):
            a = policy.act(b)
            _, r, term, trunc, info = env.step(a)
            b = np.array([info["sx"], info["sy"], info["sz"]])
            total += r; steps += 1
        rets.append(total); lens.append(steps); surv.append(bool(trunc and not term))
    rets = np.array(rets)
    return {
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std(ddof=1)),
        "return_median": float(np.median(rets)),
        "length_mean": float(np.mean(lens)),
        "survival_rate": float(np.mean(surv)),
        "returns": rets.tolist(),
    }


def true_propagator_bloch(env):
    """The physics-aware greedy's rotations, as 3x3 Bloch matrices, for scoring."""
    props = build_action_propagators(env.omega_R, env.dt)
    Ms = []
    for U in props:
        cols = []
        for e in np.eye(3):
            # find a state with this Bloch vector, push it through U
            sx, sy, sz = e
            # build density-free pure state from Bloch vector
            theta = np.arccos(np.clip(sz, -1, 1)); phi = np.arctan2(sy, sx)
            v = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
            cols.append(np.array(_bloch(U @ v)))
        Ms.append(np.stack(cols, axis=1))
    return Ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise-rabi", type=float, default=0.5)
    p.add_argument("--fit-steps", type=int, default=20000)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    sigma = sigma_for_noise_rabi(args.noise_rabi)
    print(f"=== physics-free greedy, noise/Rabi = {args.noise_rabi} ===")
    print(f"model fit from {args.fit_steps} random-policy transitions "
          f"(no Omega_R, no Hamiltonian, no reward formula)\n")

    ref_env = QuantumSpinCartPoleEnv(seed=0, sigma_ou=sigma)
    true_Ms = true_propagator_bloch(ref_env)

    runs = []
    for s in range(args.seeds):
        t0 = time.time()
        B, A, B2, R = collect(sigma, args.fit_steps, seed=100 + s)
        Ms = fit_dynamics(B, A, B2)
        rnet, rmse = fit_reward(B2, A, R, seed=s)
        err = max(float(np.abs(Ms[a] - true_Ms[a]).max()) for a in range(N_ACTIONS))
        ev = evaluate(LearnedGreedy(Ms, rnet), sigma, n_episodes=args.episodes)
        runs.append({"seed": s, "dyn_max_err": err, "reward_mse": rmse, **ev})
        print(f"  seed{s}: learned-vs-true dynamics max |err| {err:.4f}   "
              f"reward MSE {rmse:.4f}   ->  return {ev['return_mean']:7.2f} "
              f"+- {ev['return_std']:.2f}  surv {ev['survival_rate']*100:.0f}%  "
              f"({time.time()-t0:.0f}s)")

    m = np.array([r["return_mean"] for r in runs])
    print(f"\nlearned greedy: {m.mean():.2f} +- {m.std(ddof=1) if len(m)>1 else 0:.2f} "
          f"over {args.seeds} fits")

    out = Path(args.out or f"results/greedy_learned_nr{args.noise_rabi}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "noise_rabi": args.noise_rabi, "sigma_ou": sigma,
        "fit_steps": args.fit_steps, "runs": runs,
        "mean_over_fits": float(m.mean()),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
