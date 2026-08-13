"""
Greedy under a masked observation: what happens if sx and sy are hidden?

The 6D observation is [sx, sy, sz, dsx, dsy, dsz]. Masking sx/sy (and their
deltas, which would otherwise leak the same information) leaves the agent with
[sz, dsz] only. That does NOT remove reward information -- the env's fidelity is
(1+|sz|)/2, a function of sz alone -- but it removes the *azimuth*. Knowing sz
fixes the polar angle from the north pole; it says nothing about which way the
Bloch vector points in the xy-plane, and that is exactly what you need in order
to choose between +X/-X/+Y/-Y.

Policies compared here:

  greedy_full    one-step lookahead on the true state (the oracle from
                 greedy_baseline.py, reproduced for reference)

  greedy_pf      the honest masked controller. It never sees sx/sy. It carries a
                 particle filter over the hidden azimuth phi, propagates each
                 particle through its own OU noise draw (the agent knows the
                 noise *statistics*, not the realization), reweights particles by
                 how well they predict the observed sz, resamples, and then
                 projects each particle back onto the observed sz -- the
                 observation is exact, so the belief must respect it. The action
                 is the argmax of expected one-step reward over the belief.

  greedy_blind   ablation: same lookahead, but the belief is re-randomized every
                 step instead of being filtered. Isolates how much of greedy_pf's
                 performance comes from the filter actually inferring phi rather
                 than from the lookahead's structure.

  idle / random  references.

Usage:
    python scripts/greedy_masked.py --episodes 50 --sigma-ou 6.283185307e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from greedy_baseline import (  # noqa: E402
    IDLE,
    GreedyPolicy,
    _bloch,
    _fidelity,
    build_action_propagators,
    sigma_for_noise_rabi,
)
from quantum_spin_cartpole import QuantumSpinCartPoleEnv  # noqa: E402
from quantum_spin_cartpole.constants import ACTION_MAP  # noqa: E402


def mask_obs(obs: np.ndarray) -> np.ndarray:
    """Zero out sx, sy and their time-differences; keep [.., .., sz, .., .., dsz]."""
    out = np.asarray(obs, dtype=np.float32).copy()
    out[[0, 1, 3, 4]] = 0.0
    return out


# ----------------------------------------------------------------------------
# Bloch-vector rotation (Rodrigues), vectorized over particles
# ----------------------------------------------------------------------------
def rotate(r: np.ndarray, f: np.ndarray, dt: float) -> np.ndarray:
    """
    Rotate Bloch vectors r (..,3) under field f (..,3) for time dt.

    H = (1/2) f.sigma  =>  Bloch vector precesses about f by angle |f|*dt.
    """
    r = np.atleast_2d(r)
    f = np.atleast_2d(f)
    norm = np.linalg.norm(f, axis=-1, keepdims=True)
    alpha = norm * dt
    n = np.divide(f, norm, out=np.zeros_like(f), where=norm > 0)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cross = np.cross(n, r)
    dot = np.sum(n * r, axis=-1, keepdims=True)
    return r * ca + cross * sa + n * dot * (1.0 - ca)


def _project_to_sz(r: np.ndarray, sz_obs: float) -> np.ndarray:
    """Rescale the xy part so each particle sits exactly on the observed sz."""
    out = r.copy()
    target_xy = np.sqrt(max(0.0, 1.0 - sz_obs * sz_obs))
    xy_norm = np.linalg.norm(out[:, :2], axis=1, keepdims=True)
    scale = np.divide(target_xy, xy_norm, out=np.zeros_like(xy_norm), where=xy_norm > 1e-12)
    # particles whose xy collapsed get a fresh random azimuth
    degenerate = (xy_norm <= 1e-12).ravel()
    out[:, :2] *= scale
    if degenerate.any():
        phi = np.random.uniform(0, 2 * np.pi, degenerate.sum())
        out[degenerate, 0] = target_xy * np.cos(phi)
        out[degenerate, 1] = target_xy * np.sin(phi)
    out[:, 2] = sz_obs
    return out


class MaskedGreedyPolicy:
    """One-step-lookahead greedy over a particle-filter belief on the azimuth."""

    def __init__(self, env, n_particles=256, rng=None, filtered=True, obs_tau=0.02,
                 probe=False, probe_thresh=0.6, probe_sz_floor=0.5):
        self.dt = env.dt
        self.omega_R = env.omega_R
        self.theta_ou = env.theta_ou
        self.sigma_ou = env.sigma_ou
        self.c_ctrl = env.c_ctrl
        self.r_penalty = env.r_penalty
        self.terminate_on_violation = env.terminate_on_violation
        self.n = n_particles
        self.rng = rng or np.random.default_rng(0)
        self.filtered = filtered
        self.obs_tau = obs_tau
        self.probe = probe
        self.probe_thresh = probe_thresh
        self.probe_sz_floor = probe_sz_floor
        self._probe_cycle = 0

        # noise-free per-action drive fields
        self.fields = np.array(
            [[self.omega_R * ux, self.omega_R * uy, 0.0] for ux, uy in ACTION_MAP]
        )

    def reset(self, sz_obs: float):
        phi = self.rng.uniform(0, 2 * np.pi, self.n)
        xy = np.sqrt(max(0.0, 1.0 - sz_obs * sz_obs))
        self.r = np.stack([xy * np.cos(phi), xy * np.sin(phi), np.full(self.n, sz_obs)], axis=1)
        # each particle carries its own OU noise state, drawn from stationary dist
        std = self.sigma_ou / np.sqrt(2.0 * self.theta_ou)
        self.delta = self.rng.normal(0.0, std, size=(self.n, 3))

    def concentration(self) -> float:
        """Circular resultant length of the belief over phi: 0 = uniform, 1 = certain."""
        phi = np.arctan2(self.r[:, 1], self.r[:, 0])
        return float(np.abs(np.exp(1j * phi).mean()))

    def act(self) -> int:
        # Non-myopic bit: a probe has negative one-step value, so pure greedy
        # never takes one and the belief never sharpens. Force a probe while the
        # belief is diffuse -- but only from a safe sz, so probing cannot itself
        # end the episode.
        if self.probe and self.concentration() < self.probe_thresh:
            if self.r[:, 2].mean() > self.probe_sz_floor:
                self._probe_cycle += 1
                return [0, 2][self._probe_cycle % 2]  # alternate +X / +Y: both axes

        best_a, best_v = IDLE, -np.inf
        for a in range(5):
            f = np.broadcast_to(self.fields[a], (self.n, 3))
            nxt = rotate(self.r, f, self.dt)
            sz = nxt[:, 2]
            rew = (1.0 + np.abs(sz)) / 2.0  # env fidelity == (1+|sz|)/2
            if a != IDLE:
                rew = rew - self.c_ctrl
            if self.terminate_on_violation:
                rew = rew + self.r_penalty * (sz < 0.0)
            v = float(rew.mean())
            if v > best_v:
                best_a, best_v = a, v
        return best_a

    def update(self, action: int, sz_obs: float):
        """Propagate the belief through the executed action, then condition on sz."""
        if not self.filtered:
            # ablation: throw the belief away and re-randomize on the new sz
            self.reset(sz_obs)
            return

        # 1. advance each particle's private OU noise (same recursion as the env)
        eta = self.rng.normal(0.0, 1.0, size=(self.n, 3))
        self.delta += -self.theta_ou * self.delta * self.dt + self.sigma_ou * np.sqrt(self.dt) * eta

        # 2. rotate under drive + that particle's noise
        f = self.fields[action][None, :] + self.delta
        pred = rotate(self.r, f, self.dt)

        # 3. reweight by agreement with the observed sz
        resid = pred[:, 2] - sz_obs
        logw = -0.5 * (resid / self.obs_tau) ** 2
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            self.reset(sz_obs)
            return
        w /= s

        # 4. systematic resampling
        pos = (self.rng.uniform() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(w), pos)
        idx = np.clip(idx, 0, self.n - 1)
        self.r = pred[idx]
        self.delta = self.delta[idx]

        # 5. the sz observation is exact -- put every particle on it
        self.r = _project_to_sz(self.r, sz_obs)


def run_episode(env, policy_name, full_greedy, masked, rng, seed, keep_trace=False):
    obs, _ = env.reset(seed=seed)
    env._ou.reset(seed=seed)
    obs = mask_obs(obs)  # the agent only ever sees the masked observation

    if masked is not None:
        masked.reset(float(env._prev_sz))

    total, steps, actions = 0.0, 0, []
    trace = []
    terminated = truncated = False
    sz = float(env._prev_sz)
    phi_err = []

    while not (terminated or truncated):
        if policy_name == "greedy_full":
            action = full_greedy.act(env._state.full().ravel())
        elif policy_name in ("greedy_pf", "greedy_blind", "greedy_pf_probe"):
            action = masked.act()
        elif policy_name == "random":
            action = int(rng.integers(0, 5))
        else:
            action = IDLE

        # measure how well the belief tracks the true azimuth, before the update
        if masked is not None and policy_name in ("greedy_pf", "greedy_pf_probe"):
            tsx, tsy, _ = _bloch(env._state.full().ravel())
            if tsx * tsx + tsy * tsy > 1e-8:
                true_phi = np.arctan2(tsy, tsx)
                bel_phi = np.arctan2(masked.r[:, 1].mean(), masked.r[:, 0].mean())
                d = np.abs(np.angle(np.exp(1j * (bel_phi - true_phi))))
                phi_err.append(float(d))

        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        actions.append(action)
        sz = info["sz"]
        if masked is not None:
            masked.update(action, sz)
        if keep_trace:
            trace.append([info["sx"], info["sy"], info["sz"], action, reward])

    counts = np.bincount(actions, minlength=5).tolist()
    out = {
        "seed": seed,
        "return": total,
        "steps": steps,
        "survived": bool(truncated and not terminated),
        "final_sz": sz,
        "action_counts": counts,
        "idle_frac": counts[IDLE] / max(1, steps),
    }
    if phi_err:
        out["phi_err_mean"] = float(np.mean(phi_err))
    if keep_trace:
        out["trace"] = trace
    return out


def summarize(name, eps):
    ret = np.array([e["return"] for e in eps])
    length = np.array([e["steps"] for e in eps])
    s = {
        "policy": name,
        "episodes": len(eps),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
        "return_median": float(np.median(ret)),
        "length_mean": float(length.mean()),
        "survival_rate": float(np.mean([e["survived"] for e in eps])),
        "idle_frac": float(np.mean([e["idle_frac"] for e in eps])),
    }
    errs = [e["phi_err_mean"] for e in eps if "phi_err_mean" in e]
    if errs:
        s["phi_err_mean_rad"] = float(np.mean(errs))
    return s


def verify_rotation(env):
    """Rodrigues rotation must agree with the 2x2 propagator used by greedy_full."""
    props = build_action_propagators(env.omega_R, env.dt)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        r0 = np.array(_bloch(v))
        for a, (ux, uy) in enumerate(ACTION_MAP):
            ref = np.array(_bloch(props[a] @ v))
            mine = rotate(r0[None, :], np.array([[env.omega_R * ux, env.omega_R * uy, 0.0]]), env.dt)[0]
            worst = max(worst, float(np.abs(ref - mine).max()))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sigma-ou", type=float, default=None)
    g.add_argument("--noise-rabi", type=float, default=None,
                   help="set SIGMA_OU so that (noise std)/Omega_R equals this ratio")
    p.add_argument("--particles", type=int, default=256)
    p.add_argument(
        "--policies",
        default="greedy_full,greedy_pf,greedy_pf_probe,greedy_blind,idle,random",
    )
    p.add_argument("--tag", default="masked")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    env_kwargs = {}
    if args.sigma_ou is not None:
        env_kwargs["sigma_ou"] = args.sigma_ou
    elif args.noise_rabi is not None:
        env_kwargs["sigma_ou"] = sigma_for_noise_rabi(args.noise_rabi)

    probe = QuantumSpinCartPoleEnv(seed=0, **env_kwargs)
    noise_std = probe.sigma_ou / np.sqrt(2.0 * probe.theta_ou)
    print(f"=== masked-observation greedy [{args.tag}] ===")
    print("observation given to the agent: [0, 0, sz, 0, 0, dsz]   (sx, sy, dsx, dsy masked)")
    print(f"Omega_R = {probe.omega_R:.6f} rad/ns   sigma_ou = {probe.sigma_ou:.6f}")
    print(f"noise/Rabi = {noise_std/probe.omega_R:.2f}x   particles = {args.particles}")
    err = verify_rotation(probe)
    print(f"Rodrigues vs 2x2 propagator: max component error {err:.2e}")
    print()

    results, all_eps = {}, {}
    t0 = time.time()
    for name in args.policies.split(","):
        name = name.strip()
        env = QuantumSpinCartPoleEnv(seed=0, **env_kwargs)
        full_greedy = GreedyPolicy(env)
        masked = None
        if name in ("greedy_pf", "greedy_blind", "greedy_pf_probe"):
            masked = MaskedGreedyPolicy(
                env,
                n_particles=args.particles,
                rng=np.random.default_rng(7),
                filtered=(name != "greedy_blind"),
                probe=(name == "greedy_pf_probe"),
            )
        rng = np.random.default_rng(12345)
        eps = [
            run_episode(env, name, full_greedy, masked, rng, seed=1000 + k)
            for k in range(args.episodes)
        ]
        all_eps[name] = eps
        results[name] = summarize(name, eps)

    elapsed = time.time() - t0

    hdr = f"{'policy':<14} {'return (mean+-sd)':>22} {'median':>9} {'len':>7} {'survive':>8} {'idle%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in results.items():
        print(
            f"{name:<14} {s['return_mean']:>13.2f} +- {s['return_std']:<6.2f} "
            f"{s['return_median']:>8.2f} {s['length_mean']:>7.1f} "
            f"{s['survival_rate']*100:>7.0f}% {s['idle_frac']*100:>6.1f}%"
        )
        if "phi_err_mean_rad" in s:
            print(f"{'':<14} belief azimuth error: {s['phi_err_mean_rad']:.3f} rad "
                  f"({np.degrees(s['phi_err_mean_rad']):.1f} deg; 1.571 = uninformative)")
    print(f"\n{args.episodes} episodes/policy, {elapsed:.1f}s")

    out = Path(args.out or f"results/greedy_masked_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag,
        "episodes": args.episodes,
        "particles": args.particles,
        "config": {
            "omega_R": probe.omega_R,
            "sigma_ou": probe.sigma_ou,
            "noise_std_over_rabi": float(noise_std / probe.omega_R),
        },
        "summary": results,
        "per_episode": all_eps,
        "wall_clock_s": elapsed,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
