"""
Greedy (one-step lookahead) baseline for QuantumSpinCartPole-v0.

The greedy controller is model-based but *not* clairvoyant: at each step it
knows its own drive Hamiltonian and rolls each of the 5 actions forward one
timestep through the noise-free propagator, then picks the action with the
highest predicted reward. It never sees the OU noise realization that the
environment is about to apply -- that would be cheating.

Predicted reward for action a (mirrors env.step's reward exactly):
    r_hat(a) = fidelity(sx', sy') - c_ctrl * [a != IDLE] + r_penalty * [sz' < 0]

Two reference policies are run alongside it so the number means something:
    random  -- uniform over the 5 actions
    idle    -- always action 4 (no drive); measures how much the noise alone does

Usage:
    python scripts/greedy_baseline.py                 # default constants, 20 episodes
    python scripts/greedy_baseline.py --episodes 50
    python scripts/greedy_baseline.py --sigma-ou 6.283e-3   # see NOTE below

NOTE on SIGMA_OU: constants.py sets SIGMA_OU = 2*pi*1.0 rad/ns^(1/2) with the
comment "noise/Rabi ~ 2%", but OMEGA_R encodes 100 MHz as 2*pi*100.0e-3 (GHz
units, since rad/ns needs GHz). By the same convention 1 MHz is 2*pi*1.0e-3,
so the default sigma is 1000x its documented intent. --sigma-ou lets you run
both and compare.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quantum_spin_cartpole import QuantumSpinCartPoleEnv
from quantum_spin_cartpole.constants import ACTION_MAP, C_CTRL, OMEGA_R, R_PENALTY, THETA_OU


def sigma_for_noise_rabi(ratio: float, omega_R: float = OMEGA_R, theta: float = THETA_OU) -> float:
    """
    sigma_ou that makes (stationary noise std)/Omega_R equal `ratio`.

    stationary std of the OU process is sigma/sqrt(2*theta), so
        sigma = ratio * Omega_R * sqrt(2*theta)
    """
    return ratio * omega_R * np.sqrt(2.0 * theta)

IDLE = 4
ACTION_NAMES = ["+X", "-X", "+Y", "-Y", "IDLE"]

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def build_action_propagators(omega_R: float, dt: float) -> list[np.ndarray]:
    """
    Noise-free one-step propagator for each action.

    H = (1/2)(fx sx + fy sy),  fx = omega_R*ux, fy = omega_R*uy
    U = exp(-i H dt) = cos(t/2) I - i sin(t/2) (n.sigma),  t = |f| dt
    """
    props = []
    for ux, uy in ACTION_MAP:
        fx, fy = omega_R * float(ux), omega_R * float(uy)
        norm = np.hypot(fx, fy)
        if norm == 0.0:
            props.append(_I2.copy())
            continue
        theta = norm * dt
        nx, ny = fx / norm, fy / norm
        props.append(np.cos(theta / 2) * _I2 - 1j * np.sin(theta / 2) * (nx * _SX + ny * _SY))
    return props


def _bloch(vec: np.ndarray) -> tuple[float, float, float]:
    """Pauli expectation values of a 2-component state vector."""
    bra = vec.conj()
    return (
        float((bra @ (_SX @ vec)).real),
        float((bra @ (_SY @ vec)).real),
        float((bra @ (_SZ @ vec)).real),
    )


def _fidelity(sx: float, sy: float) -> float:
    """Same formula env._compute_fidelity uses: (1 + sqrt(1 - sx^2 - sy^2)) / 2."""
    return (1.0 + np.sqrt(max(0.0, 1.0 - sx * sx - sy * sy))) / 2.0


class GreedyPolicy:
    """Argmax over one-step predicted reward under the noise-free model."""

    def __init__(self, env: QuantumSpinCartPoleEnv):
        self.props = build_action_propagators(env.omega_R, env.dt)
        self.c_ctrl = env.c_ctrl
        self.r_penalty = env.r_penalty
        self.terminate_on_violation = env.terminate_on_violation

    def act(self, state_vec: np.ndarray) -> int:
        best_a, best_r = IDLE, -np.inf
        for a, U in enumerate(self.props):
            nxt = U @ state_vec
            sx, sy, sz = _bloch(nxt)
            r = _fidelity(sx, sy)
            if a != IDLE:
                r -= self.c_ctrl
            if self.terminate_on_violation and sz < 0.0:
                r += self.r_penalty
            if r > best_r:
                best_a, best_r = a, r
        return best_a


def run_episode(env, policy_name, greedy, rng, seed, keep_trace=False):
    """One episode. Returns a stats dict (plus a per-step trace if requested)."""
    env.reset(seed=seed)
    # env.reset() reseeds the OU sampler from system entropy, so seeding the
    # episode alone does not make the noise reproducible. Pin it explicitly.
    env._ou.reset(seed=seed)

    total, steps, actions = 0.0, 0, []
    trace = []
    terminated = truncated = False
    sz = float(env._prev_sz)

    while not (terminated or truncated):
        if policy_name == "greedy":
            action = greedy.act(env._state.full().ravel())
        elif policy_name == "random":
            action = int(rng.integers(0, 5))
        else:  # idle
            action = IDLE

        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        actions.append(action)
        sz = info["sz"]
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
    if keep_trace:
        out["trace"] = trace
    return out


def summarize(name, eps):
    ret = np.array([e["return"] for e in eps])
    length = np.array([e["steps"] for e in eps])
    surv = np.mean([e["survived"] for e in eps])
    idle = np.mean([e["idle_frac"] for e in eps])
    return {
        "policy": name,
        "episodes": len(eps),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
        "return_median": float(np.median(ret)),
        "length_mean": float(length.mean()),
        "length_median": float(np.median(length)),
        "survival_rate": float(surv),
        "idle_frac": float(idle),
    }


def verify_propagator(env, tol=1e-9):
    """Confirm the analytic lookahead matches the env's own qutip processor."""
    props = build_action_propagators(env.omega_R, env.dt)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(5):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        v /= np.linalg.norm(v)
        import qutip as qt

        psi = qt.Qobj(v.reshape(2, 1))
        for a, (ux, uy) in enumerate(ACTION_MAP):
            ref = env._processor.run_state(
                init_state=psi, fx=env.omega_R * float(ux), fy=env.omega_R * float(uy), fz=0.0
            )
            ref_v = ref.full().ravel()
            mine = props[a] @ v
            # compare up to global phase
            fid = abs(np.vdot(ref_v, mine)) ** 2
            worst = max(worst, abs(1.0 - fid))
    return worst, worst < tol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sigma-ou", type=float, default=None, help="override SIGMA_OU directly")
    g.add_argument("--noise-rabi", type=float, default=None,
                   help="set SIGMA_OU so that (noise std)/Omega_R equals this ratio")
    p.add_argument("--policies", default="greedy,random,idle")
    p.add_argument("--out", default=None, help="path for the JSON result file")
    p.add_argument("--tag", default="default")
    args = p.parse_args()

    env_kwargs = {}
    if args.sigma_ou is not None:
        env_kwargs["sigma_ou"] = args.sigma_ou
    elif args.noise_rabi is not None:
        env_kwargs["sigma_ou"] = sigma_for_noise_rabi(args.noise_rabi)

    probe = QuantumSpinCartPoleEnv(seed=0, **env_kwargs)
    noise_std = probe.sigma_ou / np.sqrt(2.0 * probe.theta_ou)
    print(f"=== QuantumSpinCartPole greedy baseline [{args.tag}] ===")
    print(f"Omega_R      = {probe.omega_R:.6f} rad/ns   (drive rotation {probe.omega_R*probe.dt:.4f} rad/step)")
    print(f"sigma_ou     = {probe.sigma_ou:.6f}         stationary noise std = {noise_std:.4f} rad/ns")
    print(f"noise / Rabi = {noise_std/probe.omega_R:.2f}x")
    print(f"t_max={probe.t_max}  c_ctrl={probe.c_ctrl}  r_penalty={probe.r_penalty}")

    err, ok = verify_propagator(probe)
    print(f"analytic lookahead vs qutip processor: max infidelity {err:.2e}  ({'MATCH' if ok else 'MISMATCH'})")
    print()

    results, all_eps = {}, {}
    trace = None
    t0 = time.time()

    for name in args.policies.split(","):
        name = name.strip()
        env = QuantumSpinCartPoleEnv(seed=0, **env_kwargs)
        greedy = GreedyPolicy(env)
        rng = np.random.default_rng(12345)
        eps = []
        for k in range(args.episodes):
            keep = name == "greedy" and k == 0
            e = run_episode(env, name, greedy, rng, seed=1000 + k, keep_trace=keep)
            if keep:
                trace = e.pop("trace")
            eps.append(e)
        all_eps[name] = eps
        results[name] = summarize(name, eps)

    elapsed = time.time() - t0

    hdr = f"{'policy':<8} {'return (mean+-sd)':>22} {'median':>9} {'len':>7} {'survive':>8} {'idle%':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in results.items():
        print(
            f"{name:<8} {s['return_mean']:>13.2f} +- {s['return_std']:<6.2f} "
            f"{s['return_median']:>8.2f} {s['length_mean']:>7.1f} "
            f"{s['survival_rate']*100:>7.0f}% {s['idle_frac']*100:>6.1f}%"
        )
    print(f"\n{args.episodes} episodes/policy, {elapsed:.1f}s total")

    out = Path(args.out or f"results/greedy_baseline_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "episodes": args.episodes,
        "config": {
            "omega_R": probe.omega_R,
            "sigma_ou": probe.sigma_ou,
            "theta_ou": probe.theta_ou,
            "noise_std_over_rabi": float(noise_std / probe.omega_R),
            "t_max": probe.t_max,
            "c_ctrl": C_CTRL,
            "r_penalty": R_PENALTY,
        },
        "summary": results,
        "per_episode": all_eps,
        "greedy_trace_ep0": trace,
        "wall_clock_s": elapsed,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
