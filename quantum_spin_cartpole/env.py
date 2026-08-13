"""
Quantum Spin-CartPole V2 Gymnasium environment.

Observation: [⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩, Δ⟨σ_x⟩, Δ⟨σ_y⟩, Δ⟨σ_z⟩] — 6D vector
Action: Discrete(5) — +X, -X, +Y, -Y, IDLE
Dynamics: ModelProcessor-based pure-state evolution with triaxial OU noise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import qutip as qt
import gymnasium as gym
from gymnasium import spaces

from .constants import (
    DT,
    OMEGA_R,
    THETA_OU,
    SIGMA_OU,
    T_MAX,
    C_CTRL,
    R_PENALTY,
    ACTION_MAP,
    OBS_LOW,
    OBS_HIGH,
    HBAR,
)
from .noise import OU3Sampler
from .processor import SpinProcessor
from .hamiltonian import create_wavefunction, SIGMA_X, SIGMA_Y, SIGMA_Z


class QuantumSpinCartPoleEnv(gym.Env):
    """
    Gymnasium environment for SU(2) spin-1/2 control (V2).

    Dynamics under the V2 Hamiltonian:
        H(t) = H_x(t) + H_y(t) + H_z(t)

    Where:
        H_k(t) = (ħ/2) · σ_k · f_k(t)

    Channel amplitudes:
        f_x = Ω_R · u_x(a_t) + δΔ_x(t)
        f_y = Ω_R · u_y(a_t) + δΔ_y(t)
        f_z = δΔ_z(t)

    Observation (6D):
        o_t = [⟨σ_x⟩_t, ⟨σ_y⟩_t, ⟨σ_z⟩_t, Δ⟨σ_x⟩_t, Δ⟨σ_y⟩_t, Δ⟨σ_z⟩_t]

    Episode terminates when ⟨σ_z⟩_t < 0 (south hemisphere violation).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = DT,
        omega_R: float = OMEGA_R,
        theta_ou: float = THETA_OU,
        sigma_ou: float = SIGMA_OU,
        t_max: int = T_MAX,
        c_ctrl: float = C_CTRL,
        r_penalty: float = R_PENALTY,
        seed: Optional[int] = None,
        terminate_on_violation: bool = True,
    ):
        """
        Initialize the QuantumSpinCartPoleEnv V2.

        Args:
            dt: Simulation timestep (ns).
            omega_R: Rabi angular frequency (rad/ns).
            theta_ou: OU noise decay rate (ns⁻¹), unified for all axes.
            sigma_ou: OU noise diffusion term (rad·ns⁻¹/²), unified for all axes.
            t_max: Maximum steps per episode.
            c_ctrl: Control penalty coefficient.
            r_penalty: Failure penalty (applied when sz < 0).
            seed: Random seed for reproducibility.
            terminate_on_violation: If True, episode terminates when sz < 0.
                If False, sz < 0 is allowed and the episode continues until t_max.
        """
        super().__init__()

        self.dt = dt
        self.omega_R = omega_R
        self.theta_ou = theta_ou
        self.sigma_ou = sigma_ou
        self.t_max = t_max
        self.c_ctrl = c_ctrl
        self.r_penalty = r_penalty
        self.terminate_on_violation = terminate_on_violation

        # QuTiP ModelProcessor
        self._processor = SpinProcessor(Omega_R=self.omega_R, dt=self.dt)

        # Triaxial OU noise sampler (unified theta/sigma)
        self._ou = OU3Sampler(
            theta_x=self.theta_ou,
            theta_y=self.theta_ou,
            theta_z=self.theta_ou,
            sigma_x=self.sigma_ou,
            sigma_y=self.sigma_ou,
            sigma_z=self.sigma_ou,
            dt=self.dt,
            seed=seed,
        )

        # Internal quantum state
        self._state: Optional[qt.Qobj] = None
        self._step_count: int = 0

        # History cache for time-difference observations
        self._prev_sx: float = 0.0
        self._prev_sy: float = 0.0
        self._prev_sz: float = 0.0

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            shape=(6,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(5)

        # RNG for reset
        self._np_rng = np.random.Generator(
            np.random.PCG64(seed) if seed is not None else np.random.PCG64()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_fidelity(self, sx: float, sy: float) -> float:
        """
        Compute fidelity toward |0⟩ (spin-up) from Bloch x/y components.

        F = |⟨0|ψ⟩|² = (1 + ⟨σ_z⟩) / 2
          = (1 + sqrt(1 - ⟨σ_x⟩² - ⟨σ_y⟩²)) / 2

        Args:
            sx: ⟨σ_x⟩ (expectation value).
            sy: ⟨σ_y⟩ (expectation value).

        Returns:
            float: Fidelity in [0, 1].
        """
        sz_sq = 1.0 - sx * sx - sy * sy
        sz = np.sqrt(max(0.0, sz_sq))
        return (1.0 + sz) / 2.0

    def _expectation(self, psi: qt.Qobj) -> tuple[float, float, float]:
        """
        Compute Pauli expectation values for a normalized state.

        Args:
            psi: Normalized state ket.

        Returns:
            tuple: (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)
        """
        sx = float((psi.dag() * SIGMA_X * psi).real)
        sy = float((psi.dag() * SIGMA_Y * psi).real)
        sz = float((psi.dag() * SIGMA_Z * psi).real)
        return sx, sy, sz

    def _build_obs(
        self,
        sx: float,
        sy: float,
        sz: float,
        prev_sx: float,
        prev_sy: float,
        prev_sz: float,
    ) -> np.ndarray:
        """
        Build the 6D observation vector.

        Args:
            sx: Current ⟨σ_x⟩.
            sy: Current ⟨σ_y⟩.
            sz: Current ⟨σ_z⟩.
            prev_sx: Previous ⟨σ_x⟩.
            prev_sy: Previous ⟨σ_y⟩.
            prev_sz: Previous ⟨σ_z⟩.

        Returns:
            np.ndarray: 6D observation [sx, sy, sz, Δsx, Δsy, Δsz].
        """
        return np.array(
            [sx, sy, sz, sx - prev_sx, sy - prev_sy, sz - prev_sz],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options=None):
        """
        Reset the environment to a fresh episode.

        Samples an initial state uniformly from the upper hemisphere
        (⟨σ_z⟩ ≥ 0) on the Bloch sphere.

        Args:
            seed: Optional random seed override.
            options: Optional dict (unused, for gymnasium API compatibility).

        Returns:
            tuple: (observation, info dict)
        """
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._np_rng = np.random.Generator(np.random.PCG64(seed))

        # Sample uniform upper-hemisphere Bloch state
        u1 = self._np_rng.uniform(0.0, 1.0)
        u2 = self._np_rng.uniform(0.0, 1.0)
        theta = np.arccos(u1)  # ∈ [0, π/2] — upper hemisphere
        phi = 2.0 * np.pi * u2  # ∈ [0, 2π)

        self._state = create_wavefunction(theta, phi)
        self._step_count = 0

        # Reset triaxial OU noise
        self._ou.reset()

        # Initial expectation values
        sx0, sy0, sz0 = self._expectation(self._state)

        # Initialize history cache (Δ components = 0 at step 0)
        self._prev_sx = sx0
        self._prev_sy = sy0
        self._prev_sz = sz0

        obs = self._build_obs(sx0, sy0, sz0, sx0, sy0, sz0)  # Δsx=0, Δsy=0, Δsz=0
        info = {
            "episode_step": 0,
            "sx": sx0,
            "sy": sy0,
            "sz": sz0,
        }
        return obs, info

    def step(self, action: int):
        """
        Advance the simulation by one timestep.

        Args:
            action: Integer in [0, 4]:
                0 (+X), 1 (-X), 2 (+Y), 3 (-Y), 4 (IDLE)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 1. Sample triaxial OU noise
        delta_x, delta_y, delta_z = self._ou.step()

        # 2. Determine (u_x, u_y) from action
        ux, uy = ACTION_MAP[action]

        # 3. Compute channel amplitudes
        fx = self.omega_R * ux + delta_x
        fy = self.omega_R * uy + delta_y
        fz = delta_z

        # 4. Evolve state via ModelProcessor
        self._state = self._processor.run_state(
            init_state=self._state,
            fx=fx,
            fy=fy,
            fz=fz,
        )

        # 5. Compute current expectation values
        sx, sy, sz = self._expectation(self._state)

        # 6. Build observation with time differences
        obs = self._build_obs(sx, sy, sz, self._prev_sx, self._prev_sy, self._prev_sz)

        # 7. Update history cache
        self._prev_sx = sx
        self._prev_sy = sy
        self._prev_sz = sz

        self._step_count += 1

        # 8. Termination checks
        # terminated is triggered by south-hemisphere violation only when
        # terminate_on_violation is enabled (default True for backward compat).
        # When disabled, sz < 0 does NOT terminate — the episode runs to t_max.
        terminated = bool(self.terminate_on_violation and sz < 0.0)
        truncated = bool(self._step_count >= self.t_max)

        # 9. Reward
        reward = self._compute_fidelity(sx, sy)
        if action != 4:  # penalize non-idle actions
            reward -= self.c_ctrl
        if terminated:
            reward += self.r_penalty

        info = {
            "episode_step": self._step_count,
            "sx": sx,
            "sy": sy,
            "sz": sz,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        """Gymnasium render — not implemented (physics-only environment)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
