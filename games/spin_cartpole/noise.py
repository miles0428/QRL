"""
Ornstein-Uhlenbeck (OU) stochastic process sampler for the Quantum Spin-CartPole environment.

The OU process drives the detuning noise δΔ(t) during each episode.
Episode-to-episode randomness is handled by reseeding the internal RNG in reset().
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Union


class OU3Sampler:
    """
    Triaxial Ornstein-Uhlenbeck sampler for the Quantum Spin-CartPole environment.

    Manages three independent OU processes (one per spin axis) for the
    stochastic magnetic field perturbations δΔ_x(t), δΔ_y(t), δΔ_z(t).

    The discrete update rule for each axis i ∈ {x, y, z} is:
        δΔ_{i,t+Δt} = δΔ_{i,t} - θ_i · δΔ_{i,t} · Δt + σ_i · √Δt · η_{i,t}
    where η_{i,t} ~ N(0, 1) are independently sampled.

    On reset(), the RNG is reseeded and each axis is independently drawn from
    its stationary distribution N(0, σ_i² / (2θ_i)).
    """

    def __init__(
        self,
        theta_x: float,
        theta_y: float,
        theta_z: float,
        sigma_x: float,
        sigma_y: float,
        sigma_z: float,
        dt: float,
        seed: Optional[int] = None,
    ):
        """
        Initialize the triaxial OU sampler.

        Args:
            theta_x: OU decay rate for x-axis (θ_x), in ns⁻¹.
            theta_y: OU decay rate for y-axis (θ_y), in ns⁻¹.
            theta_z: OU decay rate for z-axis (θ_z), in ns⁻¹.
            sigma_x: OU diffusion coefficient for x-axis (σ_x), in rad·ns⁻¹ᐟ².
            sigma_y: OU diffusion coefficient for y-axis (σ_y), in rad·ns⁻¹ᐟ².
            sigma_z: OU diffusion coefficient for z-axis (σ_z), in rad·ns⁻¹ᐟ².
            dt: Simulation timestep (Δt) in ns.
            seed: Random seed for reproducibility.
        """
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.dt = dt

        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._delta_x: Optional[float] = None
        self._delta_y: Optional[float] = None
        self._delta_z: Optional[float] = None

    def reset(self, seed: Optional[int] = None) -> tuple[float, float, float]:
        """
        Reseed the RNG and draw initial values from each axis' stationary distribution.

        The stationary distribution for axis i is N(0, σ_i² / (2θ_i)).

        Args:
            seed: Optional new random seed. If None, reseed from system entropy.

        Returns:
            tuple: (δΔ_x0, δΔ_y0, δΔ_z0) — the three initial detuning values.
        """
        if seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self._rng = np.random.Generator(np.random.PCG64())

        stationary_std_x = self.sigma_x / np.sqrt(2.0 * self.theta_x)
        stationary_std_y = self.sigma_y / np.sqrt(2.0 * self.theta_y)
        stationary_std_z = self.sigma_z / np.sqrt(2.0 * self.theta_z)

        self._delta_x = self._rng.normal(0.0, stationary_std_x)
        self._delta_y = self._rng.normal(0.0, stationary_std_y)
        self._delta_z = self._rng.normal(0.0, stationary_std_z)

        return self._delta_x, self._delta_y, self._delta_z

    def step(self) -> tuple[float, float, float]:
        """
        Advance all three OU processes by one timestep (Δt).

        Each axis is updated independently using the Euler-Maruyama scheme:
            δΔ_{i,t+Δt} = δΔ_{i,t} - θ_i·δΔ_{i,t}·Δt + σ_i·√Δt·η_{i,t}

        Returns:
            tuple: (δΔ_x, δΔ_y, δΔ_z) — the three updated detuning values.
        """
        eta_x = self._rng.normal(0.0, 1.0)
        eta_y = self._rng.normal(0.0, 1.0)
        eta_z = self._rng.normal(0.0, 1.0)

        delta_dx = -self.theta_x * self._delta_x * self.dt + self.sigma_x * np.sqrt(self.dt) * eta_x
        delta_dy = -self.theta_y * self._delta_y * self.dt + self.sigma_y * np.sqrt(self.dt) * eta_y
        delta_dz = -self.theta_z * self._delta_z * self.dt + self.sigma_z * np.sqrt(self.dt) * eta_z

        self._delta_x = self._delta_x + delta_dx
        self._delta_y = self._delta_y + delta_dy
        self._delta_z = self._delta_z + delta_dz

        return self._delta_x, self._delta_y, self._delta_z

    @property
    def current(self) -> tuple[float, float, float]:
        """Return the current detuning values (delta_x, delta_y, delta_z)."""
        return self._delta_x, self._delta_y, self._delta_z


class OUSampler:
    """
    Sampler for the Ornstein-Uhlenbeck process used to model dephasing noise.

    The discrete update rule is:
        δΔ_{t+Δt} = δΔ_t - θ · δΔ_t · Δt + σ · √Δt · η_t
    where η_t ~ N(0, 1) is an independent standard normal sample.

    On reset(), the RNG is reseeded and the initial value is drawn from the
    stationary Gaussian distribution N(0, σ² / (2θ)).
    """

    def __init__(
        self,
        theta: float,
        sigma: float,
        dt: float,
        seed: Optional[int] = None,
    ):
        """
        Initialize the OU sampler.

        Args:
            theta: OU decay rate (θ_OU), in ns⁻¹.
            sigma: OU diffusion coefficient (σ_OU), in rad·ns⁻¹ᐟ².
            dt: Simulation timestep (Δt) in ns.
            seed: Random seed for reproducibility.
        """
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._current: Optional[float] = None

    def reset(self, seed: Optional[int] = None) -> float:
        """
        Reseed the RNG and draw the initial detuning from the stationary distribution.

        The stationary distribution is N(0, σ² / (2θ)).

        Args:
            seed: Optional new random seed. If None, reseed from system entropy.

        Returns:
            float: The initial detuning value δΔ₀.
        """
        if seed is not None:
            self._rng = np.random.Generator(np.random.PCG64(seed))
        else:
            self._rng = np.random.Generator(np.random.PCG64())

        # Stationary variance = sigma^2 / (2 * theta)
        stationary_std = self.sigma / np.sqrt(2.0 * self.theta)
        self._current = self._rng.normal(0.0, stationary_std)
        return self._current

    def step(self) -> float:
        """
        Advance the OU process by one timestep (Δt).

        Uses the Euler-Maruyama update:
            δΔ_{t+Δt} = δΔ_t - θ·δΔ_t·Δt + σ·√Δt·η_t

        Returns:
            float: The updated detuning value.
        """
        eta = self._rng.normal(0.0, 1.0)
        delta = (
            -self.theta * self._current * self.dt
            + self.sigma * np.sqrt(self.dt) * eta
        )
        self._current = self._current + delta
        return self._current

    @property
    def current(self) -> float:
        """Return the current detuning value."""
        return self._current
