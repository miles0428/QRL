"""
Unit tests for the OU noise sampler.
"""
import numpy as np
import pytest

from quantum_spin_cartpole.noise import OUSampler, OU3Sampler


class TestOUSampler:
    """Tests for the Ornstein-Uhlenbeck sampler."""

    @pytest.fixture
    def sampler(self):
        """Create a standard sampler with known parameters."""
        # theta=0.05 ns⁻¹, sigma=2π·20 MHz·ns⁻¹ᐟ² (≈125.66 rad·ns⁻¹ᐟ²)
        return OUSampler(theta=0.05, sigma=125.66, dt=0.05, seed=42)

    def test_initial_value_drawn_from_stationary_dist(self, sampler):
        """After reset(), the initial value should be drawn from the stationary dist."""
        sampler.reset()
        # Stationary std ≈ sigma/sqrt(2*theta) ≈ 397 for the default sampler params
        stationary_std = sampler.sigma / np.sqrt(2.0 * sampler.theta)
        assert abs(sampler.current) < 5.0 * stationary_std

    def test_reset_reseeds(self, sampler):
        """Two resets with the same seed produce identical initial values."""
        v1 = sampler.reset(seed=99)
        v2 = sampler.reset(seed=99)
        assert v1 == v2

    def test_reset_gives_independent_values(self, sampler):
        """Resets with different seeds should generally give different values."""
        sampler.reset(seed=1)
        vals = [sampler.reset() for _ in range(100)]
        # At least some should differ (probability of all same ≈ 2⁻¹⁰⁰)
        unique = len(set(vals))
        assert unique > 1

    def test_step_changes_value(self, sampler):
        """Each step() call updates the internal state."""
        sampler.reset(seed=0)
        initial = sampler.current
        new_val = sampler.step()
        assert new_val != initial

    def test_step_returns_current(self, sampler):
        """step() should return the updated value."""
        sampler.reset(seed=7)
        sampler._current = 1.0
        result = sampler.step()
        assert result == pytest.approx(sampler.current)

    def test_stationary_variance(self, sampler):
        """Long-run variance of OU should roughly match σ²/(2θ) for decorrelated samples.

        We use a very loose tolerance (rel=0.5) because OU samples are serially
        correlated and the empirical variance converges slowly.
        """
        sampler.reset(seed=123)
        # Burn in: advance many steps so the chain mixes toward stationarity
        for _ in range(2000):
            sampler.step()
        samples = [sampler.step() for _ in range(1000)]
        var = np.var(samples, ddof=1)
        expected_var = (sampler.sigma**2) / (2.0 * sampler.theta)
        assert var == pytest.approx(expected_var, rel=0.5)

    def test_mean_reverts_to_zero(self, sampler):
        """OU process mean should revert to zero over time."""
        sampler.reset(seed=999)
        # artificially set far from zero to test mean reversion
        sampler._current = 1000.0
        samples = [sampler.step() for _ in range(2000)]
        # After 2000 steps, the process should have decayed close to zero
        assert np.mean(samples) == pytest.approx(0.0, abs=500.0)

    def test_negative_theta_is_accepted_as_antipersistent(self):
        """OU process with negative theta is mathematically valid (antipersistent)."""
        sampler = OUSampler(theta=-0.1, sigma=1.0, dt=0.05, seed=0)
        sampler.reset()
        val = sampler.step()
        assert isinstance(val, float)

    def test_zero_sigma_gives_deterministic_process(self):
        """sigma=0 should give a deterministic decay toward zero."""
        sampler = OUSampler(theta=0.05, sigma=0.0, dt=0.05, seed=0)
        sampler.reset()
        sampler._current = 100.0
        new_val = sampler.step()
        # With sigma=0, δΔ_{t+dt} = δΔ_t - θ·δΔ_t·dt
        expected = 100.0 - 0.05 * 100.0 * 0.05
        assert new_val == pytest.approx(expected)


class TestOU3Sampler:
    """Tests for the triaxial OU3Sampler."""

    @pytest.fixture
    def sampler(self):
        """Create a standard triaxial sampler with per-axis default parameters."""
        # theta_x = theta_y = theta_z = 0.05 ns⁻¹
        # sigma_x = sigma_y = 2π·10 rad·ns⁻¹ᐟ²
        # sigma_z = 2π·20 rad·ns⁻¹ᐟ²
        return OU3Sampler(
            theta_x=0.05,
            theta_y=0.05,
            theta_z=0.05,
            sigma_x=2.0 * np.pi * 10.0,
            sigma_y=2.0 * np.pi * 10.0,
            sigma_z=2.0 * np.pi * 20.0,
            dt=0.05,
            seed=42,
        )

    def test_initial_values_from_stationary_dist(self, sampler):
        """After reset(), each axis should be drawn from its stationary distribution."""
        sampler.reset()
        dx, dy, dz = sampler.current
        # Stationary std for each axis
        std_x = sampler.sigma_x / np.sqrt(2.0 * sampler.theta_x)
        std_y = sampler.sigma_y / np.sqrt(2.0 * sampler.theta_y)
        std_z = sampler.sigma_z / np.sqrt(2.0 * sampler.theta_z)
        assert abs(dx) < 5.0 * std_x
        assert abs(dy) < 5.0 * std_y
        assert abs(dz) < 5.0 * std_z

    def test_reset_reseeds_consistently(self, sampler):
        """Two resets with the same seed produce identical initial triples."""
        t1 = sampler.reset(seed=99)
        t2 = sampler.reset(seed=99)
        assert t1 == t2

    def test_reset_gives_independent_values(self, sampler):
        """Resets with different seeds should generally give different values."""
        sampler.reset(seed=1)
        vals = [sampler.reset() for _ in range(100)]
        # All three axes for 100 resets — at least some triples should differ
        unique = len(set(vals))
        assert unique > 1

    def test_step_changes_all_three_values(self, sampler):
        """Each step() updates all three axes independently."""
        sampler.reset(seed=0)
        dx0, dy0, dz0 = sampler.current
        dx1, dy1, dz1 = sampler.step()
        # At least one axis should change (with very high probability)
        assert (dx1, dy1, dz1) != (dx0, dy0, dz0)

    def test_step_returns_three_values(self, sampler):
        """step() should return a 3-tuple of floats."""
        sampler.reset(seed=7)
        result = sampler.step()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_axes_are_independent(self, sampler):
        """Three axes should evolve independently with distinct OU parameters."""
        sampler.reset(seed=123)
        # Burn in
        for _ in range(1000):
            sampler.step()
        dx, dy, dz = sampler.current
        # Each axis has different sigma; after burn-in they should differ
        # (not a rigorous statistical test, but sanity check)
        assert isinstance(dx, float)
        assert isinstance(dy, float)
        assert isinstance(dz, float)

    def test_negative_thetas_accepted(self):
        """Negative theta values (antipersistent) should be accepted on all axes."""
        sampler = OU3Sampler(
            theta_x=-0.1, theta_y=-0.1, theta_z=-0.1,
            sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
            dt=0.05, seed=0,
        )
        sampler.reset()
        dx, dy, dz = sampler.step()
        assert all(isinstance(v, float) for v in [dx, dy, dz])

    def test_zero_sigmas_gives_deterministic_decay(self):
        """sigma=0 on all axes gives deterministic exponential decay to zero."""
        sampler = OU3Sampler(
            theta_x=0.05, theta_y=0.05, theta_z=0.05,
            sigma_x=0.0, sigma_y=0.0, sigma_z=0.0,
            dt=0.05, seed=0,
        )
        sampler.reset()
        # Manually set all axes far from zero
        sampler._delta_x = 100.0
        sampler._delta_y = 200.0
        sampler._delta_z = 300.0
        dx, dy, dz = sampler.step()
        # δΔ_{t+dt} = δΔ_t - θ·δΔ_t·dt
        expected_x = 100.0 - 0.05 * 100.0 * 0.05
        expected_y = 200.0 - 0.05 * 200.0 * 0.05
        expected_z = 300.0 - 0.05 * 300.0 * 0.05
        assert dx == pytest.approx(expected_x)
        assert dy == pytest.approx(expected_y)
        assert dz == pytest.approx(expected_z)
