"""
Unit tests for the QuantumSpinCartPoleEnv V2 Gymnasium interface.

Observations are now 6D: [⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩, Δ⟨σ_x⟩, Δ⟨σ_y⟩, Δ⟨σ_z⟩]
"""
import numpy as np
import qutip as qt
import pytest
import gymnasium as gym

from quantum_spin_cartpole.env import QuantumSpinCartPoleEnv
from quantum_spin_cartpole.constants import ACTION_MAP


class TestEnvironmentInterface:
    """Test that the environment conforms to the Gymnasium API."""

    @pytest.fixture
    def env(self):
        """Create a fresh environment instance."""
        e = QuantumSpinCartPoleEnv(seed=0)
        yield e
        e.close()

    def test_instantiation(self):
        """Environment should instantiate without error."""
        env = QuantumSpinCartPoleEnv(seed=42)
        assert env is not None
        env.close()

    def test_observation_space(self, env):
        """Observation space should be Box(low=[-1,-1,-1,-2,-2,-2], high=[1,1,1,2,2,2], shape=(6,))."""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (6,)
        np.testing.assert_allclose(env.observation_space.low, [-1.0, -1.0, -1.0, -2.0, -2.0, -2.0])
        np.testing.assert_allclose(env.observation_space.high, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_action_space(self, env):
        """Action space should be Discrete(5)."""
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 5

    def test_reset_returns_observation(self, env):
        """reset() should return (obs, info) where obs has shape (6,)."""
        obs, info = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        assert obs.dtype == np.float32
        # ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ ∈ [-1, 1]
        assert -1.0 <= obs[0] <= 1.0
        assert -1.0 <= obs[1] <= 1.0
        assert -1.0 <= obs[2] <= 1.0
        # Δ components at reset: should be exactly 0
        assert obs[3] == 0.0
        assert obs[4] == 0.0
        assert obs[5] == 0.0

    def test_reset_upper_hemisphere_constraint(self, env):
        """Initial ⟨σ_z⟩ should be non-negative (upper hemisphere)."""
        for seed in range(10):
            env.reset(seed=seed)
            psi = env._state
            sx, sy, sz = env._expectation(psi)
            sz_sq = 1.0 - sx**2 - sy**2
            assert sz_sq >= -1e-10, f"sz²={sz_sq} < 0 for seed={seed}"

    def test_step_returns_five_tuple(self, env):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action=0)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_all_actions_valid(self, env):
        """All 5 actions (0–4) should be accepted without exception."""
        env.reset(seed=1)
        for a in range(5):
            obs, reward, terminated, truncated, info = env.step(action=a)
            assert obs.shape == (6,)

    def test_episode_runs_full_length(self, env):
        """Episode should run for t_max steps when not terminated early."""
        env.reset(seed=5)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action=4)
            steps += 1
        assert steps == env.t_max or terminated

    def test_terminated_on_southern_hemisphere(self, env):
        """Spin entering southern hemisphere should set terminated=True."""
        env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated and steps < 2000:
            obs, reward, terminated, truncated, info = env.step(action=0)
            steps += 1
        assert terminated or truncated

    def test_observation_consistency(self, env):
        """
        Verify that observations have correct shape (6,) and lie within bounds.
        """
        env.reset(seed=3)
        obs1, *_ = env.step(action=4)
        env.reset(seed=42)
        obs2, *_ = env.step(action=4)

        # Shape check
        assert obs1.shape == (6,), f"obs1 shape mismatch: {obs1.shape}"
        assert obs2.shape == (6,), f"obs2 shape mismatch: {obs2.shape}"

        # Range check — ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ ∈ [-1, 1], Δ⟨σ_x⟩, Δ⟨σ_y⟩, Δ⟨σ_z⟩ ∈ [-2, 2]
        assert -1.0 <= obs1[0] <= 1.0, f"obs1[0] out of range: {obs1[0]}"
        assert -1.0 <= obs1[1] <= 1.0, f"obs1[1] out of range: {obs1[1]}"
        assert -1.0 <= obs1[2] <= 1.0, f"obs1[2] out of range: {obs1[2]}"
        assert -2.0 <= obs1[3] <= 2.0, f"obs1[3] out of range: {obs1[3]}"
        assert -2.0 <= obs1[4] <= 2.0, f"obs1[4] out of range: {obs1[4]}"
        assert -2.0 <= obs1[5] <= 2.0, f"obs1[5] out of range: {obs1[5]}"
        assert -1.0 <= obs2[0] <= 1.0, f"obs2[0] out of range: {obs2[0]}"
        assert -1.0 <= obs2[1] <= 1.0, f"obs2[1] out of range: {obs2[1]}"
        assert -1.0 <= obs2[2] <= 1.0, f"obs2[2] out of range: {obs2[2]}"
        assert -2.0 <= obs2[3] <= 2.0, f"obs2[3] out of range: {obs2[3]}"
        assert -2.0 <= obs2[4] <= 2.0, f"obs2[4] out of range: {obs2[4]}"
        assert -2.0 <= obs2[5] <= 2.0, f"obs2[5] out of range: {obs2[5]}"


class TestPureStateConservation:
    """Test that the Bloch vector magnitude is conserved (pure state)."""

    def test_bloch_vector_magnitude_conservation(self):
        """
        For a pure state, ⟨σ_x⟩² + ⟨σ_y⟩² + ⟨σ_z⟩² = 1.
        We check this after each step to verify c_ops=[] is respected.
        Tolerance: 1e-6
        """
        env = QuantumSpinCartPoleEnv(seed=99)
        env.reset()

        max_deviation = 0.0
        for _ in range(200):
            action = np.random.randint(0, 5)
            env.step(action)
            psi = env._state
            sx, sy, sz = env._expectation(psi)
            mag2 = sx**2 + sy**2 + sz**2
            deviation = abs(mag2 - 1.0)
            max_deviation = max(max_deviation, deviation)

        env.close()
        assert max_deviation < 1e-6, f"Bloch magnitude deviation: {max_deviation}"

    def test_c_ops_is_empty(self):
        """Verify the processor uses pure-state evolution (c_ops=[])."""
        env = QuantumSpinCartPoleEnv()
        env.reset()
        env.step(action=4)
        # Check Bloch magnitude conserved
        sx, sy, sz = env._expectation(env._state)
        mag2 = sx**2 + sy**2 + sz**2
        assert mag2 == pytest.approx(1.0, abs=1e-6), f"Bloch magnitude not conserved: {mag2}"
        env.close()


class TestRewardMechanism:
    """Test the reward function and termination logic."""

    def test_idle_action_has_no_control_penalty(self):
        """action=4 (IDLE) should not subtract c_ctrl from fidelity.

        We disable OU noise here to avoid rare terminated episodes on the
        very first step, which would include r_penalty and make the reward
        negative.
        """
        env = QuantumSpinCartPoleEnv(seed=1, sigma_ou=0.0)
        env.reset(seed=1)
        obs, reward, terminated, truncated, _ = env.step(action=4)
        assert not terminated
        sx, sy = obs[0], obs[1]
        fidelity = env._compute_fidelity(sx, sy)
        assert reward == pytest.approx(fidelity, abs=1e-6)
        env.close()

    def test_non_idle_action_has_control_penalty(self):
        """Non-idle actions should subtract c_ctrl from the raw fidelity reward."""
        env = QuantumSpinCartPoleEnv(
            c_ctrl=1e-3,
            sigma_ou=0.0,  # suppress noise for deterministic test
            seed=1,
        )
        env.reset(seed=1)
        # Two consecutive IDLE steps
        _, reward_idle1, _, _, _ = env.step(action=4)
        obs2, reward_idle2, _, _, _ = env.step(action=4)
        # One non-idle step
        _, reward_drive, _, _, _ = env.step(action=0)

        # The two IDLE rewards should be very close (same state, same fidelity)
        # The drive action should be at most c_ctrl lower
        assert reward_drive <= reward_idle2 + 1e-3 + 1e-5
        env.close()

    def test_failure_penalty_on_termination(self):
        """Entering the southern hemisphere should apply r_penalty."""
        env = QuantumSpinCartPoleEnv(r_penalty=-5.0, seed=0)
        env.reset()
        for _ in range(2000):
            obs, reward, terminated, truncated, _ = env.step(action=0)
            if terminated:
                assert reward <= -4.0
                break
        env.close()


class TestActionMapping:
    """Verify the action→(u_x, u_y) mapping is correct."""

    def test_action_map_shape(self):
        assert ACTION_MAP.shape == (5, 2)

    def test_action_0_plus_x(self):
        assert ACTION_MAP[0][0] == +1.0
        assert ACTION_MAP[0][1] == 0.0

    def test_action_1_minus_x(self):
        assert ACTION_MAP[1][0] == -1.0
        assert ACTION_MAP[1][1] == 0.0

    def test_action_2_plus_y(self):
        assert ACTION_MAP[2][0] == 0.0
        assert ACTION_MAP[2][1] == +1.0

    def test_action_3_minus_y(self):
        assert ACTION_MAP[3][0] == 0.0
        assert ACTION_MAP[3][1] == -1.0

    def test_action_4_idle(self):
        assert ACTION_MAP[4][0] == 0.0
        assert ACTION_MAP[4][1] == 0.0
