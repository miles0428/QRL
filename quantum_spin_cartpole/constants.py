"""
Physical constants and default hyperparameters for the Quantum Spin-CartPole environment.

All parameters follow the V2 blueprint specification.
"""
import numpy as np

# -----------------------------------------------------------------------------
# Physical Constants
# -----------------------------------------------------------------------------
HBAR = 1.0  # Reduced Planck constant (normalized to 1)

# -----------------------------------------------------------------------------
# Default Hyperparameters (V2)
# -----------------------------------------------------------------------------
# Simulation timestep (ns)
DT = 1.0  # ns

# Rabi drive strength (Ω_R = 2π × 100.0 MHz → rad/ns)
# Noise/Rabi ratio ≈ 2%, clean coherent rotation
OMEGA_R = 2.0 * np.pi * 100.0e-3  # rad/ns  (≈ 0.62832 rad/ns)

# OU noise decay rate (single unified value for all three axes, ns⁻¹)
THETA_OU = 0.05  # ns⁻¹

# OU noise diffusion term (unified for all three axes), in rad·ns⁻¹/².
#
# Set from the physically meaningful quantity: the ratio of the noise field's
# stationary standard deviation to the Rabi frequency. The OU process has
# stationary std sigma/sqrt(2*theta), so
#
#     sigma = ratio * OMEGA_R * sqrt(2 * THETA_OU)
#
# 0.35 is the usable difficulty. Measured with a one-step-lookahead controller,
# 20 episodes per point: 0.03 -> 491 (100% survive), 0.10 -> 491 (100%),
# 0.25 -> 477 (95%), 0.35 -> 400 (68%), 0.50 -> 138 (0%), 0.75 -> 31,
# >= 2.0 -> indistinguishable from doing nothing. Below 0.25 the task is
# saturated and a learned policy cannot show any advantage; above ~0.5 no policy
# has the control authority to matter. 0.35 leaves ~100 points of headroom.
#
# This previously read `SIGMA_OU = 2*pi*1.0`, which is 1000x larger. OMEGA_R
# encodes 100 MHz as 2*pi*100.0e-3 -- GHz units, which is what rad/ns requires --
# so by the same convention 1 MHz is 2*pi*1.0e-3, and the missing e-3 put the
# noise at 31.6x the Rabi frequency rather than the ~2% the comment claimed. At
# that level the environment is uncontrollable: greedy, random and always-idle
# all score about -2 over 3.6 steps and are statistically indistinguishable.
NOISE_RABI_RATIO = 0.35
SIGMA_OU = NOISE_RABI_RATIO * OMEGA_R * np.sqrt(2.0 * THETA_OU)  # ≈ 0.06954

# Episode maximum step count
T_MAX = 500

# Control penalty coefficient
C_CTRL = 1e-3

# Failure penalty (applied when sz < 0)
R_PENALTY = -5.0

# -----------------------------------------------------------------------------
# Observation Space Bounds (6D)
# -----------------------------------------------------------------------------
# o_t = [⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩, Δ⟨σ_x⟩, Δ⟨σ_y⟩, Δ⟨σ_z⟩]
OBS_LOW = np.array([-1.0, -1.0, -1.0, -2.0, -2.0, -2.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)

# -----------------------------------------------------------------------------
# Action Space
# -----------------------------------------------------------------------------
# Discrete(5): 0(+X), 1(-X), 2(+Y), 3(-Y), 4(IDLE)
# Maps action index → (u_x, u_y) coherent drive amplitudes
ACTION_MAP = np.array(
    [[+1.0, 0.0], [-1.0, 0.0], [0.0, +1.0], [0.0, -1.0], [0.0, 0.0]],
    dtype=np.float32,
)
