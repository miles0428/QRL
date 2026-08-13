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

# OU noise diffusion term (unified for all three axes, MHz·ns⁻¹/² → rad·ns⁻¹/²)
# sigma = 2π × 1.0 MHz·ns⁻¹/²; noise/Rabi ≈ 2%, clean coherent rotation
_SIGMA_OU_MHZ = 1.0  # MHz·ns⁻¹/²
SIGMA_OU = 2.0 * np.pi * _SIGMA_OU_MHZ  # rad·ns⁻¹/²  (≈ 6.283 rad·ns⁻¹/²)

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
