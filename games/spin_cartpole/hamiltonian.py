"""
Hamiltonian factory for the Quantum Spin-CartPole V2 environment.

Generates operator-coefficient pairs for the three control channels
(X, Y, Z) used in the ModelProcessor framework.

Each channel k ∈ {x, y, z} takes the form:
    H_k(t) = (ħ/2) · σ_k · f_k(t)

Where:
    f_x(t) = Ω_R · u_x(a_t) + δΔ_x(t)
    f_y(t) = Ω_R · u_y(a_t) + δΔ_y(t)
    f_z(t) = δΔ_z(t)

The old lab-frame building functions (build_static_zeeman, build_drive_x, etc.)
are retained for backward compatibility with existing tests.
"""
import numpy as np
import qutip as qt
from typing import Callable, Tuple

# Pauli matrices
SIGMA_X = qt.sigmax()
SIGMA_Y = qt.sigmay()
SIGMA_Z = qt.sigmaz()
IDENTITY = qt.identity(2)

# -----------------------------------------------------------------------------
# Channel factory functions (V2 — ModelProcessor interface)
# -----------------------------------------------------------------------------
# Each returns (operator, coefficient_func) where operator = (ħ/2)·σ_k
# and coefficient_func returns the total amplitude f_k(t) at runtime.


def make_X_channel(
    Omega_R: float,
) -> Tuple[qt.Qobj, Callable[[float, dict], float]]:
    """
    Create the X-channel operator and its coefficient function.

    The X channel Hamiltonian is:
        H_x(t) = (ħ/2) · σ_x · f_x(t)
    where f_x(t) = Ω_R · u_x + δΔ_x.

    Args:
        Omega_R: Rabi angular frequency (rad/ns).

    Returns:
        tuple: (operator, coefficient_func)
            - operator: (ħ/2)·σ_x (Qobj)
            - coefficient_func: callable(f_k, args) returning f_x
    """
    op = (1.0 / 2.0) * SIGMA_X  # (ħ/2)·σ_x with ħ=1

    def coeff(f_k: float, args: dict) -> float:  # pragma: no cover
        # f_k is the X-channel amplitude passed via processor.set_data
        # args carries additional context (not used in this design)
        return f_k

    return op, coeff


def make_Y_channel(
    Omega_R: float,
) -> Tuple[qt.Qobj, Callable[[float, dict], float]]:
    """
    Create the Y-channel operator and its coefficient function.

    The Y channel Hamiltonian is:
        H_y(t) = (ħ/2) · σ_y · f_y(t)
    where f_y(t) = Ω_R · u_y + δΔ_y.

    Args:
        Omega_R: Rabi angular frequency (rad/ns).

    Returns:
        tuple: (operator, coefficient_func)
            - operator: (ħ/2)·σ_y (Qobj)
            - coefficient_func: callable(f_k, args) returning f_y
    """
    op = (1.0 / 2.0) * SIGMA_Y  # (ħ/2)·σ_y with ħ=1

    def coeff(f_k: float, args: dict) -> float:  # pragma: no cover
        return f_k

    return op, coeff


def make_Z_channel() -> Tuple[qt.Qobj, Callable[[float, dict], float]]:
    """
    Create the Z-channel operator and its coefficient function.

    The Z channel Hamiltonian is:
        H_z(t) = (ħ/2) · σ_z · f_z(t)
    where f_z(t) = δΔ_z(t) (pure noise, no coherent drive).

    Returns:
        tuple: (operator, coefficient_func)
            - operator: (ħ/2)·σ_z (Qobj)
            - coefficient_func: callable(f_k, args) returning f_z
    """
    op = (1.0 / 2.0) * SIGMA_Z  # (ħ/2)·σ_z with ħ=1

    def coeff(f_k: float, args: dict) -> float:  # pragma: no cover
        return f_k

    return op, coeff


# -----------------------------------------------------------------------------
# Legacy factory functions (retained for backward compatibility with tests)
# -----------------------------------------------------------------------------

def build_static_zeeman(omega0: float, hbar: float = 1.0) -> qt.Qobj:
    """
    Build the static Zeeman term H₀ = (ħ ω₀ / 2) σ_z.

    Args:
        omega0: Larmor angular frequency ω₀ = 2π·f₀, in rad·ns⁻¹.
        hbar: Reduced Planck constant (default 1.0, normalized).

    Returns:
        qt.Qobj: The static Zeeman Hamiltonian operator.
    """
    return (hbar * omega0 / 2.0) * SIGMA_Z


def build_noise_x(hbar: float = 1.0) -> qt.Qobj:
    """Build the x-axis noise operator: (ħ/2) σ_x."""
    return (hbar / 2.0) * SIGMA_X


def build_noise_y(hbar: float = 1.0) -> qt.Qobj:
    """Build the y-axis noise operator: (ħ/2) σ_y."""
    return (hbar / 2.0) * SIGMA_Y


def build_noise_z(hbar: float = 1.0) -> qt.Qobj:
    """Build the z-axis noise operator: (ħ/2) σ_z."""
    return (hbar / 2.0) * SIGMA_Z


def build_drive_x(Omega_R: float, hbar: float = 1.0) -> qt.Qobj:
    """Build the x-axis drive operator: (ħ Ω_R / 2) σ_x."""
    return (hbar * Omega_R / 2.0) * SIGMA_X


def build_drive_y(Omega_R: float, hbar: float = 1.0) -> qt.Qobj:
    """Build the y-axis drive operator: (ħ Ω_R / 2) σ_y."""
    return (hbar * Omega_R / 2.0) * SIGMA_Y


def create_wavefunction(theta: float, phi: float) -> qt.Qobj:
    """
    Create a pure spin-1/2 state from spherical coordinates on the Bloch sphere.

    The state is |ψ⟩ = cos(θ/2) |0⟩ + e^{iφ} sin(θ/2) |1⟩.

    Args:
        theta: Polar angle from +z axis (0 ≤ θ ≤ π), in radians.
        phi: Azimuthal angle in the xy-plane (0 ≤ φ < 2π), in radians.

    Returns:
        qt.Qobj: A normalized pure state ket (column vector).
    """
    ket0 = qt.basis(2, 0)  # |0⟩
    ket1 = qt.basis(2, 1)  # |1⟩
    psi = np.cos(theta / 2.0) * ket0 + np.exp(1.0j * phi) * np.sin(theta / 2.0) * ket1
    return psi.unit()
