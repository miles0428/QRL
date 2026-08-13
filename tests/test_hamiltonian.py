"""
Unit tests for the hamiltonian module.
"""
import numpy as np
import qutip as qt
import pytest

from quantum_spin_cartpole import hamiltonian as hm


class TestPauliMatrices:
    """Test that Pauli operators have correct properties."""

    def test_sigma_x_squared_is_identity(self):
        assert qt.sigmax() ** 2 == qt.identity(2)

    def test_sigma_y_squared_is_identity(self):
        assert qt.sigmay() ** 2 == qt.identity(2)

    def test_sigma_z_squared_is_identity(self):
        assert qt.sigmaz() ** 2 == qt.identity(2)

    def test_sigma_x_sigma_y_anticommutator(self):
        """{σ_x, σ_y} = 0 (anticommute)."""
        anticomm = qt.sigmax() * qt.sigmay() + qt.sigmay() * qt.sigmax()
        assert anticomm.tr() == 0.0  # off-diagonal only → trace is 0

    def test_trace_of_pauli_is_zero(self):
        """All Pauli matrices are traceless."""
        for sigma in [qt.sigmax(), qt.sigmay(), qt.sigmaz()]:
            assert sigma.tr() == pytest.approx(0.0)


class TestHamiltonianTerms:
    """Test Hamiltonian factory functions."""

    def test_static_zeeman_is_hermitian(self):
        H0 = hm.build_static_zeeman(omega0=2.0 * np.pi)
        assert H0.isherm

    def test_static_zeeman_at_omega0_zero(self):
        H0 = hm.build_static_zeeman(omega0=0.0)
        assert H0 == qt.Qobj([[0, 0], [0, 0]])

    def test_noise_operator_prefactor(self):
        """Noise operator is (ħ/2) σ_z with zero trace."""
        H_noise = hm.build_noise_z()
        assert H_noise.isoper
        assert H_noise.tr() == pytest.approx(0.0)

    def test_noise_x_operator_is_hermitian(self):
        """x-axis noise operator should be Hermitian."""
        H_nx = hm.build_noise_x()
        assert H_nx.isherm
        assert H_nx.tr() == pytest.approx(0.0)

    def test_noise_y_operator_is_hermitian(self):
        """y-axis noise operator should be Hermitian."""
        H_ny = hm.build_noise_y()
        assert H_ny.isherm
        assert H_ny.tr() == pytest.approx(0.0)

    def test_noise_z_operator_is_hermitian(self):
        """z-axis noise operator should be Hermitian."""
        H_nz = hm.build_noise_z()
        assert H_nz.isherm
        assert H_nz.tr() == pytest.approx(0.0)

    def test_noise_x_different_from_noise_y(self):
        """x and y noise operators should be different."""
        H_nx = hm.build_noise_x()
        H_ny = hm.build_noise_y()
        assert H_nx != H_ny

    def test_drive_x_is_hermitian(self):
        Hx = hm.build_drive_x(Omega_R=2.0 * np.pi)
        assert Hx.isherm

    def test_drive_y_is_hermitian(self):
        Hy = hm.build_drive_y(Omega_R=2.0 * np.pi)
        assert Hy.isherm

    def test_drive_x_at_omega_r_zero(self):
        Hx = hm.build_drive_x(Omega_R=0.0)
        assert Hx == qt.Qobj([[0, 0], [0, 0]])

    def test_drive_y_at_omega_r_zero(self):
        Hy = hm.build_drive_y(Omega_R=0.0)
        assert Hy == qt.Qobj([[0, 0], [0, 0]])


class TestCreateWavefunction:
    """Test Bloch-sphere state construction."""

    def test_normalization(self):
        """All constructed states must be normalized."""
        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            for phi in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                psi = hm.create_wavefunction(theta, phi)
                assert psi.norm() == pytest.approx(1.0)

    def test_spin_up_at_theta_0(self):
        """θ=0 should give |0⟩ (spin-up along +z)."""
        psi = hm.create_wavefunction(theta=0.0, phi=0.0)
        expected = qt.basis(2, 0)
        assert (psi - expected).norm() < 1e-12

    def test_spin_down_at_theta_pi(self):
        """θ=π should give |1⟩ (spin-down along -z)."""
        psi = hm.create_wavefunction(theta=np.pi, phi=0.0)
        expected = qt.basis(2, 1)
        assert (psi - expected).norm() < 1e-12

    def test_equatorial_state(self):
        """θ=π/2, φ=0 gives (|0⟩+|1⟩)/√2 (|+x⟩)."""
        psi = hm.create_wavefunction(theta=np.pi / 2, phi=0.0)
        expected = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        assert (psi - expected).norm() < 1e-12

    def test_bloch_vector_magnitude(self):
        """
        For any Bloch state, |⟨S⟩| = ħ/2.
        We check ⟨S_x⟩² + ⟨S_y⟩² + ⟨S_z⟩² = (ħ/2)² = 1 (with ħ=1).
        """
        sx_op = qt.sigmax() / 2
        sy_op = qt.sigmay() / 2
        sz_op = qt.sigmaz() / 2
        for theta in np.linspace(0.1, np.pi - 0.1, 5):
            for phi in np.linspace(0, 2 * np.pi, 5, endpoint=False):
                psi = hm.create_wavefunction(theta, phi)
                # In qutip 5.x, .tr() returns a complex number; use .tr().real
                sx = float((psi.dag() * sx_op * psi).real)
                sy = float((psi.dag() * sy_op * psi).real)
                sz = float((psi.dag() * sz_op * psi).real)
                mag2 = sx**2 + sy**2 + sz**2
                assert mag2 == pytest.approx(0.25, abs=1e-12)
