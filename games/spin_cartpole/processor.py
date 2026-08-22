"""
ModelProcessor wrapper for the Quantum Spin-CartPole V2 environment.

This module intentionally uses ``qutip.qip.device.ModelProcessor`` as the
backend (per blueprint).

At each environment step, we create three discrete pulses (X/Y/Z channels)
for the interval [0, dt] and evolve a pure state with ``c_ops = []``.

Hamiltonian form:
    H(t) = f_x · (ħ/2)σ_x + f_y · (ħ/2)σ_y + f_z · (ħ/2)σ_z

Where (per step):
    f_x = Ω_R·u_x + δΔ_x
    f_y = Ω_R·u_y + δΔ_y
    f_z = δΔ_z
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import qutip as qt
from qutip.qip.device import ModelProcessor
from qutip.qip.pulse import Pulse

from .hamiltonian import SIGMA_X, SIGMA_Y, SIGMA_Z


class SpinProcessor:
    """A thin wrapper around QuTiP's ModelProcessor for single-qubit dynamics."""

    def __init__(self, Omega_R: float, dt: float):
        """
        Initialize the spin processor.

        Args:
            Omega_R: Rabi angular frequency (rad/ns).
            dt: Simulation timestep (ns).
        """
        self.Omega_R = Omega_R
        self.dt = float(dt)

        self._proc = ModelProcessor(num_qubits=1)
        self._proc.c_ops = []  # No collapse operators

        # Time-independent operator parts (ħ = 1)
        self._sx_op = (1.0 / 2.0) * SIGMA_X
        self._sy_op = (1.0 / 2.0) * SIGMA_Y
        self._sz_op = (1.0 / 2.0) * SIGMA_Z

        self._tlist = np.array([0.0, self.dt], dtype=float)

    def run_state(self, init_state: qt.Qobj, fx: float, fy: float, fz: float) -> qt.Qobj:
        """
        Evolve the quantum state for one timestep.

        Args:
            init_state: Initial ket state.
            fx: X-channel total amplitude.
            fy: Y-channel total amplitude.
            fz: Z-channel total amplitude.

        Returns:
            qt.Qobj: Final ket state after one timestep.
        """
        # Clear previous pulses and add three channel pulses for this step.
        self._proc.clear_pulses()

        # In discrete pulse mode, coeff has length len(tlist)-1.
        self._proc.add_pulse(
            Pulse(self._sx_op, targets=[0], tlist=self._tlist, coeff=np.array([float(fx)]))
        )
        self._proc.add_pulse(
            Pulse(self._sy_op, targets=[0], tlist=self._tlist, coeff=np.array([float(fy)]))
        )
        self._proc.add_pulse(
            Pulse(self._sz_op, targets=[0], tlist=self._tlist, coeff=np.array([float(fz)]))
        )

        result = self._proc.run_state(init_state=init_state, tlist=self._tlist)
        return result.final_state
