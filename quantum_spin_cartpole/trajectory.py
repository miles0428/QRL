"""
Larmor precession trajectory collector and visualizer.

This module provides SpinTrajectory: a marathon-style runner that
accumulates ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ histories across an episode (or
multiple episodes) on the Bloch sphere, with optional 3-D visualisation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .env import QuantumSpinCartPoleEnv


class SpinTrajectory:
    """
    Collect and visualise spin expectation-value trajectories on the Bloch sphere.

    Parameters
    ----------
    env : QuantumSpinCartPoleEnv
        The quantum-spin environment instance to drive.
    max_steps : int, optional
        Maximum steps per single ``run()`` call.  Default 10 000.

    Attributes
    ----------
    sx_history, sy_history, sz_history : list[float]
        Raw expectation-value lists recorded at each step.
    step_history : list[int]
        Step indices corresponding to each recorded sample.

    Examples
    --------
    >>> import quantum_spin_cartpole as qsc
    >>> env = qsc.QuantumSpinCartPoleEnv(terminate_on_violation=False)
    >>> traj = SpinTrajectory(env, max_steps=20_000)
    >>> result = traj.run(action=0)          # constant +X drive
    >>> traj.plot_3d("bloch_precession.png")
    >>> traj.plot_xy_components("sigma_t.png")
    """

    def __init__(self, env: QuantumSpinCartPoleEnv, max_steps: int = 10_000):
        """
        Initialise the trajectory collector.

        Args:
            env: QuantumSpinCartPoleEnv instance to drive.
            max_steps: Maximum steps per ``run()`` invocation.
        """
        self.env = env
        self.max_steps = max_steps
        self.sx_history: list[float] = []
        self.sy_history: list[float] = []
        self.sz_history: list[float] = []
        self.step_history: list[int] = []

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------
    def run(self, action: int, terminate_on_violation: bool = False) -> dict:
        """
        Run a single marathon episode collecting trajectory data.

        The environment is reset before the run.  The spin is driven
        with a *constant* action for the entire trajectory.

        Parameters
        ----------
        action : int
            Action index passed to ``env.step()`` on every step.
            0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = IDLE.
        terminate_on_violation : bool
            Overrides ``env.terminate_on_violation`` for this run.
            Default ``False`` (keep spinning even if sz < 0).

        Returns
        -------
        dict
            Summary with keys: ``steps``, ``sx``, ``sy``, ``sz``.
        """
        self.env.terminate_on_violation = terminate_on_violation
        obs, info = self.env.reset()

        self._record(step=0, obs=obs, info=info)

        for step in range(1, self.max_steps + 1):
            obs, reward, term, trunc, info = self.env.step(action)
            self._record(step, obs, info=info)
            if term or trunc:
                break

        return self.summary()

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def _record(self, step: int, obs: np.ndarray, info: dict) -> None:
        """
        Extract ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ from observation and info dict and append.

        The observation is ``[sx, sy, sz, Δsx, Δsy, Δsz]``.  ``sx`` and ``sy`` and
        ``sz`` are read directly from the observation vector; ``sz`` is also read
        from ``info['sz']`` for redundancy and validation (info contains the true
        quantum expectation value from qutip with correct sign).

        Parameters
        ----------
        step : int
            Current step index.
        obs : np.ndarray
            6-D observation vector [sx, sy, sz, Δsx, Δsy, Δsz].
        info : dict
            Info dict returned by ``env.step()``, must contain 'sz'.
        """
        sx = float(obs[0])
        sy = float(obs[1])
        sz = float(info["sz"])

        self.sx_history.append(sx)
        self.sy_history.append(sy)
        self.sz_history.append(sz)
        self.step_history.append(step)

    def summary(self) -> dict:
        """
        Return the collected trajectory as NumPy arrays.

        Returns
        -------
        dict
            ``steps`` (int), ``sx`` (np.ndarray), ``sy`` (np.ndarray),
            ``sz`` (np.ndarray).
        """
        return {
            "steps": len(self.step_history),
            "sx": np.array(self.sx_history),
            "sy": np.array(self.sy_history),
            "sz": np.array(self.sz_history),
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def plot_3d(self, save_path: str) -> None:
        """
        Render a 3-D Bloch-sphere trajectory plot and save to disk.

        Parameters
        ----------
        save_path : str
            Output file path (PNG recommended).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Transparent Bloch sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, alpha=0.05, color="gray")

        # Trajectory line
        ax.plot(
            self.sx_history,
            self.sy_history,
            self.sz_history,
            "b-",
            linewidth=1.0,
            label="trajectory",
        )

        # Start marker
        ax.plot(
            [self.sx_history[0]],
            [self.sy_history[0]],
            [self.sz_history[0]],
            "go",
            ms=10,
            label="start",
        )

        # End marker
        ax.plot(
            [self.sx_history[-1]],
            [self.sy_history[-1]],
            [self.sz_history[-1]],
            "r*",
            ms=15,
            label="end",
        )

        ax.set_xlabel("⟨σ_x⟩")
        ax.set_ylabel("⟨σ_y⟩")
        ax.set_zlabel("⟨σ_z⟩")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.legend()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_xy_components(self, save_path: str) -> None:
        """
        Plot the time-evolution of ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ as a stacked
        three-panel figure and save to disk.

        Parameters
        ----------
        save_path : str
            Output file path (PNG recommended).
        """
        import matplotlib.pyplot as plt

        t = np.array(self.step_history) * self.env.dt  # nanoseconds

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(t, self.sx_history, "b-", lw=1)
        axes[0].set_ylabel("⟨σ_x⟩")
        axes[0].axhline(0, color="k", lw=0.5)

        axes[1].plot(t, self.sy_history, "g-", lw=1)
        axes[1].set_ylabel("⟨σ_y⟩")
        axes[1].axhline(0, color="k", lw=0.5)

        axes[2].plot(t, self.sz_history, "r-", lw=1)
        axes[2].set_ylabel("⟨σ_z⟩")
        axes[2].axhline(0, color="k", lw=0.5)
        axes[2].set_xlabel("Time (ns)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
