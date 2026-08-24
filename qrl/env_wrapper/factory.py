"""Environment factory — decouples qrl/ from games/ via gymnasium registry.

Strict rule: this module MUST NOT contain ``from games import`` or ``import games``.
Games register themselves with gymnasium.envs.registration; this factory only calls
``gymnasium.make()``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


def make_env(
    env_id: str,
    env_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Create a gymnasium environment by ID, passing optional keyword arguments.

    Args:
        env_id: Gymnasium-registered environment ID, e.g. ``"CartPole-v1"`` or
            ``"QuantumSpinCartPole-v0"``.
        env_params: Optional keyword arguments forwarded to the environment constructor.
        seed: Random seed passed to ``gymnasium.make()`` via the ``seed`` kwarg
            (only supported for environments that accept it).

    Returns:
        A gymnasium environment instance.

    Raises:
        ModuleNotFoundError: If ``env_id`` is not registered in the gymnasium
            registry (i.e. the corresponding games package has not been imported).
    """
    kwargs: dict[str, Any] = {}
    if env_params:
        kwargs.update(env_params)

    try:
        env = gym.make(env_id, **kwargs)
    except gym.error.NameNotFound as exc:
        raise ModuleNotFoundError(
            f"Environment {env_id!r} not found in gymnasium registry. "
            "Ensure the corresponding games package is installed and imported "
            "(e.g. ``import games.spin_cartpole`` to register QuantumSpinCartPole-v0)."
        ) from exc

    # Set seed via env.reset() — seed is NOT a gym.make() kwarg for CartPole-v1
    if seed is not None:
        env.reset(seed=seed)

    return env


def make_vector_env(
    env_id: str,
    n_envs: int = 1,
    env_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> gym.VectorEnv:
    """Create a vectorized gymnasium environment (sync, to avoid QuTiP fork deadlocks).

    Uses ``gymnasium.vector.SyncVectorEnv`` rather than ``AsyncVectorEnv`` because
    QuTiP's C extensions are not fork-safe; parallel subprocess workers can deadlock
    on PyTorch CUDA initialisation when forking.

    Args:
        env_id: Gymnasium-registered environment ID.
        n_envs: Number of parallel environments. ``1`` returns a single env
            (not wrapped in a vector).
        env_params: Forwarded to each ``make_env`` call.
        seed: Base seed; each worker receives ``seed + i`` for ``i in range(n_envs)``.

    Returns:
        A ``SyncVectorEnv`` when ``n_envs > 1``, otherwise a single ``gym.Env``.
    """
    if n_envs == 1:
        return make_env(env_id, env_params=env_params, seed=seed)

    def _make_single(i: int) -> gym.Env:
        s = (seed + i) if seed is not None else None
        return make_env(env_id, env_params=env_params, seed=s)

    return gym.vector.SyncVectorEnv([lambda i=i: _make_single(i) for i in range(n_envs)])
