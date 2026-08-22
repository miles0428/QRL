"""QRL models package — VQC and MLP builders for all algorithm families."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

from qrl.config_schema import algo_for


def build_model(
    cfg: dict,
    obs_space: "gym.Space",
    action_space: "gym.Space",
) -> object:
    """Build and return the correct model(s) for the given configuration.

    Routing is based on ``cfg["type"]`` (e.g. ``"vqc_a2c"``, ``"mlp_policy"``,
    ``"cnn_dqn"``), which maps to an algorithm family via
    :func:`qrl.config_schema.algo_for`.

    - **DQN** (``cfg["type"]`` in ``{"vqc", "mlp", "cnn_dqn"}``):
        Returns a single Q-network.
    - **PG**  (``cfg["type"]`` in ``{"vqc_policy", "mlp_policy", "cnn_pg"}``):
        Returns a single policy network.
    - **A2C** (``cfg["type"]`` in ``{"vqc_a2c", "mlp_a2c", "q2c", "a2q", "cnn_a2c"}``):
        Returns a tuple ``(actor, critic)``.

    Args:
        cfg: Flat dict config as returned by :func:`qrl.config.load_config`.
        obs_space: Gymnasium observation space (used for input-dimension checks).
        action_space: Gymnasium action space (used for output-dimension checks).

    Returns:
        A model instance, or a tuple ``(actor, critic)`` for A2C configs.

    Raises:
        ValueError: If ``cfg["type"]`` is not a known model family.
    """
    model_type = cfg["type"]
    algo = algo_for(model_type)

    if algo == "dqn":
        return _build_dqn_model(cfg, obs_space, action_space)
    elif algo == "pg":
        return _build_pg_model(cfg, obs_space, action_space)
    elif algo == "a2c":
        return _build_a2c_models(cfg, obs_space, action_space)
    else:
        raise ValueError(
            f"Unknown algorithm {algo!r} for model type {model_type!r}"
        )


# ---------------------------------------------------------------------------
# Algorithm-specific builders
# ---------------------------------------------------------------------------


def _build_dqn_model(cfg, obs_space, action_space):
    """Build a DQN Q-network (VQC, MLP, ResNet+MLP, or CNN+VQC)."""
    # Lazy import to avoid circular deps and keep qrl/ free of games/env imports
    from qrl.models.vqc import VQCQFunction
    from qrl.models.mlp import MLPQFunction
    from qrl.models.cnn import CNNVQCDQNActorCritic

    hidden = cfg["hidden"] or 6

    if cfg["type"] == "cnn_dqn":
        vqc_cfg = cfg.get("vqc", {})
        return CNNVQCDQNActorCritic(
            n_actions=int(action_space.n),
            n_qubits=vqc_cfg.get("n_qubits", 6),
            n_layers=vqc_cfg.get("n_layers", 3),
            reuploading=vqc_cfg.get("reuploading", True),
            observable=vqc_cfg.get("observable", "zz"),
            entangler=vqc_cfg.get("entangler", "cx"),
            hidden=cfg["hidden"] or 256,
        )
    elif cfg["type"].startswith("vqc"):
        vqc_cfg = cfg.get("vqc", {})
        return VQCQFunction(
            n_actions=int(action_space.n),
            n_qubits=cfg.get("n_qubits") or vqc_cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers") or vqc_cfg.get("n_layers", 5),
            reuploading=cfg.get("reuploading") if cfg.get("reuploading") is not None else vqc_cfg.get("reuploading", True),
            observables=cfg.get("observables") or vqc_cfg.get("observables", ["ZZII", "IIZZ"]),
            backend=vqc_cfg.get("backend", "torch_sv"),
        )
    else:
        # Classical MLP
        return MLPQFunction(
            n_inputs=obs_space.shape[0] if hasattr(obs_space, "shape") else int(obs_space.n),
            n_actions=int(action_space.n),
            hidden=hidden,
        )


def _build_pg_model(cfg, obs_space, action_space):
    """Build a REINFORCE policy network (VQC, MLP, or CNN+VQC)."""
    from qrl.models.vqc_policy import VQCPolicy
    from qrl.models.mlp import MLPPolicy
    from qrl.models.cnn import CNNVQCPolicy

    hidden = cfg["hidden"] or 6

    if cfg["type"] == "cnn_pg":
        vqc_cfg = cfg.get("vqc", {})
        return CNNVQCPolicy(
            n_actions=int(action_space.n),
            n_qubits=vqc_cfg.get("n_qubits", 6),
            n_layers=vqc_cfg.get("n_layers", 3),
            reuploading=vqc_cfg.get("reuploading", True),
            observable=vqc_cfg.get("observable", "zz"),
            entangler=vqc_cfg.get("entangler", "cx"),
            hidden=cfg["hidden"] or 256,
        )
    elif cfg["type"].startswith("vqc"):
        vqc_cfg = cfg.get("vqc", {})
        return VQCPolicy(
            n_actions=int(action_space.n),
            n_qubits=cfg.get("n_qubits") or vqc_cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers") or vqc_cfg.get("n_layers", 5),
            reuploading=cfg.get("reuploading") if cfg.get("reuploading") is not None else vqc_cfg.get("reuploading", True),
            observables=cfg.get("observables") or vqc_cfg.get("observables", ["ZZII", "IIZZ"]),
            backend=vqc_cfg.get("backend", "torch_sv"),
        )
    else:
        return MLPPolicy(
            n_inputs=obs_space.shape[0] if hasattr(obs_space, "shape") else int(obs_space.n),
            n_actions=int(action_space.n),
            hidden=hidden,
        )


def _build_a2c_models(cfg, obs_space, action_space):
    """Build actor and critic for A2C (VQC, MLP, or CNN+VQC)."""
    from qrl.models.vqc_policy import VQCPolicy
    from qrl.models.vqc_value import VQCValue
    from qrl.models.mlp import MLPPolicy, MLPValue
    from qrl.models.cnn import CNNVQCA2CActorCritic

    hidden = cfg["hidden"] or 6
    vqc_cfg = cfg.get("vqc", {})

    if cfg["type"] == "cnn_a2c":
        actor_critic = CNNVQCA2CActorCritic(
            n_actions=int(action_space.n),
            n_qubits=vqc_cfg.get("n_qubits", 6),
            n_layers=vqc_cfg.get("n_layers", 3),
            reuploading=vqc_cfg.get("reuploading", True),
            observable=vqc_cfg.get("observable", "zz"),
            entangler=vqc_cfg.get("entangler", "cx"),
            hidden=cfg["hidden"] or 256,
        )
        return actor_critic, actor_critic
    elif cfg["type"].startswith("vqc"):
        actor = VQCPolicy(
            n_actions=int(action_space.n),
            n_qubits=cfg.get("n_qubits") or vqc_cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers") or vqc_cfg.get("n_layers", 5),
            reuploading=cfg.get("reuploading") if cfg.get("reuploading") is not None else vqc_cfg.get("reuploading", True),
            observables=cfg.get("observables") or vqc_cfg.get("observables", ["ZZII", "IIZZ"]),
            backend=vqc_cfg.get("backend", "torch_sv"),
        )
        critic = VQCValue(
            n_qubits=cfg.get("n_qubits") or vqc_cfg.get("n_qubits", 4),
            n_layers=cfg.get("n_layers") or vqc_cfg.get("n_layers", 5),
            reuploading=cfg.get("reuploading") if cfg.get("reuploading") is not None else vqc_cfg.get("reuploading", True),
            observable=cfg.get("critic_observable") or vqc_cfg.get("critic_observable", "ZZZZ"),
            backend=vqc_cfg.get("backend", "torch_sv"),
        )
    else:
        actor = MLPPolicy(
            n_inputs=obs_space.shape[0] if hasattr(obs_space, "shape") else int(obs_space.n),
            n_actions=int(action_space.n),
            hidden=hidden,
        )
        critic = MLPValue(
            n_inputs=obs_space.shape[0] if hasattr(obs_space, "shape") else int(obs_space.n),
            hidden=cfg.get("critic_hidden", 7),
        )

    return actor, critic


__all__ = ["build_model"]
