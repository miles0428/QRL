"""Four-layer configuration schema for QRL experiments.

This module defines the four-layer (runtime / env / model / trainer) configuration
schema mandated by YC and adopted in BLUEPRINT §6. It supersedes the legacy
sectioned layout (``model:`` / ``trainer:`` / ``optim:`` / ``gradient:`` / ``eval:`` /
``critic:``) previously normalized by ``qrl.config.normalize_config``.

The dataclasses here are the canonical typed view of a QRL experiment:
* :class:`RuntimeConfig` -- experiment_name, seed, device, precision, output_dir
* :class:`EnvConfig`     -- gymnasium env id + arbitrary physics/parameter block
* :class:`ModelConfig`   -- encoder / vqc / head breakdown
* :class:`TrainerConfig` -- algorithm + hyperparams + optimizer
* :class:`ConfigSchema`  -- the four above in one bundle

``MODEL_ALGO`` and ``algo_for()`` are preserved verbatim from ``qrl.config`` so
the trainer dispatch logic does not depend on the wire format.

Why this lives in its own file:
    ``qrl.config`` is LOCKED (it was vendored from origin/a2c as the trainer-side
    parser, and the v3 trainers/sweep scripts depend on its flat-dict shape).
    The new four-layer view is a sibling parser; the old flat layout can be
    migrated to it independently and the two can coexist during the transition.

This module deliberately avoids qiskit / quantum framework imports: it is loaded
by ``scripts/train.py`` and friends, which themselves promise not to import
qiskit transitively at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Algorithm routing (preserved from qrl.config.MODEL_ALGO)
# ---------------------------------------------------------------------------

# Which trainer a model_type belongs to. scripts/train.py and
# scripts/train_pg.py each check this rather than silently running a policy
# through the TD loss (which would "work" -- softmax logits are shaped exactly
# like Q-values -- and produce a plausible, meaningless curve).
MODEL_ALGO: dict[str, str] = {
    "vqc": "dqn",
    "mlp": "dqn",
    "cnn_dqn": "dqn",
    "vqc_policy": "pg",
    "mlp_policy": "pg",
    "cnn_pg": "pg",
    "vqc_a2c": "a2c",  # Q2Q: quantum actor, quantum critic
    "mlp_a2c": "a2c",  # A2C: classical actor, classical critic
    "q2c": "a2c",      # Q2C: quantum actor, classical critic
    "a2q": "a2c",      # A2Q: classical actor, quantum critic
    "cnn_a2c": "a2c",
}


def algo_for(model_type: str) -> str:
    """Map a model_type identifier to its trainer algorithm.

    Args:
        model_type (str): One of the keys in :data:`MODEL_ALGO`.

    Returns:
        str: ``"dqn"`` / ``"pg"`` / ``"a2c"`` -- the trainer the model_type
        should be dispatched to.

    Raises:
        ValueError: If ``model_type`` is not a recognised model family.
    """
    if model_type not in MODEL_ALGO:
        raise ValueError(
            f"unknown model type {model_type!r}; expected one of {sorted(MODEL_ALGO)}"
        )
    return MODEL_ALGO[model_type]


# ---------------------------------------------------------------------------
# Dataclasses (four-layer schema)
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    """Runtime namespace -- who is running this experiment, where, and how.

    Fields:
        experiment_name (str): Human label; becomes the result/ subdirectory.
        seed (int): PRNG seed for torch / numpy / python random.
        device (str): ``"cpu"`` / ``"cuda"`` / ``"cuda:0"`` etc.
        precision (str): ``"float32"`` / ``"float64"``; downcasts VQC tensors.
        output_dir (str): Root directory for result/ artefacts.
    """

    experiment_name: str = ""
    seed: int = 0
    device: str = "cpu"
    precision: str = "float32"
    output_dir: str = "result/"


@dataclass
class EnvConfig:
    """Environment namespace -- which gym env and its configuration block.

    Fields:
        id (str): gymnasium environment id, e.g. ``"CartPole-v1"`` or
            ``"QuantumSpinCartPole-v0"``. The four-layer scheme intentionally
            keeps env_id out of the model/trainer namespaces.
        params (dict): Free-form physics/environment parameters (noise_scale,
            hamiltonian sub-block, etc.). Typed loosely because each game may
            define its own schema.
    """

    id: str = "CartPole-v1"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model namespace -- encoder / vqc / head breakdown.

    Fields:
        type (str): legacy identifier (``vqc``, ``mlp``, ``resnet_mlp_q``, ...).
            Kept because trainer dispatch reads it through ``algo_for``.
        encoder (str): Pre-VQC observation encoder name. ``"normalize_observation"``
            for the quantum baselines; ``"cnn"`` / ``"transformer"`` are open.
        vqc (dict): VQC body parameters (n_qubits, n_layers, reuploading,
            observables, per_layer_encoding, output_rescaling, ...).
        head (str): ``"policy"`` | ``"qfunction"`` | ``"value"`` -- tells the
            trainer which head semantics to attach.
        hidden (int): Width for the classical MLP variant.
        hidden_proj (int): Hidden dimension for ResNetMLPQFunction projection head.
        output_dim (int): Output dimension of the projection head (before Q-head).
    """

    type: str = ""
    encoder: str = "normalize_observation"
    vqc: dict[str, Any] = field(default_factory=dict)
    head: str = "policy"
    hidden: int = 6
    hidden_proj: int = 256
    output_dim: int = 6


@dataclass
class TrainerConfig:
    """Trainer namespace -- algorithm + hyperparameters + optimizer.

    Fields:
        algorithm (str): ``"dqn"`` | ``"pg"`` | ``"a2c"`` -- typically derived
            from ``ModelConfig.type`` via :func:`algo_for`, but stored explicitly
            so configs remain self-describing.
        hyperparams (dict): RL-side knobs (gamma, gae_lambda, n_envs,
            episodes_per_update, value_coef, ...).
        optimizer (dict): Optimizer-side knobs (lr_variational, lr_lam,
            lr_w, amsgrad, lr_critic_*, ...).
    """

    algorithm: str = ""
    hyperparams: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigSchema:
    """Bundle of the four top-level namespaces.

    A loaded config is one of these. ``load_config(path)`` returns it; trainers
    read ``config.runtime``, ``config.env``, ``config.model``, ``config.trainer``.
    """

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # -- convenience ---------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        """Return the four namespaces as a plain dict-of-dicts.

        Mirrors the on-disk YAML layout so validators that walk the file can
        also walk the in-memory object.
        """
        return {
            "runtime": asdict(self.runtime),
            "env": asdict(self.env),
            "model": asdict(self.model),
            "trainer": asdict(self.trainer),
        }


# ---------------------------------------------------------------------------
# Defaults (carried over from qrl.config.DEFAULTS)
# ---------------------------------------------------------------------------

# RL-side defaults extracted from qrl.config.DEFAULTS. ``model_type`` ->
# ``runtime.experiment_name`` is handled at load time, so it is intentionally
# absent here.
DEFAULTS: dict[str, Any] = {
    "auto_batch_size": False,
    "batch_size": 16,
    "buffer_capacity": 10_000,
    "min_buffer_size": 16,
    "gamma": 0.99,
    "max_steps_per_episode": 500,
    "solve_threshold": 475.0,
    "solve_window": 100,
    "env_id": "CartPole-v1",
    "eval_every": 50,
    "eval_episodes": 5,
    "final_eval_episodes": 100,
    "epsilon_schedule": "exponential_episodes",
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.99,
    "epsilon_decay_steps": 20_000,
    "amsgrad": True,
    "n_envs": 1,
    "steps_per_update": 10,
    "loss_fn": "huber",
    "output_rescaling": True,
    "per_layer_encoding": False,
    "reuploading": True,
    "observables": ["ZZII", "IIZZ"],
    "backend": "torch_sv",
    "gradient_method": "adjoint",
    "beta_init": 1.0,
    "trainable_beta": True,
    "lr_beta": 0.1,
    "episodes_per_update": 8,
    "baseline": "batch_mean",
    "normalize_advantages": True,
    "entropy_coef": 0.0,
    "max_grad_norm": None,
    "gae_lambda": 0.95,
    "value_coef": 0.5,
    "value_loss_fn": "huber",
    "critic_observable": "ZZZZ",
    "critic_hidden": 7,
    "critic_w_init": None,
    "lr_critic_vqc": 0.01,
    "lr_critic_lam": 0.01,
    "lr_critic_w": 0.1,
    "lr_critic": 0.05,
}


# ---------------------------------------------------------------------------
# Migration: legacy sectioned (origin/a2c) -> four-layer schema
# ---------------------------------------------------------------------------

# (legacy_section, legacy_key) -> (target_namespace, target_key)
# Mirrors qrl.config._SECTION_MAP, except that the targets are the new
# four-layer namespaces rather than flat names.
_LEGACY_TO_LAYERED: dict[tuple[str, str], tuple[str, str]] = {
    ("model", "type"): ("model", "type"),
    ("model", "n_qubits"): ("model.vqc", "n_qubits"),
    ("model", "n_layers"): ("model.vqc", "n_layers"),
    ("model", "reuploading"): ("model.vqc", "reuploading"),
    ("model", "observables"): ("model.vqc", "observables"),
    ("model", "hidden"): ("model", "hidden"),
    ("trainer", "episodes"): ("trainer.hyperparams", "max_episodes"),
    ("trainer", "batch_size"): ("trainer.hyperparams", "batch_size"),
    ("trainer", "auto_batch_size"): ("trainer.hyperparams", "auto_batch_size"),
    ("trainer", "gamma"): ("trainer.hyperparams", "gamma"),
    ("trainer", "buffer_capacity"): ("trainer.hyperparams", "buffer_capacity"),
    ("trainer", "min_buffer_size"): ("trainer.hyperparams", "min_buffer_size"),
    ("trainer", "target_update_every"): ("trainer.hyperparams", "target_update_every"),
    ("trainer", "steps_per_update"): ("trainer.hyperparams", "steps_per_update"),
    ("trainer", "n_envs"): ("trainer.hyperparams", "n_envs"),
    ("trainer", "loss_fn"): ("trainer.hyperparams", "loss_fn"),
    ("model", "output_rescaling"): ("model.vqc", "output_rescaling"),
    ("model", "per_layer_encoding"): ("model.vqc", "per_layer_encoding"),
    ("trainer", "eps_schedule"): ("trainer.hyperparams", "epsilon_schedule"),
    ("trainer", "eps_init"): ("trainer.hyperparams", "epsilon_start"),
    ("trainer", "eps_min"): ("trainer.hyperparams", "epsilon_end"),
    ("trainer", "eps_decay"): ("trainer.hyperparams", "epsilon_decay"),
    ("trainer", "eps_decay_steps"): ("trainer.hyperparams", "epsilon_decay_steps"),
    ("trainer", "max_steps_per_episode"): ("trainer.hyperparams", "max_steps_per_episode"),
    ("optim", "lr_variational"): ("trainer.optimizer", "lr_variational"),
    ("optim", "lr_input_scaling"): ("trainer.optimizer", "lr_lam"),
    ("optim", "lr_output_scaling"): ("trainer.optimizer", "lr_w"),
    ("optim", "lr"): ("trainer.optimizer", "lr"),
    ("optim", "amsgrad"): ("trainer.optimizer", "amsgrad"),
    ("model", "beta_init"): ("model.vqc", "beta_init"),
    ("model", "trainable_beta"): ("model.vqc", "trainable_beta"),
    ("optim", "lr_beta"): ("trainer.optimizer", "lr_beta"),
    ("trainer", "episodes_per_update"): ("trainer.hyperparams", "episodes_per_update"),
    ("trainer", "baseline"): ("trainer.hyperparams", "baseline"),
    ("trainer", "normalize_advantages"): ("trainer.hyperparams", "normalize_advantages"),
    ("trainer", "entropy_coef"): ("trainer.hyperparams", "entropy_coef"),
    ("trainer", "max_grad_norm"): ("trainer.hyperparams", "max_grad_norm"),
    ("trainer", "gae_lambda"): ("trainer.hyperparams", "gae_lambda"),
    ("trainer", "value_coef"): ("trainer.hyperparams", "value_coef"),
    ("trainer", "value_loss_fn"): ("trainer.hyperparams", "value_loss_fn"),
    ("critic", "observable"): ("model.vqc", "critic_observable"),
    ("critic", "hidden"): ("model", "critic_hidden"),
    ("critic", "w_init"): ("model.vqc", "critic_w_init"),
    ("optim", "lr_critic_variational"): ("trainer.optimizer", "lr_critic_vqc"),
    ("optim", "lr_critic_input_scaling"): ("trainer.optimizer", "lr_critic_lam"),
    ("optim", "lr_critic_output_scaling"): ("trainer.optimizer", "lr_critic_w"),
    ("optim", "lr_critic"): ("trainer.optimizer", "lr_critic"),
    ("gradient", "method"): ("trainer.optimizer", "gradient_method"),
    ("gradient", "backend"): ("trainer.optimizer", "backend"),
    ("eval", "solve_threshold"): ("trainer.hyperparams", "solve_threshold"),
    ("eval", "solve_window"): ("trainer.hyperparams", "solve_window"),
    ("eval", "env_id"): ("env", "id"),
    ("eval", "eval_every"): ("trainer.hyperparams", "eval_every"),
    ("eval", "eval_episodes"): ("trainer.hyperparams", "eval_episodes"),
    ("eval", "final_eval_episodes"): ("trainer.hyperparams", "final_eval_episodes"),
}


def _coerce_from_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy ``{section: {key: value}}`` dict into the new schema.

    The legacy ``qrl.config._SECTION_MAP`` is the single source of truth on the
    wire-format -> flat-dict mapping; this function reuses it as a target for
    the four-layer namespaces.

    Args:
        raw (dict): YAML loaded from a legacy sectioned config file.

    Returns:
        dict: A dict matching the four-layer schema (``runtime`` / ``env`` /
        ``model`` / ``trainer``).
    """
    out: dict[str, Any] = {
        "runtime": dict(asdict(RuntimeConfig())),
        "env": dict(asdict(EnvConfig())),
        "model": dict(asdict(ModelConfig())),
        "trainer": dict(asdict(TrainerConfig())),
    }
    # Ensure sub-dicts exist so the loop below can set nested keys directly.
    out["model"]["vqc"] = {}
    out["trainer"]["hyperparams"] = {}
    out["trainer"]["optimizer"] = {}

    # Legacy flat form: ``model: vqc`` (string).
    if isinstance(raw.get("model"), str):
        out["model"]["type"] = raw["model"]

    for (section, key), (namespace, target) in _LEGACY_TO_LAYERED.items():
        block = raw.get(section)
        if isinstance(block, dict) and key in block:
            if "." in namespace:
                ns, sub = namespace.split(".", 1)
                if sub not in out[ns] or not isinstance(out[ns][sub], dict):
                    out[ns][sub] = {}
                out[ns][sub][target] = block[key]
            else:
                out[namespace][target] = block[key]

    # Top-level scalars (legacy flat configs).
    if "name" in raw and isinstance(raw["name"], str):
        out["runtime"]["experiment_name"] = raw["name"]

    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str) -> ConfigSchema:
    """Load a four-layer (or legacy sectioned) YAML config.

    New four-layer files are loaded directly. Legacy sectioned files (those
    whose top level has ``model:`` / ``trainer:`` / ``optim:`` / ``gradient:`` /
    ``eval:`` / ``critic:`` rather than ``runtime:`` / ``env:`` / ``model:`` /
    ``trainer:``) are auto-converted via :func:`_coerce_from_legacy`.

    Args:
        path (str): Path to a YAML config file.

    Returns:
        ConfigSchema: The parsed and type-checked configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``model.type`` is missing or not in :data:`MODEL_ALGO`.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Detect legacy format: presence of ``optim:`` / ``gradient:`` / ``eval:``
    # at top level, or absence of any of the four new sections.
    legacy_keys = {"optim", "gradient", "eval", "critic"}
    if legacy_keys & set(raw.keys()):
        layered = _coerce_from_legacy(raw)
    else:
        layered = raw

    runtime = RuntimeConfig(
        experiment_name=layered.get("runtime", {}).get("experiment_name", ""),
        seed=layered.get("runtime", {}).get("seed", 0),
        device=layered.get("runtime", {}).get("device", "cpu"),
        precision=layered.get("runtime", {}).get("precision", "float32"),
        output_dir=layered.get("runtime", {}).get("output_dir", "result/"),
    )
    env = EnvConfig(
        id=layered.get("env", {}).get("id", "CartPole-v1"),
        params=layered.get("env", {}).get("params", {}) or {},
    )

    model_section = layered.get("model", {}) or {}
    model = ModelConfig(
        type=model_section.get("type", ""),
        encoder=model_section.get("encoder", "normalize_observation"),
        vqc=model_section.get("vqc", {}) or {},
        head=model_section.get("head", "policy"),
        hidden=model_section.get("hidden", 6),
        hidden_proj=model_section.get("hidden_proj", 256),
        output_dim=model_section.get("output_dim", 6),
    )

    trainer_section = layered.get("trainer", {}) or {}
    trainer = TrainerConfig(
        algorithm=trainer_section.get("algorithm", ""),
        hyperparams=trainer_section.get("hyperparams", {}) or {},
        optimizer=trainer_section.get("optimizer", {}) or {},
    )

    if not model.type:
        raise ValueError(
            f"config {path!r} is missing required key: model.type (model family)"
        )

    # Validate model.type against the routing table early so callers see a clear
    # error rather than a downstream KeyError.
    if model.type not in MODEL_ALGO:
        raise ValueError(
            f"config {path!r} has unknown model.type {model.type!r}; "
            f"expected one of {sorted(MODEL_ALGO)}"
        )

    # Auto-fill trainer.algorithm from model.type if absent, so configs that
    # only carry one of the two stay usable.
    if not trainer.algorithm:
        trainer.algorithm = algo_for(model.type)

    return ConfigSchema(
        runtime=runtime,
        env=env,
        model=model,
        trainer=trainer,
    )


__all__ = [
    "MODEL_ALGO",
    "algo_for",
    "RuntimeConfig",
    "EnvConfig",
    "ModelConfig",
    "TrainerConfig",
    "ConfigSchema",
    "DEFAULTS",
    "load_config",
]