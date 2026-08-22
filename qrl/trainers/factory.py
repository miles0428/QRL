"""Trainer factory — dispatches to the correct algorithm-specific training loop."""

from __future__ import annotations

from qrl.config_schema import ConfigSchema, TrainerConfig, algo_for
from qrl.trainers.dqn_trainer import train as train_dqn
from qrl.trainers.a2c_trainer import train_a2c
from qrl.trainers.pg_trainer import train_pg
import inspect

# Parameter whitelist per trainer — keys not in this set are silently dropped
# to tolerate extra hyperparams declared in YAML configs.
_TRAIN_PARAM_BLACKLIST = {
    "gradient_method",  # declared in optimizer block but handled by the trainer itself
}

def _filter_params(train_fn, hyperparams):
    """Drop keys not accepted by ``train_fn`` to prevent TypeError."""
    sig = inspect.signature(train_fn)
    accepted = set(sig.parameters) - _TRAIN_PARAM_BLACKLIST
    return {k: v for k, v in hyperparams.items() if k in accepted}


def build_trainer(
    cfg: TrainerConfig,
    model,
    env,
    runtime,
    **kwargs,
) -> object:
    """Build and return the correct trainer instance based on ``cfg.algorithm``.

    Routing is derived from the trainer's own ``algorithm`` field (which is
    auto-filled from ``model.type`` via :func:`qrl.config_schema.algo_for` when
    the YAML config is loaded).

    Args:
        cfg: The ``trainer`` namespace from :class:`ConfigSchema`.
        model: The model (or tuple of actor/critic for A2C) returned by
            :func:`qrl.models.build_model`.
        env: The gymnasium environment (or vector env) created by
            :func:`qrl.env_wrapper.factory.make_env`.
        runtime: The ``runtime`` namespace from :class:`ConfigSchema`, used
            for ``seed``, ``device``, ``output_dir``, and trainer-level defaults.

    Returns:
        A trainer object with a ``.train()`` method that runs the training loop
        and returns a summary dict.

    Raises:
        ValueError: If ``cfg.algorithm`` is not one of ``"dqn"``, ``"pg"``,
            ``"a2c"``.
    """
    algo = cfg.algorithm
    hyperparams = cfg.hyperparams or {}
    optimizer_cfg = cfg.optimizer or {}

    # env_id lives in cfg.env.id, which is carried through ConfigSchema
    # We receive it as a parameter so the factory stays decoupled from env config
    env_id = kwargs.pop("env_id", "CartPole-v1")

    if algo == "dqn":
        # DQN trainer receives a single Q-network and a flat hyperparams dict.
        return _build_dqn_trainer(model, hyperparams, optimizer_cfg, runtime, env_id)
    elif algo == "pg":
        # PG trainer receives a single policy network.
        return _build_pg_trainer(model, hyperparams, optimizer_cfg, runtime, env_id)
    elif algo == "a2c":
        # A2C trainer receives (actor, critic) tuple and optimizer with two
        # parameter groups.
        return _build_a2c_trainer(model, hyperparams, optimizer_cfg, runtime, env_id)
    else:
        raise ValueError(
            f"Unsupported algorithm: {algo!r}; expected one of 'dqn', 'pg', 'a2c'"
        )


# ---------------------------------------------------------------------------
# Private builders — these construct the concrete trainer callables
# ---------------------------------------------------------------------------


def _build_dqn_trainer(model, hyperparams, optimizer_cfg, runtime, env_id):
    """Construct and return a DQN trainer callable."""
    import torch

    # Assemble optimizer kwargs from optimizer block
    optim_kwargs = {
        "params": model.parameters(),
        "lr": optimizer_cfg.get("lr", 0.01),
    }
    if optimizer_cfg.get("amsgrad", False):
        optim_kwargs["amsgrad"] = True
    optimizer = torch.optim.Adam(**optim_kwargs)

    # Results path
    results_path, checkpoint_path = _results_path(runtime, "dqn")

    return _DQNTrainerWrapper(
        model=model,
        optimizer=optimizer,
        results_path=results_path,
        checkpoint_path=checkpoint_path,
        seed=runtime.seed,
        env_id=env_id,
        **hyperparams,
    )


def _build_pg_trainer(model, hyperparams, optimizer_cfg, runtime, env_id):
    """Construct and return a PG (REINFORCE) trainer callable."""
    import torch

    optim_kwargs = {
        "params": model.parameters(),
        "lr": optimizer_cfg.get("lr", 0.01),
    }
    if optimizer_cfg.get("amsgrad", False):
        optim_kwargs["amsgrad"] = True
    optimizer = torch.optim.Adam(**optim_kwargs)

    results_path, checkpoint_path = _results_path(runtime, "pg")

    return _PGTrainerWrapper(
        model=model,
        optimizer=optimizer,
        results_path=results_path,
        checkpoint_path=checkpoint_path,
        seed=runtime.seed,
        env_id=env_id,
        **hyperparams,
    )


def _build_a2c_trainer(model, hyperparams, optimizer_cfg, runtime, env_id):
    """Construct and return an A2C trainer callable.

    ``model`` is expected to be a (actor, critic) tuple from
    :func:`qrl.models.build_model`.
    """
    import torch

    actor, critic = model

    # Two parameter groups with potentially different learning rates
    param_groups = [
        {"params": actor.parameters(), "lr": optimizer_cfg.get("lr", 0.01)},
        {
            "params": critic.parameters(),
            "lr": optimizer_cfg.get("lr_critic", 0.05),
        },
    ]
    optimizer = torch.optim.Adam(param_groups, amsgrad=optimizer_cfg.get("amsgrad", True))

    results_path, checkpoint_path = _results_path(runtime, "a2c")

    return _A2CTrainerWrapper(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        results_path=results_path,
        checkpoint_path=checkpoint_path,
        seed=runtime.seed,
        env_id=env_id,
        **hyperparams,
    )


def _results_path(runtime, algo: str) -> tuple[str, str]:
    """Build CSV and model-checkpoint paths under ``runtime.output_dir``.

    Directory structure: ``result/{experiment_name}/{timestamp}/``

    Returns:
        A 2-tuple of (results_csv_path, checkpoint_path).
    """
    import os
    from datetime import datetime

    output_dir = getattr(runtime, "output_dir", "result/")
    name = getattr(runtime, "experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.csv")
    checkpoint_path = os.path.join(run_dir, "model.pt")
    return results_path, checkpoint_path


# ---------------------------------------------------------------------------
# Thin wrappers so the same ``.train()`` interface works for all three
# ---------------------------------------------------------------------------


class _DQNTrainerWrapper:
    """Wrapper presenting a consistent ``.train()`` interface for DQN."""

    def __init__(self, model, optimizer, results_path, checkpoint_path, seed, env_id, **hyperparams):
        self._model = model
        self._optimizer = optimizer
        self._results_path = results_path
        self._checkpoint_path = checkpoint_path
        self._seed = seed
        self._env_id = env_id
        self._hyperparams = hyperparams

    def train(self):
        filtered = _filter_params(train_dqn, self._hyperparams)
        return train_dqn(
            model=self._model,
            optimizer=self._optimizer,
            results_path=self._results_path,
            checkpoint_path=self._checkpoint_path,
            seed=self._seed,
            env_id=self._env_id,
            **filtered,
        )


class _PGTrainerWrapper:
    """Wrapper presenting a consistent ``.train()`` interface for PG."""

    def __init__(self, model, optimizer, results_path, checkpoint_path, seed, env_id, **hyperparams):
        self._model = model
        self._optimizer = optimizer
        self._results_path = results_path
        self._checkpoint_path = checkpoint_path
        self._seed = seed
        self._env_id = env_id
        self._hyperparams = hyperparams

    def train(self):
        filtered = _filter_params(train_pg, self._hyperparams)
        return train_pg(
            model=self._model,
            optimizer=self._optimizer,
            results_path=self._results_path,
            checkpoint_path=self._checkpoint_path,
            seed=self._seed,
            env_id=self._env_id,
            **filtered,
        )


class _A2CTrainerWrapper:
    """Wrapper presenting a consistent ``.train()`` interface for A2C."""

    def __init__(self, actor, critic, optimizer, results_path, checkpoint_path, seed, env_id, **hyperparams):
        self._actor = actor
        self._critic = critic
        self._optimizer = optimizer
        self._results_path = results_path
        self._checkpoint_path = checkpoint_path
        self._seed = seed
        self._env_id = env_id
        self._hyperparams = hyperparams

    def train(self):
        filtered = _filter_params(train_a2c, self._hyperparams)
        return train_a2c(
            actor=self._actor,
            critic=self._critic,
            optimizer=self._optimizer,
            results_path=self._results_path,
            checkpoint_path=self._checkpoint_path,
            seed=self._seed,
            env_id=self._env_id,
            **filtered,
        )
