"""Config loading and normalization.

v3 moved `configs/*.yaml` to the sectioned layout the brief specifies
(`model:` / `trainer:` / `optim:` / `gradient:` / `eval:`). Everything
downstream -- `scripts/train.py`, `scripts/sweep_seeds.py`,
`scripts/benchmark_backends.py` -- consumes a single *flat* dict instead of
reaching into sections, so there is exactly one place that knows the file
layout: `normalize_config()` below.

Pre-v3 flat configs still load unchanged; keys found at top level are used
as-is when the corresponding section is absent. That is deliberate rather than
merely tidy -- it means an old config file and an old checkpoint can still be
re-run for comparison without editing them.

This module must not import qiskit or any quantum framework: it is imported by
the trainer-side scripts, and the "trainer.py never imports qiskit" rule is
only enforceable if its dependencies obey it too.
"""

from __future__ import annotations

from typing import Any

import yaml

# nested location -> flat key.
# 2-level: (section, key_in_section) -> flat_name
# 3-level: (section, subsection, key) -> flat_name  (e.g. trainer.hyperparams.*)
_SECTION_MAP: dict[tuple[str, ...], str] = {
    # model (2-level)
    ("model", "type"): "model_type",
    ("model", "n_qubits"): "n_qubits",
    ("model", "n_layers"): "n_layers",
    ("model", "reuploading"): "reuploading",
    ("model", "observables"): "observables",
    ("model", "hidden"): "hidden",
    ("model", "output_rescaling"): "output_rescaling",
    ("model", "per_layer_encoding"): "per_layer_encoding",
    ("model", "beta_init"): "beta_init",
    ("model", "trainable_beta"): "trainable_beta",
    ("model", "critic_observable"): "critic_observable",
    # model.vqc (3-level)
    ("model", "vqc", "n_qubits"): "n_qubits",
    ("model", "vqc", "n_layers"): "n_layers",
    ("model", "vqc", "reuploading"): "reuploading",
    ("model", "vqc", "observables"): "observables",
    ("model", "vqc", "output_rescaling"): "output_rescaling",
    ("model", "vqc", "beta_init"): "beta_init",
    ("model", "vqc", "trainable_beta"): "trainable_beta",
    ("model", "vqc", "critic_observable"): "critic_observable",
    # trainer (2-level) — legacy flat or optimizer sub-section
    ("trainer", "algorithm"): "algorithm",
    ("trainer", "episodes"): "max_episodes",
    ("trainer", "batch_size"): "batch_size",
    ("trainer", "auto_batch_size"): "auto_batch_size",
    ("trainer", "gamma"): "gamma",
    ("trainer", "buffer_capacity"): "buffer_capacity",
    ("trainer", "min_buffer_size"): "min_buffer_size",
    ("trainer", "target_update_every"): "target_update_every",
    ("trainer", "steps_per_update"): "steps_per_update",
    ("trainer", "n_envs"): "n_envs",
    ("trainer", "loss_fn"): "loss_fn",
    ("trainer", "eps_schedule"): "epsilon_schedule",
    ("trainer", "eps_init"): "epsilon_start",
    ("trainer", "eps_min"): "epsilon_end",
    ("trainer", "eps_decay"): "epsilon_decay",
    ("trainer", "eps_decay_steps"): "epsilon_decay_steps",
    ("trainer", "max_steps_per_episode"): "max_steps_per_episode",
    ("trainer", "episodes_per_update"): "episodes_per_update",
    ("trainer", "baseline"): "baseline",
    ("trainer", "normalize_advantages"): "normalize_advantages",
    ("trainer", "entropy_coef"): "entropy_coef",
    ("trainer", "max_grad_norm"): "max_grad_norm",
    ("trainer", "gae_lambda"): "gae_lambda",
    ("trainer", "value_coef"): "value_coef",
    ("trainer", "value_loss_fn"): "value_loss_fn",
    # trainer.hyperparams (3-level)
    ("trainer", "hyperparams", "max_episodes"): "max_episodes",
    ("trainer", "hyperparams", "batch_size"): "batch_size",
    ("trainer", "hyperparams", "auto_batch_size"): "auto_batch_size",
    ("trainer", "hyperparams", "gamma"): "gamma",
    ("trainer", "hyperparams", "buffer_capacity"): "buffer_capacity",
    ("trainer", "hyperparams", "min_buffer_size"): "min_buffer_size",
    ("trainer", "hyperparams", "target_update_every"): "target_update_every",
    ("trainer", "hyperparams", "steps_per_update"): "steps_per_update",
    ("trainer", "hyperparams", "n_envs"): "n_envs",
    ("trainer", "hyperparams", "loss_fn"): "loss_fn",
    ("trainer", "hyperparams", "epsilon_schedule"): "epsilon_schedule",
    ("trainer", "hyperparams", "epsilon_start"): "epsilon_start",
    ("trainer", "hyperparams", "epsilon_end"): "epsilon_end",
    ("trainer", "hyperparams", "epsilon_decay"): "epsilon_decay",
    ("trainer", "hyperparams", "epsilon_decay_steps"): "epsilon_decay_steps",
    ("trainer", "hyperparams", "max_steps_per_episode"): "max_steps_per_episode",
    ("trainer", "hyperparams", "episodes_per_update"): "episodes_per_update",
    ("trainer", "hyperparams", "baseline"): "baseline",
    ("trainer", "hyperparams", "normalize_advantages"): "normalize_advantages",
    ("trainer", "hyperparams", "entropy_coef"): "entropy_coef",
    ("trainer", "hyperparams", "max_grad_norm"): "max_grad_norm",
    ("trainer", "hyperparams", "gae_lambda"): "gae_lambda",
    ("trainer", "hyperparams", "value_coef"): "value_coef",
    ("trainer", "hyperparams", "value_loss_fn"): "value_loss_fn",
    ("trainer", "hyperparams", "eval_every"): "eval_every",
    ("trainer", "hyperparams", "eval_episodes"): "eval_episodes",
    ("trainer", "hyperparams", "final_eval_episodes"): "final_eval_episodes",
    ("trainer", "hyperparams", "solve_threshold"): "solve_threshold",
    ("trainer", "hyperparams", "solve_window"): "solve_window",
    # trainer.optimizer (3-level)
    ("trainer", "optimizer", "lr"): "lr",
    ("trainer", "optimizer", "lr_variational"): "lr_vqc",
    ("trainer", "optimizer", "lr_input_scaling"): "lr_lam",
    ("trainer", "optimizer", "lr_output_scaling"): "lr_w",
    ("trainer", "optimizer", "lr_beta"): "lr_beta",
    ("trainer", "optimizer", "lr_critic"): "lr_critic",
    ("trainer", "optimizer", "lr_critic_variational"): "lr_critic_vqc",
    ("trainer", "optimizer", "lr_critic_input_scaling"): "lr_critic_lam",
    ("trainer", "optimizer", "lr_critic_output_scaling"): "lr_critic_w",
    ("trainer", "optimizer", "amsgrad"): "amsgrad",
    ("trainer", "optimizer", "gradient_method"): "gradient_method",
    ("trainer", "optimizer", "backend"): "backend",
    # optim (2-level — legacy flat)
    ("optim", "lr_variational"): "lr_vqc",
    ("optim", "lr_input_scaling"): "lr_lam",
    ("optim", "lr_output_scaling"): "lr_w",
    ("optim", "lr"): "lr",
    ("optim", "amsgrad"): "amsgrad",
    ("optim", "lr_beta"): "lr_beta",
    ("optim", "lr_critic_variational"): "lr_critic_vqc",
    ("optim", "lr_critic_input_scaling"): "lr_critic_lam",
    ("optim", "lr_critic_output_scaling"): "lr_critic_w",
    ("optim", "lr_critic"): "lr_critic",
    # gradient (2-level)
    ("gradient", "method"): "gradient_method",
    ("gradient", "backend"): "backend",
    # eval (2-level)
    ("eval", "solve_threshold"): "solve_threshold",
    ("eval", "solve_window"): "solve_window",
    ("eval", "env_id"): "env_id",
    ("eval", "eval_every"): "eval_every",
    ("eval", "eval_episodes"): "eval_episodes",
    ("eval", "final_eval_episodes"): "final_eval_episodes",
    # critic (2-level)
    ("critic", "observable"): "critic_observable",
    ("critic", "hidden"): "critic_hidden",
    ("critic", "w_init"): "critic_w_init",
    # runtime (3-level)
    ("runtime", "experiment_name"): "name",
    # env (2-level)
    ("env", "id"): "env",
}

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
    # Greedy (epsilon=0) evaluation. The training reward columns are measured
    # under exploration and understate the policy; see src/trainer.py::train.
    # eval_every=0 disables the periodic curve entirely.
    "eval_every": 50,
    "eval_episodes": 5,
    # The periodic 5-episode sample is too noisy to *claim* a solve; this is the
    # one-off run at the end that the solve claim is made from.
    "final_eval_episodes": 100,
    # Epsilon: "exponential_episodes" is the Skolik reference schedule
    # (eps *= decay once per episode); "linear_steps" is the pre-v3 schedule
    # (linear in environment steps). See src/trainer.py::epsilon_at.
    "epsilon_schedule": "exponential_episodes",
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.99,
    "epsilon_decay_steps": 20_000,
    # Quantum-only; ignored by the MLP path.
    # Reference-implementation values (TFQ tutorial / Skolik et al.), adopted
    # after the 10-minute convergence requirement landed. steps_per_update is
    # also the single largest wall-clock lever: a gradient step costs far more
    # than an environment step when the Q-function is a circuit.
    "amsgrad": True,
    "n_envs": 1,
    "steps_per_update": 10,
    "loss_fn": "huber",
    "output_rescaling": True,
    "per_layer_encoding": False,
    "reuploading": True,
    "observables": ["ZZII", "IIZZ"],
    # v3 training backend. Not qtm: see the backend note in configs/qdqn.yaml.
    "backend": "torch_sv",
    "gradient_method": "adjoint",
    # --- policy gradient (model_type vqc_policy | mlp_policy) ----------------
    # Unused by the DQN path, and vice versa: a config declares one model_type,
    # and the keys the other algorithm needs simply sit at their defaults. They
    # live in the same DEFAULTS dict so that normalize_config() stays the single
    # place that knows the file layout.
    "beta_init": 1.0,
    "trainable_beta": True,
    "lr_beta": 0.1,  # like lr_w on the DQN side: the head must move faster than the circuit
    "episodes_per_update": 8,
    "baseline": "batch_mean",  # "batch_mean" | "none"
    "normalize_advantages": True,
    "entropy_coef": 0.0,
    "max_grad_norm": None,
    # --- A2C (model_type vqc_a2c | mlp_a2c) ----------------------------------
    "gae_lambda": 0.95,
    "value_coef": 0.5,
    "value_loss_fn": "huber",
    "critic_observable": "ZZZZ",
    "critic_hidden": 7,  # one output unit, so wider than the actor -- see mlp.py
    # None means derive it: (1-gamma^T)/(1-gamma), the largest return the
    # discounting admits. A critic head starting at 1 cannot reach V* inside an
    # A2C run's ~60 gradient steps. See src/models/vqc_value.py.
    "critic_w_init": None,
    "lr_critic_vqc": 0.01,
    "lr_critic_lam": 0.01,
    "lr_critic_w": 0.1,  # must climb from 1 to ~V* = 99, exactly as the DQN head does
    "lr_critic": 0.05,   # classical critic, single group
}

REQUIRED = ("name", "model_type", "max_episodes")

# Which trainer a model_type belongs to. scripts/train.py and
# scripts/train_pg.py each check this rather than silently running a policy
# through the TD loss (which would "work" -- softmax logits are shaped exactly
# like Q-values -- and produce a plausible, meaningless curve).
MODEL_ALGO = {
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


def algo_for(config: dict) -> str:
    model_type = config["model_type"]
    if model_type not in MODEL_ALGO:
        raise ValueError(
            f"unknown model type {model_type!r}; expected one of {sorted(MODEL_ALGO)}"
        )
    return MODEL_ALGO[model_type]


def normalize_config(raw: dict) -> dict:
    """Flatten a sectioned (or legacy flat) config into one dict, with defaults."""
    flat: dict[str, Any] = dict(DEFAULTS)
    from_section: set[str] = set()

    # Legacy flat form: `model: vqc` is a string rather than a section.
    if isinstance(raw.get("model"), str):
        flat["model_type"] = raw["model"]
        from_section.add("model_type")

    for (section, *rest), flat_key in _SECTION_MAP.items():
        block = raw.get(section)
        if not isinstance(block, dict):
            continue
        if len(rest) == 1:
            # 2-level: (section, key) -> flat_key
            key = rest[0]
            if key in block:
                flat[flat_key] = block[key]
                from_section.add(flat_key)
        elif len(rest) == 2:
            # 3-level: (section, subsection, key) -> flat_key
            sub_key, key = rest
            sub_block = block.get(sub_key)
            if isinstance(sub_block, dict) and key in sub_block:
                flat[flat_key] = sub_block[key]
                from_section.add(flat_key)

    # Legacy top-level scalars fill in only where no section supplied the value,
    # so a sectioned file is never silently overridden by a stale top-level key.
    for key, value in raw.items():
        if isinstance(value, dict) or key == "model":
            continue
        if key not in from_section:
            flat[key] = value

    missing = [k for k in REQUIRED if flat.get(k) is None]
    if missing:
        raise ValueError(f"config is missing required key(s): {', '.join(missing)}")
    if flat["epsilon_schedule"] not in ("exponential_episodes", "linear_steps"):
        raise ValueError(
            f"unknown eps_schedule {flat['epsilon_schedule']!r}; "
            f"expected 'exponential_episodes' or 'linear_steps'"
        )
    if flat["baseline"] not in ("batch_mean", "none"):
        raise ValueError(
            f"unknown baseline {flat['baseline']!r}; expected 'batch_mean' or 'none'"
        )
    if flat["model_type"] not in MODEL_ALGO:
        raise ValueError(
            f"unknown model type {flat['model_type']!r}; expected one of {sorted(MODEL_ALGO)}"
        )
    if flat.get("observables") is not None:
        flat["observables"] = list(flat["observables"])
    # Convenience alias so callers can use cfg["type"] instead of cfg["model_type"].
    flat["type"] = flat["model_type"]
    return flat


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return normalize_config(raw)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:] or ["configs/qdqn.yaml", "configs/mlp_baseline.yaml"]:
        cfg = load_config(p)
        print(f"=== {p} ===")
        for k in sorted(cfg):
            print(f"  {k}: {cfg[k]}")
