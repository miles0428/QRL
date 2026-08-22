"""YAML config validator for the four-layer QRL schema.

Validates that a YAML file conforms to BLUEPRINT §6.2:

* Top-level keys are EXACTLY ``runtime``, ``env``, ``model``, ``trainer``.
* Each namespace exposes the documented required fields.
* Known enum-like fields carry an allowed value.

This validator is intentionally string-and-dataclass-free at the validation
core -- it works on the parsed dict so authors can lint configs without
importing pydantic or anything heavier than ``yaml``.

Run directly:
    python -c "from qrl.config_validator import validate; validate('configs/qa2c.yaml')"
"""

from __future__ import annotations

from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Required structure (mirrors BLUEPRINT §6.2)
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = {"runtime", "env", "model", "trainer"}

RUNTIME_FIELDS = {"experiment_name", "seed", "device", "precision", "output_dir"}
ENV_FIELDS = {"id", "params"}
MODEL_FIELDS = {"encoder", "vqc", "head"}
TRAINER_FIELDS = {"algorithm", "hyperparams", "optimizer"}

# Optional nested fields that may or may not be present depending on the model
# family. They are documented but not enforced as required by this validator.
OPTIONAL_MODEL_VQC_FIELDS = {
    "n_qubits", "n_layers", "reuploading", "observables",
    "per_layer_encoding", "output_rescaling",
    "beta_init", "trainable_beta", "critic_observable", "critic_w_init",
}
OPTIONAL_TRAINER_HYPERPARAM_FIELDS = {
    "max_episodes", "batch_size", "auto_batch_size", "gamma",
    "buffer_capacity", "min_buffer_size", "target_update_every",
    "steps_per_update", "n_envs", "loss_fn",
    "epsilon_schedule", "epsilon_start", "epsilon_end", "epsilon_decay",
    "epsilon_decay_steps", "max_steps_per_episode",
    "episodes_per_update", "baseline", "normalize_advantages",
    "entropy_coef", "max_grad_norm",
    "gae_lambda", "value_coef", "value_loss_fn",
    "solve_threshold", "solve_window",
    "eval_every", "eval_episodes", "final_eval_episodes",
}
OPTIONAL_TRAINER_OPTIMIZER_FIELDS = {
    "lr", "lr_variational", "lr_lam", "lr_w", "lr_beta", "amsgrad",
    "lr_critic", "lr_critic_vqc", "lr_critic_lam", "lr_critic_w",
    "backend", "gradient_method",
}

# Enum-like fields and their allowed values. Out-of-range values fail validation.
ENUM_FIELDS: dict[tuple[str, str], set[str]] = {
    ("trainer.hyperparams", "epsilon_schedule"): {"exponential_episodes", "linear_steps"},
    ("trainer.hyperparams", "baseline"): {"batch_mean", "none"},
    ("trainer.hyperparams", "loss_fn"): {"huber", "mse"},
    ("trainer.hyperparams", "value_loss_fn"): {"huber", "mse"},
    ("trainer.optimizer", "backend"): {"torch_sv", "qtm", "qiskit_ml"},
    ("trainer.optimizer", "gradient_method"): {"adjoint", "spsa", "param_shift", "lin_comb"},
    ("model", "head"): {"policy", "qfunction", "value"},
    ("model", "encoder"): {"normalize_observation", "cnn", "transformer", "mlp"},
    ("runtime", "device"): {"cpu", "cuda", "cuda:0", "mps"},
    ("runtime", "precision"): {"float32", "float64"},
}


class ConfigValidationError(ValueError):
    """Raised when a YAML config fails four-layer schema validation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _missing_fields(section: dict[str, Any], required: set[str]) -> set[str]:
    """Return the set of required keys that are absent from ``section``."""
    return {key for key in required if key not in section}


def _validate_enum(section_path: str, key: str, value: Any) -> None:
    """If ``(section_path, key)`` is an enum-like field, check ``value`` is allowed."""
    allowed = ENUM_FIELDS.get((section_path, key))
    if allowed is None or value is None:
        return
    if value not in allowed:
        raise ConfigValidationError(
            f"invalid value for {section_path}.{key}: {value!r}; "
            f"expected one of {sorted(allowed)}"
        )


def _validate_section(
    section_path: str,
    section: Any,
    required: set[str],
    optional_allowed: set[str],
) -> None:
    """Validate one namespace against its declared required/optional fields.

    Args:
        section_path (str): Dotted path used in error messages, e.g.
            ``"model"`` or ``"model.vqc"``.
        section (Any): The candidate section (must be a mapping).
        required (set[str]): Field names that MUST be present.
        optional_allowed (set[str]): Field names that MAY be present (for
            documentation only -- extra keys are allowed but flagged).

    Raises:
        ConfigValidationError: If the section is missing, not a mapping, or
            missing any required field.
    """
    if not isinstance(section, dict):
        raise ConfigValidationError(
            f"section {section_path!r} must be a mapping; got {type(section).__name__}"
        )

    missing = _missing_fields(section, required)
    if missing:
        raise ConfigValidationError(
            f"section {section_path!r} is missing required field(s): "
            f"{', '.join(sorted(missing))}"
        )

    # Enum check on declared fields (only runs if the value is present).
    for key in section.keys() & required:
        _validate_enum(section_path, key, section[key])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(path: str) -> bool:
    """Validate a YAML config against the four-layer schema.

    Args:
        path (str): Path to a YAML config file.

    Returns:
        bool: True if the config is valid.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigValidationError: If the file is missing required sections or
            fields, or if an enum-like field carries an invalid value.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"top-level of {path!r} must be a mapping; got {type(raw).__name__}"
        )

    present_sections = set(raw.keys())
    if present_sections != REQUIRED_SECTIONS:
        # Distinguish missing vs extra to give a useful error message.
        missing_sections = REQUIRED_SECTIONS - present_sections
        extra_sections = present_sections - REQUIRED_SECTIONS
        problems: list[str] = []
        if missing_sections:
            problems.append(f"missing required section(s): {sorted(missing_sections)}")
        if extra_sections:
            problems.append(f"unexpected section(s): {sorted(extra_sections)}")
        raise ConfigValidationError(
            f"{path!r} does not match the four-layer schema -- "
            f"{'; '.join(problems)}"
        )

    # runtime
    _validate_section("runtime", raw["runtime"], RUNTIME_FIELDS, set())
    _validate_enum("runtime", "device", raw["runtime"].get("device"))
    _validate_enum("runtime", "precision", raw["runtime"].get("precision"))

    # env
    _validate_section("env", raw["env"], ENV_FIELDS, set())

    # model (and model.vqc, if present)
    _validate_section("model", raw["model"], MODEL_FIELDS, OPTIONAL_MODEL_VQC_FIELDS)
    _validate_enum("model", "head", raw["model"].get("head"))
    _validate_enum("model", "encoder", raw["model"].get("encoder"))
    if "vqc" in raw["model"]:
        _validate_section(
            "model.vqc", raw["model"]["vqc"], set(), OPTIONAL_MODEL_VQC_FIELDS
        )

    # trainer (and trainer.hyperparams / trainer.optimizer)
    _validate_section(
        "trainer", raw["trainer"], TRAINER_FIELDS, set()
    )
    if "hyperparams" in raw["trainer"]:
        _validate_section(
            "trainer.hyperparams",
            raw["trainer"]["hyperparams"],
            set(),
            OPTIONAL_TRAINER_HYPERPARAM_FIELDS,
        )
        for key in raw["trainer"]["hyperparams"].keys():
            _validate_enum("trainer.hyperparams", key, raw["trainer"]["hyperparams"][key])
    if "optimizer" in raw["trainer"]:
        _validate_section(
            "trainer.optimizer",
            raw["trainer"]["optimizer"],
            set(),
            OPTIONAL_TRAINER_OPTIMIZER_FIELDS,
        )
        for key in raw["trainer"]["optimizer"].keys():
            _validate_enum("trainer.optimizer", key, raw["trainer"]["optimizer"][key])

    # model.type is required because it is the routing key into MODEL_ALGO.
    # The schema above doesn't list ``type`` as a top-level model field because
    # BLUEPRINT §6.1 expresses the model namespace as ``encoder/vqc/head``;
    # but we still need it for ``algo_for()`` and downstream trainers.
    if "type" not in raw["model"]:
        raise ConfigValidationError(
            f"{path!r} is missing required field: model.type "
            f"(the model family identifier used for trainer routing)"
        )

    return True


def validate_all(paths: list[str]) -> dict[str, bool]:
    """Validate a list of config paths.

    Args:
        paths (list[str]): Paths to YAML config files.

    Returns:
        dict[str, bool]: Mapping of path -> success. Raises
        :class:`ConfigValidationError` on the first failure (does not stop
        on the first error -- a single attempt reports it).
    """
    results: dict[str, bool] = {}
    for p in paths:
        try:
            validate(p)
            results[p] = True
        except ConfigValidationError:
            results[p] = False
            raise
    return results


__all__ = [
    "REQUIRED_SECTIONS",
    "RUNTIME_FIELDS",
    "ENV_FIELDS",
    "MODEL_FIELDS",
    "TRAINER_FIELDS",
    "ConfigValidationError",
    "validate",
    "validate_all",
]