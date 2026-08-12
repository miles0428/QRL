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

# nested location -> flat key. (section, key_in_section) -> flat_name
_SECTION_MAP: dict[tuple[str, str], str] = {
    ("model", "type"): "model_type",
    ("model", "n_qubits"): "n_qubits",
    ("model", "n_layers"): "n_layers",
    ("model", "reuploading"): "reuploading",
    ("model", "observables"): "observables",
    ("model", "hidden"): "hidden",
    ("trainer", "episodes"): "max_episodes",
    ("trainer", "batch_size"): "batch_size",
    ("trainer", "auto_batch_size"): "auto_batch_size",
    ("trainer", "gamma"): "gamma",
    ("trainer", "buffer_capacity"): "buffer_capacity",
    ("trainer", "min_buffer_size"): "min_buffer_size",
    ("trainer", "target_update_every"): "target_update_every",
    ("trainer", "eps_schedule"): "epsilon_schedule",
    ("trainer", "eps_init"): "epsilon_start",
    ("trainer", "eps_min"): "epsilon_end",
    ("trainer", "eps_decay"): "epsilon_decay",
    ("trainer", "eps_decay_steps"): "epsilon_decay_steps",
    ("trainer", "max_steps_per_episode"): "max_steps_per_episode",
    ("optim", "lr_variational"): "lr_vqc",
    ("optim", "lr_input_scaling"): "lr_lam",
    ("optim", "lr_output_scaling"): "lr_w",
    ("optim", "lr"): "lr",
    ("gradient", "method"): "gradient_method",
    ("gradient", "backend"): "backend",
    ("eval", "solve_threshold"): "solve_threshold",
    ("eval", "solve_window"): "solve_window",
    ("eval", "env_id"): "env_id",
    ("eval", "eval_every"): "eval_every",
    ("eval", "eval_episodes"): "eval_episodes",
    ("eval", "final_eval_episodes"): "final_eval_episodes",
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
    "reuploading": True,
    "observables": ["ZZII", "IIZZ"],
    # v3 training backend. Not qtm: see the backend note in configs/qdqn.yaml.
    "backend": "torch_sv",
    "gradient_method": "adjoint",
}

REQUIRED = ("name", "model_type", "max_episodes")


def normalize_config(raw: dict) -> dict:
    """Flatten a sectioned (or legacy flat) config into one dict, with defaults."""
    flat: dict[str, Any] = dict(DEFAULTS)
    from_section: set[str] = set()

    # Legacy flat form: `model: vqc` is a string rather than a section.
    if isinstance(raw.get("model"), str):
        flat["model_type"] = raw["model"]
        from_section.add("model_type")

    for (section, key), flat_key in _SECTION_MAP.items():
        block = raw.get(section)
        if isinstance(block, dict) and key in block:
            flat[flat_key] = block[key]
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
    if flat.get("observables") is not None:
        flat["observables"] = list(flat["observables"])
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
