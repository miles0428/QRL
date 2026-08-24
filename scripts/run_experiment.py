#!/usr/bin/env python3
"""Unified experiment entry point — single endpoint, config-driven."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import games.spin_cartpole  # noqa: F401 — registers QuantumSpinCartPole-v0
import games.dino  # noqa: F401 — registers DinoRun-v0 and DinoGameWithResNet-v0
from qrl.env_wrapper.factory import make_env
from qrl.config_schema import load_config
from qrl.config_validator import validate, ConfigValidationError
from qrl.models import build_model
from qrl.trainers import build_trainer


def run_experiment(
    config_path: str,
    seed: int = 0,
    device: str | None = None,
    output_dir: str | None = None,
    **overrides,
) -> dict:
    """Unified experiment entry point — loads config, builds model+trainer, runs training.

    This is the primary programmatic API. It returns the full results dict from
    ``trainer.train()``, making it suitable for sweep scripts and programmatic replay.

    Args:
        config_path: Path to a YAML config file (four-layer or legacy format).
        seed: Override the seed from the config.
        device: Override the device from the config (e.g. ``"cuda"``).
        output_dir: Override the output directory from the config.
        **overrides: Additional runtime overrides passed to the trainer.

    Returns:
        dict: Training results from the selected trainer.
    """
    # Validate config before loading
    try:
        validate(config_path)
    except ConfigValidationError as e:
        raise RuntimeError(f"Config validation failed: {e}") from e

    # Load config
    cfg = load_config(config_path)

    # Override runtime parameters
    if seed is not None:
        cfg.runtime.seed = seed
    if device is not None:
        cfg.runtime.device = device
    if output_dir is not None:
        cfg.runtime.output_dir = output_dir

    # Build environment (entity-decoupled: via factory, not direct games import)
    env = make_env(
        cfg.env.id,
        env_params=cfg.env.params,
        seed=cfg.runtime.seed,
    )

    # Build model (convert dataclass ModelConfig to dict for legacy dict-based API)
    from dataclasses import asdict
    model = build_model(
        asdict(cfg.model),
        obs_space=env.observation_space,
        action_space=env.action_space,
    )

    # Build trainer (routed via algo_for)
    trainer = build_trainer(
        cfg.trainer,
        model=model,
        env=env,
        runtime=cfg.runtime,
        env_id=cfg.env.id,
    )

    # Train
    results = trainer.train(**overrides)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="QRL unified experiment entry point")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed from config"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device from config"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output_dir from config"
    )
    args = parser.parse_args()

    try:
        results = run_experiment(
            config_path=args.config,
            seed=args.seed,
            device=args.device,
            output_dir=args.output_dir,
        )
        print(f"Training complete: {results}")
        return 0
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
