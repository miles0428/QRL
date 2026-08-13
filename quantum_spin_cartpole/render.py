"""
Interactive trajectory visualisation script.

Usage
-----
    python -m quantum_spin_cartpole.render [--action {0,2,4}]
                                           [--terminate-on-violation]
                                           [--seed SEED]
                                           [--max-steps N]
                                           [--output-dir DIR]

Arguments
---------
--action : int
    Fixed action to drive the trajectory:
        0 → +X (RX),  2 → +Y (RY),  4 → IDLE.
    Default: 0.
--terminate-on-violation : bool
    Enable termination on sz < 0.  If omitted, the spin is allowed
    to cross into the south hemisphere and the episode runs to max_steps.
--seed : int
    Random seed.  Default: 42.
--max-steps : int
    Maximum steps per episode.  Default: 20 000.
--output-dir : str
    Directory to save output figures.  Default: ./trajectory_output/
"""
from __future__ import annotations

import argparse
import os
import sys

# Add package to path when run as a script
sys.path.insert(0, os.path.dirname(__file__))

from quantum_spin_cartpole import QuantumSpinCartPoleEnv
from quantum_spin_cartpole.trajectory import SpinTrajectory


# Action label map
ACTION_LABELS = {
    0: "RX (+X)",
    1: "RX (-X)",
    2: "RY (+Y)",
    3: "RY (-Y)",
    4: "IDLE",
}


def run_single_strategy(
    env: QuantumSpinCartPoleEnv,
    action: int,
    terminate_on_violation: bool,
    max_steps: int,
    output_dir: str,
) -> dict:
    """
    Run one strategy and produce 3-D Bloch + sigma-time plots.

    Returns
    -------
    dict
        Summary dict from ``SpinTrajectory.summary()``.
    """
    traj = SpinTrajectory(env, max_steps=max_steps)
    result = traj.run(action=action, terminate_on_violation=terminate_on_violation)

    label = ACTION_LABELS.get(action, f"action-{action}")
    prefix = f"action{action}_{'term' if terminate_on_violation else 'noterm'}"

    bloch_path = os.path.join(output_dir, f"{prefix}_bloch3d.png")
    sigma_path = os.path.join(output_dir, f"{prefix}_sigma_t.png")

    traj.plot_3d(bloch_path)
    traj.plot_xy_components(sigma_path)

    print(f"[{label}] steps={result['steps']}, output={bloch_path},{sigma_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spin trajectory visualizer"
    )
    parser.add_argument(
        "--action",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Fixed action (default: 0 = +X/RX)",
    )
    parser.add_argument(
        "--terminate-on-violation",
        action="store_true",
        help="Enable termination when sz < 0 (south hemisphere)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Maximum episode length (default: 20 000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trajectory_output",
        help="Output directory for figures (default: ./trajectory_output)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    env = QuantumSpinCartPoleEnv(
        seed=args.seed,
        terminate_on_violation=args.terminate_on_violation,
    )

    run_single_strategy(
        env=env,
        action=args.action,
        terminate_on_violation=args.terminate_on_violation,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
