"""Generate the figures into figures/.

Gracefully skips any figure whose input data (result CSVs / *_final.pt
checkpoints) isn't available yet, rather than failing the whole run --
useful for generating what's possible from partial/smoke-test runs before a
full multi-seed sweep has completed.

Configs are read through src.config.load_config, NOT raw yaml. v3 moved
configs/*.yaml to a sectioned layout (model:/trainer:/optim:/...), so
`yaml.safe_load(f)["n_qubits"]` -- which this script used to do -- raises
KeyError. load_config is the single place that knows the file layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .. import plots
from ..config import load_config
from ..models.mlp import MLPQFunction
from ..models.vqc import VQCQFunction
from . import announce_paths


def try_load_results(config_name: str, seeds: list[int], results_dir: str):
    try:
        return plots.load_results(config_name, seeds, results_dir)
    except FileNotFoundError as e:
        print(f"  skipping (no data yet): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate all figures from results/ and checkpoints")
    parser.add_argument("--qdqn-config", type=str, default="configs/qdqn.yaml")
    parser.add_argument("--mlp-config", type=str, default="configs/mlp_baseline.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--figures-dir", type=str, default="figures")
    args = parser.parse_args()
    announce_paths(results=args.results_dir, figures=args.figures_dir)

    qdqn_config = load_config(args.qdqn_config)
    mlp_config = load_config(args.mlp_config)

    fdir = args.figures_dir
    Path(fdir).mkdir(parents=True, exist_ok=True)

    print("(1)/(2) learning curves + sample efficiency")
    qdqn_df = try_load_results(qdqn_config["name"], args.seeds, args.results_dir)
    mlp_df = try_load_results(mlp_config["name"], args.seeds, args.results_dir)
    tfq_df = try_load_results("tfq", args.seeds, args.results_dir)
    if qdqn_df is not None or mlp_df is not None:
        plots.plot_learning_curves(qdqn_df, mlp_df, f"{fdir}/01_learning_curves")
        plots.plot_sample_efficiency(qdqn_df, mlp_df, f"{fdir}/02_sample_efficiency")
        print("  saved 01_learning_curves.{png,pdf}, 02_sample_efficiency.{png,pdf}")

    print("(1b) greedy-evaluation curves -- the figure to read policy quality from")
    arms = [(n, d) for n, d in (("QDQN (Qiskit, 46 params)", qdqn_df),
                                ("QDQN (TFQ, 94 params)", tfq_df),
                                ("MLP baseline (44 params)", mlp_df)) if d is not None]
    if arms:
        plots.plot_greedy_eval_curves(arms, f"{fdir}/09_greedy_eval")
        print("  saved 09_greedy_eval.{png,pdf}")
    else:
        print("  skipping: no result data")

    print("(3) scaling parameter evolution")
    if qdqn_df is not None:
        plots.plot_scaling_evolution(qdqn_df, f"{fdir}/03_scaling_evolution")
        print("  saved 03_scaling_evolution.{png,pdf}")
    else:
        print("  skipping: no QDQN result data")

    print("(4) circuit diagram")
    untrained_qdqn = VQCQFunction(
        n_qubits=qdqn_config["n_qubits"], n_layers=qdqn_config["n_layers"], seed=0
    )
    plots.plot_circuit_diagram(untrained_qdqn.circuit, f"{fdir}/04_circuit_diagram")
    print("  saved 04_circuit_diagram.{png,pdf}")

    print("(5) Q-value landscape (untrained vs. trained)")
    trained_qdqn_path = f"{args.results_dir}/{qdqn_config['name']}_{args.seeds[0]}_final.pt"
    if Path(trained_qdqn_path).exists():
        trained_qdqn = VQCQFunction(
            n_qubits=qdqn_config["n_qubits"], n_layers=qdqn_config["n_layers"], seed=args.seeds[0]
        )
        trained_qdqn.load_state_dict(torch.load(trained_qdqn_path, weights_only=False))
        plots.plot_q_landscape(untrained_qdqn, trained_qdqn, f"{fdir}/05_q_landscape")
        print("  saved 05_q_landscape.{png,pdf}")
    else:
        print(f"  skipping: no trained checkpoint at {trained_qdqn_path}")

    print("(6) loss / epsilon diagnostic")
    if qdqn_df is not None:
        plots.plot_loss_epsilon(qdqn_df, f"{fdir}/06_loss_epsilon")
        print("  saved 06_loss_epsilon.{png,pdf}")
    else:
        print("  skipping: no QDQN result data")

    print("(7) rollout GIF")
    if Path(trained_qdqn_path).exists():
        n_frames = plots.render_rollout_gif(trained_qdqn, f"{fdir}/07_rollout.gif", seed=args.seeds[0])
        print(f"  saved 07_rollout.gif ({n_frames} frames)")
    else:
        print("  skipping: no trained checkpoint to roll out")

    print("(8) parameter count vs. final performance")
    records = []
    for config, model_label, model_cls, model_kwargs in [
        (qdqn_config, "QDQN (VQC)", VQCQFunction, dict(n_qubits=qdqn_config["n_qubits"], n_layers=qdqn_config["n_layers"])),
        (mlp_config, "MLP baseline", MLPQFunction, dict(hidden=mlp_config["hidden"])),
    ]:
        for seed in args.seeds:
            ckpt_path = f"{args.results_dir}/{config['name']}_{seed}_final.pt"
            csv_path = f"{args.results_dir}/{config['name']}_{seed}.csv"
            if not (Path(ckpt_path).exists() and Path(csv_path).exists()):
                continue
            model = model_cls(seed=seed, **model_kwargs) if model_cls is VQCQFunction else model_cls(**model_kwargs)
            model.load_state_dict(torch.load(ckpt_path, weights_only=False))
            n_params = sum(p.numel() for p in model.parameters())

            import pandas as pd

            df = pd.read_csv(csv_path)
            final_avg = df["avg_reward_100"].iloc[-1]
            records.append({"model": model_label, "seed": seed, "n_params": n_params, "final_avg_reward": final_avg})

    if records:
        plots.plot_param_count_vs_performance(records, f"{fdir}/08_param_count_vs_performance")
        print("  saved 08_param_count_vs_performance.{png,pdf}")
    else:
        print("  skipping: no (checkpoint, csv) pairs found for either model")


if __name__ == "__main__":
    main()
