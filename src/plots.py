"""All figure-generation logic. Called from scripts/make_figures.py.

Styling: consistent across figures, and readable without color -- QDQN and
MLP series are distinguished by both color AND linestyle/marker, not color
alone.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SOLVE_THRESHOLD = 475.0

QDQN_STYLE = dict(color="#4C72B0", linestyle="-", marker="o", label="QDQN (VQC)")
MLP_STYLE = dict(color="#DD8452", linestyle="--", marker="s", label="MLP baseline")


def load_results(config_name: str, seeds: list[int], results_dir: str = "results") -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = os.path.join(results_dir, f"{config_name}_{seed}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no result CSVs found for config={config_name}, seeds={seeds} in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def _save(fig: plt.Figure, out_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    fig.savefig(f"{out_prefix}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def _median_iqr_by_episode(df: pd.DataFrame, value_col: str, x_col: str = "episode"):
    grouped = df.groupby(x_col)[value_col]
    median = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)
    return median.index.values, median.values, q25.values, q75.values


# --- (1) Learning curves: reward vs episode, median + IQR, both models --------

def plot_learning_curves(qdqn_df: pd.DataFrame | None, mlp_df: pd.DataFrame | None, out_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, style in [(qdqn_df, QDQN_STYLE), (mlp_df, MLP_STYLE)]:
        if df is None:
            continue
        x, median, q25, q75 = _median_iqr_by_episode(df, "episode_reward", "episode")
        ax.plot(x, median, color=style["color"], linestyle=style["linestyle"], label=style["label"], linewidth=2)
        ax.fill_between(x, q25, q75, color=style["color"], alpha=0.2)

    ax.axhline(SOLVE_THRESHOLD, color="green", linestyle=":", linewidth=1.5, label="solved (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Learning curves (median across seeds, IQR shaded)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_prefix)


# --- (2) Sample efficiency: reward vs environment steps -----------------------

def plot_sample_efficiency(qdqn_df: pd.DataFrame | None, mlp_df: pd.DataFrame | None, out_prefix: str, n_grid: int = 200) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for df, style in [(qdqn_df, QDQN_STYLE), (mlp_df, MLP_STYLE)]:
        if df is None:
            continue
        max_steps = df["total_env_steps"].max()
        grid = np.linspace(0, max_steps, n_grid)
        per_seed_interp = []
        for seed, seed_df in df.groupby("seed"):
            seed_df = seed_df.sort_values("total_env_steps")
            interp = np.interp(grid, seed_df["total_env_steps"], seed_df["episode_reward"])
            per_seed_interp.append(interp)
        stacked = np.vstack(per_seed_interp)
        median = np.median(stacked, axis=0)
        q25 = np.percentile(stacked, 25, axis=0)
        q75 = np.percentile(stacked, 75, axis=0)
        ax.plot(grid, median, color=style["color"], linestyle=style["linestyle"], label=style["label"], linewidth=2)
        ax.fill_between(grid, q25, q75, color=style["color"], alpha=0.2)

    ax.axhline(SOLVE_THRESHOLD, color="green", linestyle=":", linewidth=1.5, label="solved (475)")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Sample efficiency (median across seeds, IQR shaded)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_prefix)


# --- (3) Scaling parameter evolution: w and lam vs training step --------------

def _parse_param_column(series: pd.Series) -> np.ndarray | None:
    """Parse a 'v1,v2,...' string column into a 2D array [n_rows, n_values]."""
    non_null = series.dropna()
    non_null = non_null[non_null != ""]
    if len(non_null) == 0:
        return None
    parsed = non_null.apply(lambda s: [float(v) for v in str(s).split(",")])
    return np.stack(parsed.values)


def plot_scaling_evolution(qdqn_df: pd.DataFrame, out_prefix: str) -> None:
    """w[0], w[1], and each lam[i] vs training step (grad_steps), one seed's run.

    Direct visual evidence for Failure Mode 1: w should climb from 1 toward
    the tens as training makes the model represent CartPole's true Q-scale.
    """
    seed = sorted(qdqn_df["seed"].unique())[0]
    df = qdqn_df[qdqn_df["seed"] == seed].sort_values("grad_steps")

    w_values = _parse_param_column(df["w"])
    lam_values = _parse_param_column(df["lam"])
    steps = df["grad_steps"].values[: len(w_values)] if w_values is not None else df["grad_steps"].values

    fig, (ax_w, ax_lam) = plt.subplots(1, 2, figsize=(12, 5))

    if w_values is not None:
        for i in range(w_values.shape[1]):
            ax_w.plot(steps, w_values[:, i], label=f"w[{i}]", linewidth=2)
    ax_w.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="init (1.0)")
    ax_w.set_xlabel("Training step (gradient steps)")
    ax_w.set_ylabel("Output scaling w")
    ax_w.set_title("Output scaling evolution (Failure Mode 1 evidence)")
    ax_w.legend()
    ax_w.grid(alpha=0.3)

    if lam_values is not None:
        for i in range(lam_values.shape[1]):
            ax_lam.plot(steps, lam_values[:, i], label=f"lam[{i}]", linewidth=2)
    ax_lam.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="init (1.0)")
    ax_lam.set_xlabel("Training step (gradient steps)")
    ax_lam.set_ylabel("Input scaling lam")
    ax_lam.set_title("Input scaling evolution")
    ax_lam.legend()
    ax_lam.grid(alpha=0.3)

    fig.suptitle(f"Scaling parameters vs. training step (seed {seed})")
    _save(fig, out_prefix)


# --- (4) Circuit diagram -------------------------------------------------------

def plot_circuit_diagram(circuit, out_prefix: str) -> None:
    fig = circuit.draw("mpl", style="clifford")
    _save(fig, out_prefix)


# --- (5) Q-value landscape: Q(s,0) - Q(s,1) heatmap, untrained vs trained -----

def plot_q_landscape(
    untrained_model,
    trained_model,
    out_prefix: str,
    theta_range: tuple[float, float] = (-0.2095, 0.2095),
    theta_dot_range: tuple[float, float] = (-2.0, 2.0),
    grid_size: int = 100,
) -> None:
    theta_vals = np.linspace(*theta_range, grid_size)
    theta_dot_vals = np.linspace(*theta_dot_range, grid_size)
    theta_grid, theta_dot_grid = np.meshgrid(theta_vals, theta_dot_vals)

    states = np.zeros((grid_size * grid_size, 4), dtype=np.float32)
    states[:, 2] = theta_grid.ravel()
    states[:, 3] = theta_dot_grid.ravel()
    states_t = torch.as_tensor(states, dtype=torch.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, model, title in [
        (axes[0], untrained_model, "Untrained"),
        (axes[1], trained_model, "Trained"),
    ]:
        with torch.no_grad():
            q_values = model(states_t).numpy()
        diff = (q_values[:, 0] - q_values[:, 1]).reshape(grid_size, grid_size)

        im = ax.pcolormesh(theta_vals, theta_dot_vals, diff, shading="auto", cmap="RdBu", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        ax.contour(theta_vals, theta_dot_vals, diff, levels=[0], colors="black", linewidths=2)
        ax.set_xlabel("theta (rad)")
        ax.set_title(f"{title}: Q(s, push-left) - Q(s, push-right)")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("theta_dot (rad/s)")

    fig.suptitle("Q-value landscape (x=0, x_dot=0), decision boundary at Q0=Q1")
    _save(fig, out_prefix)


# --- (6) Loss and epsilon: twin-axis diagnostic --------------------------------

def plot_loss_epsilon(df: pd.DataFrame, out_prefix: str) -> None:
    seed = sorted(df["seed"].unique())[0]
    seed_df = df[df["seed"] == seed].sort_values("episode")

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_eps = ax_loss.twinx()

    losses = pd.to_numeric(seed_df["mean_loss"], errors="coerce")
    ax_loss.plot(seed_df["episode"], losses, color="#C44E52", linestyle="-", label="mean TD loss")
    ax_eps.plot(seed_df["episode"], seed_df["epsilon"], color="#55A868", linestyle="--", label="epsilon")

    ax_loss.set_xlabel("Episode")
    ax_loss.set_ylabel("Mean TD loss", color="#C44E52")
    ax_eps.set_ylabel("Epsilon", color="#55A868")
    ax_loss.set_title(f"Loss / epsilon diagnostic (seed {seed})")

    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_eps.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax_loss.grid(alpha=0.3)
    _save(fig, out_prefix)


# --- (7) Rollout animation: greedy episode -> GIF ------------------------------

def render_rollout_gif(
    model,
    out_path: str,
    env_id: str = "CartPole-v1",
    seed: int = 0,
    max_steps: int = 500,
    fps: int = 30,
) -> int:
    """Render one greedy (epsilon=0) episode to a GIF.

    Uses matplotlib's PillowWriter rather than imageio: Pillow is already a
    matplotlib dependency in this environment, so this avoids adding a new
    package for a GIF that imageio would produce identically.
    """
    import gymnasium as gym
    from matplotlib.animation import FuncAnimation, PillowWriter

    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    obs, _info = env.reset(seed=seed)
    try:
        for _ in range(max_steps):
            frames.append(env.render())
            with torch.no_grad():
                state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(torch.argmax(model(state_t), dim=1).item())
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return len(frames)


# --- (8) Parameter count vs final performance: scatter -------------------------

def plot_param_count_vs_performance(records: list[dict], out_prefix: str) -> None:
    """records: list of {"model": str, "seed": int, "n_params": int, "final_avg_reward": float}."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for model_name, style in [("QDQN (VQC)", QDQN_STYLE), ("MLP baseline", MLP_STYLE)]:
        pts = [r for r in records if r["model"] == model_name]
        if not pts:
            continue
        xs = [r["n_params"] for r in pts]
        ys = [r["final_avg_reward"] for r in pts]
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"], label=model_name, s=80, edgecolor="black")

    ax.axhline(SOLVE_THRESHOLD, color="green", linestyle=":", linewidth=1.5, label="solved (475)")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final avg reward (last-100-episode greedy eval)")
    ax.set_title("Parameter count vs. final performance (one point per seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_prefix)
