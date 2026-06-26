from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = ["fixed", "pressure", "proposed"]
TEST_METRICS = [
    ("avg_travel_time", "Average travel time (s)", "test_avg_travel_time.png"),
    ("avg_waiting_time", "Average waiting time (s)", "test_avg_waiting_time.png"),
    ("avg_time_loss", "Average time loss (s)", "test_avg_time_loss.png"),
    ("completion_rate", "Completion rate", "test_completion_rate.png"),
    ("teleports", "Teleports", "test_teleports.png"),
    ("queue_mean_step", "Mean queue per decision", "test_queue_mean_step.png"),
]


def _read_method(results: Path, method: str) -> pd.DataFrame | None:
    path = results / method / "test" / "per_scenario.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["method"] = method
    return df


def _bar_plot(df: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    methods = [m for m in METHODS if m in set(df["method"])]
    means = [pd.to_numeric(df[df["method"] == m][metric], errors="coerce").mean() for m in methods]
    stds = [pd.to_numeric(df[df["method"] == m][metric], errors="coerce").std(ddof=1) for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(methods, means, yerr=stds, capsize=4, color=["#4C78A8", "#59A14F", "#E15759"][: len(methods)])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _line_plot(df: pd.DataFrame, x: str, ys: list[tuple[str, str]], ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label in ys:
        if col in df:
            ax.plot(df[x], df[col], label=label)
    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_results(results: str | Path = "results") -> pd.DataFrame:
    root = Path(results)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    frames = [df for method in METHODS if (df := _read_method(root, method)) is not None]
    if not frames:
        raise FileNotFoundError("No per_scenario.csv files found under results/<method>/test")
    combined = pd.concat(frames, ignore_index=True)

    summary_rows: list[dict[str, float | str]] = []
    for method, group in combined.groupby("method"):
        row: dict[str, float | str] = {"method": method, "scenario_count": int(len(group))}
        for metric, _, _ in TEST_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("method")
    summary.to_csv(plots / "method_summary.csv", index=False)

    for metric, ylabel, filename in TEST_METRICS:
        if metric in combined:
            _bar_plot(combined, metric, ylabel, plots / filename)

    train_log = root / "proposed" / "train" / "train_log.csv"
    if train_log.exists():
        train = pd.read_csv(train_log)
        if "reward_mean" in train:
            _line_plot(train, "update", [("reward_mean", "reward_mean"), ("reward_sum", "reward_sum")], "Training reward", plots / "train_reward_curve.png")
        if "policy_loss" in train:
            _line_plot(train, "update", [("policy_loss", "policy_loss"), ("value_loss", "value_loss")], "Loss", plots / "train_loss_curve.png")
    return summary

