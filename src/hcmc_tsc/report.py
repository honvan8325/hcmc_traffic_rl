from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ["fixed", "pressure", "proposed"]
LOWER_IS_BETTER = ["avg_travel_time", "avg_waiting_time", "avg_time_loss", "queue_mean_step", "teleports"]
HIGHER_IS_BETTER = ["completion_rate"]


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_table(results: Path) -> str:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        aggregate = _load_json(results / method / "test" / "aggregate.json")
        metrics = aggregate.get("metrics", {})
        row = {"method": method}
        for key in LOWER_IS_BETTER + HIGHER_IS_BETTER:
            row[key] = metrics.get(key, {}).get("mean", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    return _markdown_table(df)


def _paired_improvement(results: Path, baseline: str) -> str:
    prop_path = results / "proposed" / "test" / "per_scenario.csv"
    base_path = results / baseline / "test" / "per_scenario.csv"
    if not prop_path.exists() or not base_path.exists():
        return f"No paired table available for proposed vs {baseline}."
    prop = pd.read_csv(prop_path)
    base = pd.read_csv(base_path)
    merged = prop.merge(base, on=["family", "seed"], suffixes=("_proposed", f"_{baseline}"))
    if merged.empty:
        return f"No matching family/seed pairs for proposed vs {baseline}."
    rows: list[dict[str, Any]] = []
    for metric in LOWER_IS_BETTER:
        p = pd.to_numeric(merged[f"{metric}_proposed"], errors="coerce")
        b = pd.to_numeric(merged[f"{metric}_{baseline}"], errors="coerce")
        improvement = (b - p) / b.replace(0, pd.NA) * 100.0
        rows.append({"metric": metric, "mean_improvement_pct": float(improvement.dropna().mean()) if improvement.dropna().size else 0.0})
    for metric in HIGHER_IS_BETTER:
        p = pd.to_numeric(merged[f"{metric}_proposed"], errors="coerce")
        b = pd.to_numeric(merged[f"{metric}_{baseline}"], errors="coerce")
        improvement = (p - b) / b.replace(0, pd.NA) * 100.0
        rows.append({"metric": metric, "mean_improvement_pct": float(improvement.dropna().mean()) if improvement.dropna().size else 0.0})
    return _markdown_table(pd.DataFrame(rows))


def _scenario_counts(results: Path) -> str:
    rows: list[pd.DataFrame] = []
    for method in METHODS:
        path = results / method / "test" / "per_scenario.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["method"] = method
            rows.append(df[["method", "split", "family", "seed"]])
    if not rows:
        return "No evaluated scenarios found."
    all_rows = pd.concat(rows, ignore_index=True)
    counts = all_rows.groupby(["method", "split", "family"]).size().reset_index(name="count")
    return _markdown_table(counts)


def _plot_list(results: Path) -> str:
    plots = sorted((results / "plots").glob("*.png"))
    if not plots:
        return "No plot files found."
    return "\n".join(f"- `{plot}`" for plot in plots)


def build_report(results: str | Path = "results") -> Path:
    root = Path(results)
    report_path = root / "FINAL_REPORT.md"
    proposed_run = _load_json(root / "proposed" / "test" / "run_config.json")
    checkpoint = proposed_run.get("checkpoint", "results/proposed/train/checkpoints/last.pt")
    train_log = root / "proposed" / "train" / "train_log.csv"
    training_config = {
        "checkpoint": checkpoint,
        "train_log": str(train_log) if train_log.exists() else "",
    }
    lines = [
        "# Final SUMO TSC Report",
        "",
        "## Method Definitions",
        "",
        "- `fixed`: deterministic fixed-time phase cycling over each controllable SUMO traffic light's valid green phases.",
        "- `pressure`: adaptive MaxPressure with min/max green, action masks, downstream spillback penalty, switch penalty, and starvation bonus.",
        "- `proposed`: Dynamic Directed Graph-MAPPO-BC with centralized critic, shared actor, directed graph attention, dynamic action masks, MaxPressure behavior-cloning warm-start, and PPO fine-tuning in real SUMO.",
        "",
        "The proposed method uses a MaxPressure BC warm-start. It is not pure RL from scratch.",
        "",
        "Demand scenarios use corridor-weighted realistic priors, not measured link-level turning counts. The priors combine road hierarchy, named-road land use, directional commute logic, weather/holiday assumptions, and SUMO-valid OD routes.",
        "",
        "## Training Config",
        "",
        "```json",
        json.dumps(training_config, indent=2),
        "```",
        "",
        "## Scenario Counts",
        "",
        _scenario_counts(root),
        "",
        "## Aggregate Comparison",
        "",
        _aggregate_table(root),
        "",
        "## Paired Improvement: Proposed vs Fixed",
        "",
        _paired_improvement(root, "fixed"),
        "",
        "## Paired Improvement: Proposed vs Pressure",
        "",
        _paired_improvement(root, "pressure"),
        "",
        "## Plot Files",
        "",
        _plot_list(root),
        "",
        "## Checkpoint",
        "",
        f"`{checkpoint}`",
        "",
        "## Leakage Guard",
        "",
        "Training uses only train scenarios. Final comparison tables use test scenarios only. Checkpoint selection is based on training reward (`last.pt` or `best_train.pt`) and not on test metrics.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
