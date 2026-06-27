from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import ast

import pandas as pd

from .demand import (
    DEFAULT_BASE_HOURLY,
    FAMILY_DEMAND,
    TEST_FAMILY_COUNTS_DEFAULT,
    TRAIN_FAMILY_COUNTS_DEFAULT,
    build_route_candidates,
    generate_single_scenario,
)
from .net import load_metadata, parse_net, write_demand_edges


@dataclass(frozen=True)
class ScenarioBuildConfig:
    map_root: str | Path = "map"
    metadata_path: str | Path = "map/metadata/network_metadata.json"
    train_count: int = 60
    test_count: int = 28
    duration: int = 3600
    seed: int = 42
    base_hourly: float = DEFAULT_BASE_HOURLY
    allow_low_route_rate: bool = False
    train_family_counts: dict[str, int] | None = None
    test_family_counts: dict[str, int] | None = None


def family_count_template(override: dict[str, int] | None, fallback: dict[str, int]) -> dict[str, int]:
    raw = fallback if override is None else override
    template: dict[str, int] = {}
    for family, value in raw.items():
        if family not in FAMILY_DEMAND:
            raise ValueError(f"Unknown scenario family in family counts: {family}")
        count = int(value)
        if count < 0:
            raise ValueError(f"Negative scenario count for {family}: {count}")
        if count > 0:
            template[family] = count
    if not template:
        raise ValueError("Family count template must contain at least one positive count.")
    return template


def allocate_counts(total: int, template: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError(f"Scenario count must be non-negative, got {total}")
    families = list(template)
    template_total = sum(template.values())
    if total == 0:
        return {family: 0 for family in families}
    if total == template_total:
        return {family: int(template[family]) for family in families}
    raw = {family: total * template[family] / max(1, template_total) for family in families}
    counts = {family: int(raw[family]) for family in families}
    remainder = total - sum(counts.values())
    ranked = sorted(families, key=lambda family: (-(raw[family] - counts[family]), families.index(family)))
    for family in ranked[:remainder]:
        counts[family] += 1
    return counts


def scenario_dir(map_root: Path, split: str, family: str, seed: int) -> Path:
    return map_root / "scenarios" / split / family / f"seed_{seed}"


def build_scenarios(config: ScenarioBuildConfig) -> list[dict[str, Any]]:
    map_root = Path(config.map_root)
    metadata = load_metadata(config.metadata_path)
    net_file = Path(metadata["net_file"])
    if not net_file.is_absolute():
        net_file = Path.cwd() / net_file
    parsed = parse_net(net_file)
    route_candidates = build_route_candidates(parsed, metadata)

    demand_edges_path = map_root / "metadata" / "demand_edges.json"
    write_demand_edges(metadata, demand_edges_path)

    records: list[dict[str, Any]] = []
    train_template = family_count_template(config.train_family_counts, TRAIN_FAMILY_COUNTS_DEFAULT)
    test_template = family_count_template(config.test_family_counts, TEST_FAMILY_COUNTS_DEFAULT)
    split_specs = [
        ("train", list(train_template), allocate_counts(config.train_count, train_template)),
        ("test", list(test_template), allocate_counts(config.test_count, test_template)),
    ]
    global_counter = 0
    for split, families, counts in split_specs:
        for family in families:
            for local_idx in range(counts[family]):
                scenario_seed = config.seed + global_counter
                out_dir = scenario_dir(map_root, split, family, scenario_seed)
                record = generate_single_scenario(
                    parsed=parsed,
                    metadata=metadata,
                    split=split,
                    family=family,
                    seed=scenario_seed,
                    scenario_dir=out_dir,
                    duration=config.duration,
                    base_hourly=config.base_hourly,
                    route_candidates=route_candidates,
                )
                records.append(record)
                global_counter += 1

    scenario_root = map_root / "scenarios"
    scenario_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = scenario_root / "scenario_index.csv"
    df.to_csv(csv_path, index=False)

    requested = sum(int(r["requested_vehicles"]) for r in records)
    routed = sum(int(r["routed_vehicles"]) for r in records)
    route_rate = routed / max(1, requested)
    if route_rate < 0.98 and not config.allow_low_route_rate:
        raise RuntimeError(
            f"Route rate too low: {route_rate:.3f} ({routed}/{requested}). "
            "Re-run with --allow-low-route-rate only if this is intentional."
        )
    return records


def load_scenario_index(path: str | Path, split: str | None = None, families: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario index not found: {p}")
    df = pd.read_csv(p)
    if split is not None:
        df = df[df["split"] == split]
    if families:
        df = df[df["family"].isin(families)]
    df = df.sort_values(["split", "family", "seed"]).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    records = df.to_dict(orient="records")
    for record in records:
        additional = record.get("additional_files", [])
        if isinstance(additional, str):
            try:
                record["additional_files"] = ast.literal_eval(additional)
            except (SyntaxError, ValueError):
                record["additional_files"] = [] if additional in ("", "nan") else [additional]
    return records


def counts_by_split_family(records: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    return dict(Counter((r["split"], r["family"]) for r in records))
