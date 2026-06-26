from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from hcmc_tsc.scenario import ScenarioBuildConfig, build_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--map-root")
    parser.add_argument("--metadata")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--duration", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--allow-low-route-rate", action="store_true")
    return parser.parse_args()


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {config_path}")
    return data


def choose(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(args.config)
    scenario_cfg = experiment.get("scenario_build", {}) or {}
    if not isinstance(scenario_cfg, dict):
        raise ValueError("scenario_build in config must be a mapping.")

    records = build_scenarios(ScenarioBuildConfig(
        map_root=choose(args.map_root, experiment.get("map_root"), "map"),
        metadata_path=choose(args.metadata, experiment.get("metadata"), "map/metadata/network_metadata.json"),
        train_count=int(choose(args.train_count, scenario_cfg.get("train_count"), 60)),
        test_count=int(choose(args.test_count, scenario_cfg.get("test_count"), 28)),
        duration=int(choose(args.duration, scenario_cfg.get("duration"), 3600)),
        seed=int(choose(args.seed, experiment.get("seed"), 42)),
        scale=float(choose(args.scale, scenario_cfg.get("scale"), 1.0)),
        fast=args.fast,
        allow_low_route_rate=args.allow_low_route_rate,
        train_family_counts=scenario_cfg.get("train_family_counts"),
        test_family_counts=scenario_cfg.get("test_family_counts"),
    ))
    counts = Counter((record["split"], record["family"]) for record in records)
    for (split, family), count in sorted(counts.items()):
        print(f"{split}/{family}: {count}")
    requested = sum(int(r["requested_vehicles"]) for r in records)
    routed = sum(int(r["routed_vehicles"]) for r in records)
    print(f"requested_vehicles {requested}")
    print(f"routed_vehicles {routed}")
    print(f"route_rate {routed / max(1, requested):.4f}")
    print("wrote map/scenarios/scenario_index.csv")


if __name__ == "__main__":
    main()
