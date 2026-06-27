from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .metrics import aggregate_metrics, parse_sumo_outputs, write_json
from .net import load_metadata, sha256_file
from .policies import CheckpointPolicy, FixedPolicy, MaxPressurePolicy
from .scenario import load_scenario_index
from .sumo_env import EnvConfig, SumoTSCEnv


@dataclass
class EvalConfig:
    method: str
    checkpoint: str | Path | None = None
    split: str = "test"
    map_root: str | Path = "map"
    metadata_path: str | Path = "map/metadata/network_metadata.json"
    scenario_index: str | Path = "map/scenarios/scenario_index.csv"
    output_dir: str | Path = "results/test"
    sumo_binary: str = "sumo"
    gui: bool = False
    gui_delay_ms: int = 0
    device: str = "cpu"
    sim_max_time: int = 7200
    overwrite: bool = False
    families: list[str] | None = None
    limit: int | None = None
    control_interval: int = 10
    min_green: int = 20
    max_green: int = 90
    yellow_time: int = 3
    all_red_time: int = 1
    seed: int = 42
    sumo_threads: int = 1


def make_policy(config: EvalConfig, metadata: dict[str, Any]) -> Any:
    if config.method == "fixed":
        return FixedPolicy(metadata)
    if config.method == "pressure":
        return MaxPressurePolicy(metadata)
    if config.method == "proposed":
        if not config.checkpoint:
            raise ValueError("--checkpoint is required for method=proposed")
        return CheckpointPolicy(config.metadata_path, config.checkpoint, device=config.device, deterministic=True)
    raise ValueError(f"Unknown method: {config.method}")


def evaluate(config: EvalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if config.method not in {"fixed", "pressure", "proposed"}:
        raise ValueError("--method must be fixed, pressure, or proposed")
    output = Path(config.output_dir)
    per_scenario_path = output / "per_scenario.csv"
    if per_scenario_path.exists() and not config.overwrite:
        raise FileExistsError(f"{per_scenario_path} exists. Pass --overwrite to replace it.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(config.metadata_path)
    scenario_index_hash = sha256_file(config.scenario_index)
    policy = make_policy(config, metadata)
    scenarios = load_scenario_index(config.scenario_index, split=config.split, families=config.families, limit=config.limit)
    if not scenarios:
        raise RuntimeError(f"No scenarios found for split={config.split}")

    rows: list[dict[str, Any]] = []
    for record in tqdm(scenarios, desc=f"evaluate {config.method}"):
        scenario_output = output / "scenarios" / str(record["family"]) / f"seed_{int(record['seed'])}"
        env_config = EnvConfig(
            net_file=metadata["net_file"],
            metadata_path=config.metadata_path,
            sumo_binary=config.sumo_binary,
            gui=config.gui,
            gui_delay_ms=config.gui_delay_ms,
            control_interval=config.control_interval,
            min_green=config.min_green,
            max_green=config.max_green,
            yellow_time=config.yellow_time,
            all_red_time=config.all_red_time,
            sim_max_time=config.sim_max_time,
            seed=int(record["seed"]),
            output_dir=scenario_output,
            write_xml=True,
            sumo_threads=config.sumo_threads,
        )
        env = SumoTSCEnv(env_config, record)
        total_reward = 0.0
        decision_count = 0
        queue_sum = 0.0
        pressure_sum = 0.0
        switch_count = 0
        invalid_count = 0
        forced_switch_count = 0
        arrived_delta_sum = 0.0
        waiting_delta_sum = 0.0
        waiting_growth_sum = 0.0
        spillback_sum = 0.0
        unserved_wait_sum = 0.0
        info: dict[str, Any] = {}
        start = time.time()
        try:
            state = env.reset()
            done = False
            while not done:
                actions = policy.act(state, env)
                state, reward, done, info = env.step(actions)
                total_reward += float(reward)
                decision_count += 1
                queue_sum += float(info.get("queue_total", 0.0))
                pressure_sum += float(info.get("pressure_total", 0.0))
                switch_count += int(info.get("switch_count", 0))
                invalid_count += int(info.get("invalid_action_count", 0))
                forced_switch_count += int(info.get("forced_switch_count", 0))
                arrived_delta_sum += float(info.get("arrived_delta", 0.0))
                waiting_delta_sum += float(info.get("waiting_delta", 0.0))
                waiting_growth_sum += float(info.get("waiting_growth_rate", 0.0))
                spillback_sum += float(info.get("spillback_mean", 0.0))
                unserved_wait_sum += float(info.get("unserved_wait_mean", 0.0))
        finally:
            env.close()
        wall_time = time.time() - start
        parsed = parse_sumo_outputs(
            tripinfo_path=info.get("tripinfo_path", scenario_output / "tripinfo.xml"),
            summary_path=info.get("summary_path", scenario_output / "summary.xml"),
            statistic_path=info.get("statistic_path", scenario_output / "statistic.xml"),
            departed=int(info.get("departed", 0)),
            arrived=int(info.get("arrived", 0)),
        )
        parsed["teleports"] = max(float(parsed.get("teleports", 0.0)), float(info.get("teleports", 0.0)))
        row = {
            "method": config.method,
            "split": record["split"],
            "family": record["family"],
            "seed": int(record["seed"]),
            "sumocfg": record["sumocfg"],
            "demand_model_version": record.get("demand_model_version", ""),
            "base_hourly": record.get("base_hourly", ""),
            "requested_vehicles": record.get("requested_vehicles", ""),
            "routed_vehicles": record.get("routed_vehicles", ""),
            "route_rate": record.get("route_rate", ""),
            **parsed,
            "total_reward": total_reward,
            "decision_count": decision_count,
            "queue_mean_step": queue_sum / max(1, decision_count),
            "pressure_mean_step": pressure_sum / max(1, decision_count),
            "switch_count": switch_count,
            "forced_switch_count": forced_switch_count,
            "invalid_action_count": invalid_count,
            "arrived_delta_sum": arrived_delta_sum,
            "waiting_delta_sum": waiting_delta_sum,
            "waiting_growth_mean_step": waiting_growth_sum / max(1, decision_count),
            "spillback_mean_step": spillback_sum / max(1, decision_count),
            "unserved_wait_mean_step": unserved_wait_sum / max(1, decision_count),
            "wall_time_seconds": wall_time,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(per_scenario_path, index=False)
    aggregate = aggregate_metrics(df)
    aggregate_payload = {
        "method": config.method,
        "split": config.split,
        "scenario_count": len(df),
        "scenario_index_hash": scenario_index_hash,
        "metrics": aggregate,
    }
    write_json(output / "aggregate.json", aggregate_payload)
    run_config = asdict(config)
    run_config["checkpoint"] = str(config.checkpoint) if config.checkpoint else None
    run_config["scenario_index_hash"] = scenario_index_hash
    write_json(output / "run_config.json", json.loads(json.dumps(run_config, default=str)))
    return df, aggregate_payload
