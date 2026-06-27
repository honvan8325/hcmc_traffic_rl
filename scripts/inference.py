from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hcmc_tsc.metrics import parse_sumo_outputs, write_json
from hcmc_tsc.net import load_metadata
from hcmc_tsc.policies import CheckpointPolicy, FixedPolicy, MaxPressurePolicy
from hcmc_tsc.scenario import load_scenario_index
from hcmc_tsc.sumo_env import EnvConfig, SumoTSCEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SUMO TSC inference episode and save an action trace.")
    parser.add_argument("--method", choices=["fixed", "pressure", "proposed"], default="proposed")
    parser.add_argument("--checkpoint", default="results/proposed/train/checkpoints/last.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--family")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sumocfg", help="Run a specific scenario.sumocfg instead of selecting from scenario_index.csv.")
    parser.add_argument("--metadata", default="map/metadata/network_metadata.json")
    parser.add_argument("--scenario-index", default="map/scenarios/scenario_index.csv")
    parser.add_argument("--output", default="results/inference")
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay-ms", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Sample proposed actions instead of deterministic argmax.")
    parser.add_argument("--sim-max-time", type=int, default=7200)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--min-green", type=int, default=20)
    parser.add_argument("--max-green", type=int, default=90)
    parser.add_argument("--yellow-time", type=int, default=3)
    parser.add_argument("--all-red-time", type=int, default=1)
    parser.add_argument("--sumo-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_scenario(args: argparse.Namespace) -> dict[str, Any]:
    if args.sumocfg:
        seed = int(args.seed if args.seed is not None else 42)
        return {
            "split": "custom",
            "family": Path(args.sumocfg).parent.parent.name or "custom",
            "seed": seed,
            "sumocfg": args.sumocfg,
            "routes": "",
            "additional_files": [],
            "duration": "",
            "requested_vehicles": "",
            "routed_vehicles": "",
            "route_rate": "",
        }

    families = [args.family] if args.family else None
    scenarios = load_scenario_index(args.scenario_index, split=args.split, families=families)
    if args.seed is not None:
        scenarios = [record for record in scenarios if int(record["seed"]) == int(args.seed)]
    if not scenarios:
        selector = f"split={args.split}"
        if args.family:
            selector += f", family={args.family}"
        if args.seed is not None:
            selector += f", seed={args.seed}"
        raise RuntimeError(f"No scenario found for {selector}.")
    return scenarios[0]


def make_policy(args: argparse.Namespace, metadata: dict[str, Any]) -> Any:
    if args.method == "fixed":
        return FixedPolicy(metadata)
    if args.method == "pressure":
        return MaxPressurePolicy(metadata)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return CheckpointPolicy(args.metadata, checkpoint, device=args.device, deterministic=not args.stochastic)


def output_dir(base: str | Path, method: str, scenario: dict[str, Any]) -> Path:
    family = str(scenario.get("family", "custom"))
    seed = int(scenario.get("seed", 0))
    return Path(base) / method / family / f"seed_{seed}"


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata)
    scenario = select_scenario(args)
    run_dir = output_dir(args.output, args.method, scenario)
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists. Pass --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    policy = make_policy(args, metadata)
    env_config = EnvConfig(
        net_file=metadata["net_file"],
        metadata_path=args.metadata,
        sumo_binary=args.sumo_binary,
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        control_interval=args.control_interval,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_time=args.yellow_time,
        all_red_time=args.all_red_time,
        sim_max_time=args.sim_max_time,
        seed=int(scenario.get("seed", args.seed if args.seed is not None else 42)),
        output_dir=run_dir,
        write_xml=True,
        sumo_threads=args.sumo_threads,
    )

    env = SumoTSCEnv(env_config, scenario)
    trace_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    info: dict[str, Any] = {}
    start = time.time()
    try:
        state = env.reset()
        done = False
        decision = 0
        while not done:
            actions = np.asarray(policy.act(state, env), dtype=np.int64)
            state, reward, done, info = env.step(actions)
            total_reward += float(reward)
            trace_rows.append({
                "decision": decision,
                "sim_time": info.get("sim_time", 0.0),
                "actions": json.dumps(actions.tolist()),
                "reward": float(reward),
                "queue_total": info.get("queue_total", 0.0),
                "waiting_total": info.get("waiting_total", 0.0),
                "pressure_total": info.get("pressure_total", 0.0),
                "switch_count": info.get("switch_count", 0),
                "forced_switch_count": info.get("forced_switch_count", 0),
                "invalid_action_count": info.get("invalid_action_count", 0),
                "arrived_delta": info.get("arrived_delta", 0.0),
                "waiting_delta": info.get("waiting_delta", 0.0),
                "waiting_growth_rate": info.get("waiting_growth_rate", 0.0),
                "spillback_mean": info.get("spillback_mean", 0.0),
                "spillback_fraction": info.get("spillback_fraction", 0.0),
                "unserved_wait_mean": info.get("unserved_wait_mean", 0.0),
                "departed": info.get("departed", 0),
                "arrived": info.get("arrived", 0),
                "running": info.get("running", 0),
                "teleports": info.get("teleports", 0),
                "collisions": info.get("collisions", 0),
            })
            decision += 1
    finally:
        env.close()
    wall_time = time.time() - start

    trace = pd.DataFrame(trace_rows)
    trace.to_csv(run_dir / "decision_trace.csv", index=False)
    parsed = parse_sumo_outputs(
        tripinfo_path=info.get("tripinfo_path", run_dir / "tripinfo.xml"),
        summary_path=info.get("summary_path", run_dir / "summary.xml"),
        statistic_path=info.get("statistic_path", run_dir / "statistic.xml"),
        departed=int(info.get("departed", 0)),
        arrived=int(info.get("arrived", 0)),
    )
    metrics = {
        "method": args.method,
        "split": scenario.get("split", ""),
        "family": scenario.get("family", ""),
        "seed": int(scenario.get("seed", 0)),
        "sumocfg": scenario.get("sumocfg", ""),
        "checkpoint": str(args.checkpoint) if args.method == "proposed" else "",
        "stochastic": bool(args.stochastic),
        "decision_count": len(trace_rows),
        "total_reward": total_reward,
        "queue_mean_step": float(trace["queue_total"].mean()) if not trace.empty else 0.0,
        "pressure_mean_step": float(trace["pressure_total"].mean()) if not trace.empty else 0.0,
        "switch_count": int(trace["switch_count"].sum()) if not trace.empty else 0,
        "forced_switch_count": int(trace["forced_switch_count"].sum()) if not trace.empty else 0,
        "invalid_action_count": int(trace["invalid_action_count"].sum()) if not trace.empty else 0,
        "arrived_delta_sum": float(trace["arrived_delta"].sum()) if not trace.empty else 0.0,
        "waiting_delta_sum": float(trace["waiting_delta"].sum()) if not trace.empty else 0.0,
        "waiting_growth_mean_step": float(trace["waiting_growth_rate"].mean()) if not trace.empty else 0.0,
        "spillback_mean_step": float(trace["spillback_mean"].mean()) if not trace.empty else 0.0,
        "unserved_wait_mean_step": float(trace["unserved_wait_mean"].mean()) if not trace.empty else 0.0,
        "wall_time_seconds": wall_time,
        "requested_vehicles": scenario.get("requested_vehicles", ""),
        "routed_vehicles": scenario.get("routed_vehicles", ""),
        "route_rate": scenario.get("route_rate", ""),
        **parsed,
    }
    write_json(metrics_path, metrics)
    write_json(run_dir / "run_config.json", vars(args))

    print(f"scenario {metrics['split']}/{metrics['family']}/seed_{metrics['seed']}")
    print(f"method {args.method}")
    print(f"decisions {metrics['decision_count']}")
    print(f"departed {metrics['departed']} arrived {metrics['arrived']} completion_rate {metrics['completion_rate']:.4f}")
    print(f"avg_waiting_time {metrics['avg_waiting_time']:.3f} avg_travel_time {metrics['avg_travel_time']:.3f}")
    print(f"teleports {metrics['teleports']} invalid_actions {metrics['invalid_action_count']}")
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
