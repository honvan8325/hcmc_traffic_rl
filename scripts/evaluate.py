from __future__ import annotations

import argparse

from hcmc_tsc.evaluator import EvalConfig, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fixed", "pressure", "proposed"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--map-root", default="map")
    parser.add_argument("--metadata", default="map/metadata/network_metadata.json")
    parser.add_argument("--scenario-index", default="map/scenarios/scenario_index.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay-ms", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim-max-time", type=int, default=7200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--min-green", type=int, default=20)
    parser.add_argument("--max-green", type=int, default=90)
    parser.add_argument("--yellow-time", type=int, default=3)
    parser.add_argument("--all-red-time", type=int, default=1)
    parser.add_argument("--sumo-threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, aggregate = evaluate(EvalConfig(
        method=args.method,
        checkpoint=args.checkpoint,
        split=args.split,
        map_root=args.map_root,
        metadata_path=args.metadata,
        scenario_index=args.scenario_index,
        output_dir=args.output,
        sumo_binary=args.sumo_binary,
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        device=args.device,
        sim_max_time=args.sim_max_time,
        overwrite=args.overwrite,
        families=args.families,
        limit=args.limit,
        control_interval=args.control_interval,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_time=args.yellow_time,
        all_red_time=args.all_red_time,
        sumo_threads=args.sumo_threads,
        jobs=args.jobs,
    ))
    print(f"evaluated {len(df)} scenarios")
    print(f"wrote {args.output}/per_scenario.csv")
    print(f"aggregate_metrics {list(aggregate['metrics'])}")


if __name__ == "__main__":
    main()
