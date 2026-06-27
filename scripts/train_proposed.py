from __future__ import annotations

import argparse

from hcmc_tsc.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", default="map")
    parser.add_argument("--metadata", default="map/metadata/network_metadata.json")
    parser.add_argument("--scenario-index", default="map/scenarios/scenario_index.csv")
    parser.add_argument("--output", default="results/proposed/train")
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay-ms", type=int, default=0)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--sim-max-time", type=int, default=7200)
    parser.add_argument("--bc-scenarios", type=int, default=18)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--entropy-coef-final", type=float, default=0.001)
    parser.add_argument("--entropy-decay-fraction", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = train(TrainConfig(
        map_root=args.map_root,
        metadata_path=args.metadata,
        scenario_index=args.scenario_index,
        output_dir=args.output,
        device=args.device,
        seed=args.seed,
        total_updates=args.total_updates,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        entropy_decay_fraction=args.entropy_decay_fraction,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden=args.hidden,
        graph_layers=args.graph_layers,
        resume=args.resume,
        overwrite=args.overwrite,
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        sumo_binary=args.sumo_binary,
        sim_max_time=args.sim_max_time,
        bc_scenarios=args.bc_scenarios,
        bc_epochs=args.bc_epochs,
        torch_threads=args.torch_threads,
    ))
    print(f"wrote {checkpoint}")


if __name__ == "__main__":
    main()
