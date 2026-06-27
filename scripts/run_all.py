from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--test-count", type=int, default=28)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--bc-scenarios", type=int, default=18)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--entropy-coef-final", type=float, default=0.001)
    parser.add_argument("--sim-max-time", type=int, default=7200)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--sumo-threads", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable
    run([py, "scripts/build_metadata.py"])
    build_cmd = [
        py,
        "scripts/build_scenarios.py",
        "--train-count",
        str(args.train_count),
        "--test-count",
        str(args.test_count),
        "--seed",
        str(args.seed),
    ]
    run(build_cmd)
    eval_common = [
        "--split",
        "test",
        "--sumo-binary",
        args.sumo_binary,
        "--sim-max-time",
        str(args.sim_max_time),
        "--sumo-threads",
        str(args.sumo_threads),
        "--jobs",
        str(args.jobs),
    ]
    if args.overwrite:
        eval_common.append("--overwrite")
    run([py, "scripts/evaluate.py", "--method", "fixed", "--output", "results/fixed/test", *eval_common])
    run([py, "scripts/evaluate.py", "--method", "pressure", "--output", "results/pressure/test", *eval_common])
    train_cmd = [
        py,
        "scripts/train_proposed.py",
        "--output",
        "results/proposed/train",
        "--total-updates",
        str(args.total_updates),
        "--rollout-steps",
        str(args.rollout_steps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--sumo-binary",
        args.sumo_binary,
        "--sim-max-time",
        str(args.sim_max_time),
        "--bc-scenarios",
        str(args.bc_scenarios),
        "--bc-epochs",
        str(args.bc_epochs),
        "--lr",
        str(args.lr),
        "--minibatch-size",
        str(args.minibatch_size),
        "--hidden",
        str(args.hidden),
        "--graph-layers",
        str(args.graph_layers),
        "--entropy-coef",
        str(args.entropy_coef),
        "--entropy-coef-final",
        str(args.entropy_coef_final),
        "--torch-threads",
        str(args.torch_threads),
    ]
    if args.overwrite:
        train_cmd.append("--overwrite")
    run(train_cmd)
    run([
        py,
        "scripts/evaluate.py",
        "--method",
        "proposed",
        "--checkpoint",
        "results/proposed/train/checkpoints/last.pt",
        "--output",
        "results/proposed/test",
        "--device",
        args.device,
        *eval_common,
    ])
    run([py, "scripts/plot_results.py", "--results", "results"])
    run([py, "scripts/build_report.py", "--results", "results"])


if __name__ == "__main__":
    main()
