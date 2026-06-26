from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=60)
    parser.add_argument("--test-count", type=int, default=28)
    parser.add_argument("--total-updates", type=int, default=60)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--sim-max-time", type=int, default=7200)
    parser.add_argument("--sumo-binary", default="sumo")
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
    if args.fast:
        build_cmd.append("--fast")
    run(build_cmd)
    eval_common = [
        "--split",
        "test",
        "--sumo-binary",
        args.sumo_binary,
        "--sim-max-time",
        str(args.sim_max_time),
    ]
    if args.overwrite:
        eval_common.append("--overwrite")
    run([py, "scripts/evaluate.py", "--method", "fixed", "--output", "results/fixed/test", *eval_common])
    run([py, "scripts/evaluate.py", "--method", "pressure", "--output", "results/pressure/test", *eval_common])
    run([
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
        "6",
        "--bc-epochs",
        "8",
        "--torch-threads",
        "1",
    ])
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

