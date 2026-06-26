from __future__ import annotations

import argparse

from hcmc_tsc.plotting import plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = plot_results(args.results)
    print(f"wrote {args.results}/plots")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

