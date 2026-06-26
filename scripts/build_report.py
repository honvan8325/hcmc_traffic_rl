from __future__ import annotations

import argparse

from hcmc_tsc.report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = build_report(args.results)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

