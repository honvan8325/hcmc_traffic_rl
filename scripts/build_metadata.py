from __future__ import annotations

import argparse
from pathlib import Path

from hcmc_tsc.net import build_network_metadata, write_demand_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net-file", default="map/net/network.net.xml")
    parser.add_argument("--output", default="map/metadata/network_metadata.json")
    parser.add_argument("--demand-output", default="map/metadata/demand_edges.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_network_metadata(args.net_file, args.output)
    write_demand_edges(metadata, args.demand_output)
    if metadata["num_agents"] <= 0:
        raise RuntimeError("No controllable agents found.")
    if len(metadata["boundary_sources"]) < 2 or len(metadata["boundary_sinks"]) < 2:
        raise RuntimeError("Need at least two boundary sources and two boundary sinks.")
    print(f"num_edges {metadata['edge_summary']['num_edges']}")
    print(f"num_tls_total {metadata['num_tls_total']}")
    print(f"num_agents {metadata['num_agents']}")
    print(f"num_static_tls {metadata['num_static_tls']}")
    print(f"p_max {metadata['p_max']}")
    print(f"boundary_sources {len(metadata['boundary_sources'])}")
    print(f"boundary_sinks {len(metadata['boundary_sinks'])}")
    print(f"wrote {Path(args.output)}")
    print(f"wrote {Path(args.demand_output)}")


if __name__ == "__main__":
    main()

