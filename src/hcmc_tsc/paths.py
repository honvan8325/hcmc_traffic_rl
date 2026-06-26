from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_ROOT = PROJECT_ROOT / "map"
DEFAULT_NET_FILE = DEFAULT_MAP_ROOT / "net" / "network.net.xml"
DEFAULT_METADATA = DEFAULT_MAP_ROOT / "metadata" / "network_metadata.json"
DEFAULT_DEMAND_EDGES = DEFAULT_MAP_ROOT / "metadata" / "demand_edges.json"
DEFAULT_SCENARIO_ROOT = DEFAULT_MAP_ROOT / "scenarios"
DEFAULT_SCENARIO_INDEX_CSV = DEFAULT_SCENARIO_ROOT / "scenario_index.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_parent(path: str | Path) -> Path:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
