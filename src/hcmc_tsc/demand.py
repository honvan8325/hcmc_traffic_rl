from __future__ import annotations

import heapq
import json
import math
import random
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .net import ParsedNet


VEHICLE_MIX: dict[str, float] = {
    "mc_private": 0.70,
    "mc_ridehail": 0.08,
    "car_private": 0.10,
    "car_taxi": 0.04,
    "van_delivery": 0.03,
    "truck_light": 0.015,
    "bus_city": 0.01,
    "other_service": 0.025,
}


VTYPE_SPECS: dict[str, dict[str, str]] = {
    "mc_private": {
        "vClass": "motorcycle",
        "length": "2.10",
        "minGap": "0.65",
        "accel": "2.60",
        "decel": "4.50",
        "tau": "1.00",
        "sigma": "0.55",
        "speedFactor": "1.00",
        "speedDev": "0.14",
        "color": "0.10,0.34,0.85",
    },
    "mc_ridehail": {
        "vClass": "motorcycle",
        "length": "2.15",
        "minGap": "0.60",
        "accel": "2.80",
        "decel": "4.80",
        "tau": "1.00",
        "sigma": "0.60",
        "speedFactor": "1.02",
        "speedDev": "0.16",
        "color": "0.05,0.55,0.20",
    },
    "car_private": {
        "vClass": "passenger",
        "length": "4.50",
        "minGap": "2.00",
        "accel": "2.30",
        "decel": "4.20",
        "tau": "1.00",
        "sigma": "0.45",
        "speedFactor": "1.00",
        "speedDev": "0.10",
        "color": "0.70,0.70,0.70",
    },
    "car_taxi": {
        "vClass": "taxi",
        "length": "4.55",
        "minGap": "1.80",
        "accel": "2.40",
        "decel": "4.40",
        "tau": "1.00",
        "sigma": "0.50",
        "speedFactor": "1.02",
        "speedDev": "0.12",
        "color": "0.95,0.82,0.10",
    },
    "van_delivery": {
        "vClass": "delivery",
        "length": "5.80",
        "minGap": "2.40",
        "accel": "1.80",
        "decel": "3.80",
        "tau": "1.10",
        "sigma": "0.45",
        "speedFactor": "0.92",
        "speedDev": "0.08",
        "color": "0.80,0.35,0.10",
    },
    "truck_light": {
        "vClass": "truck",
        "length": "7.50",
        "minGap": "2.80",
        "accel": "1.30",
        "decel": "3.60",
        "tau": "1.25",
        "sigma": "0.35",
        "speedFactor": "0.86",
        "speedDev": "0.07",
        "color": "0.55,0.25,0.15",
    },
    "bus_city": {
        "vClass": "bus",
        "length": "12.00",
        "minGap": "3.00",
        "accel": "1.20",
        "decel": "3.50",
        "tau": "1.30",
        "sigma": "0.30",
        "speedFactor": "0.82",
        "speedDev": "0.06",
        "color": "0.85,0.05,0.05",
    },
    "other_service": {
        "vClass": "passenger",
        "length": "4.80",
        "minGap": "2.20",
        "accel": "2.00",
        "decel": "4.00",
        "tau": "1.05",
        "sigma": "0.50",
        "speedFactor": "0.96",
        "speedDev": "0.12",
        "color": "0.35,0.35,0.45",
    },
}


DEMAND_MODEL_VERSION = "corridor_priors_v2_dense"
DEFAULT_BASE_HOURLY = 1800.0

ROAD_TIER_PRIORS: dict[str, dict[str, Any]] = {
    "primary": {
        "weight": 1.00,
        "roads": [
            "Nam Kỳ Khởi Nghĩa",
            "Điện Biên Phủ",
            "Nguyễn Thị Minh Khai",
            "Võ Thị Sáu",
            "Hai Bà Trưng",
        ],
    },
    "secondary": {
        "weight": 0.68,
        "roads": [
            "Pasteur",
            "Võ Văn Tần",
            "Trần Cao Vân",
        ],
    },
    "local": {
        "weight": 0.35,
        "roads": [
            "Nguyễn Đình Chiểu",
            "Trần Quốc Toản",
            "Tú Xương",
            "Trần Quốc Thảo",
            "Ngô Thời Nhiệm",
            "Lê Quý Đôn",
            "Phùng Khắc Khoan",
            "Mạc Đĩnh Chi",
        ],
    },
}

TRAIN_FAMILY_COUNTS_DEFAULT = {
    "base_midday": 14,
    "am_peak": 14,
    "pm_peak": 14,
    "mild_rain": 10,
    "friday_pm_leisure": 4,
    "stress_overload": 4,
}

TEST_FAMILY_COUNTS_DEFAULT = {
    "holiday_low": 6,
    "incident_lane_loss": 6,
    "rainy_peak": 4,
    "night_low": 4,
    "saturday_mixed": 4,
    "airport_holiday_edge": 4,
}

FAMILY_DEMAND = {
    "base_midday": 1.00,
    "am_peak": 1.35,
    "pm_peak": 1.45,
    "mild_rain": 0.95,
    "friday_pm_leisure": 1.35,
    "stress_overload": 1.70,
    "rainy_peak": 1.20,
    "holiday_low": 0.55,
    "night_low": 0.35,
    "saturday_mixed": 0.85,
    "airport_holiday_edge": 0.75,
    "incident_lane_loss": 1.25,
}

TIME_BIN_PROFILES = {
    "base_midday": [0.22, 0.25, 0.27, 0.26],
    "am_peak": [0.18, 0.25, 0.32, 0.25],
    "pm_peak": [0.20, 0.28, 0.32, 0.20],
    "mild_rain": [0.23, 0.27, 0.28, 0.22],
    "friday_pm_leisure": [0.16, 0.24, 0.34, 0.26],
    "stress_overload": [0.20, 0.27, 0.31, 0.22],
    "rainy_peak": [0.20, 0.28, 0.32, 0.20],
    "holiday_low": [0.24, 0.26, 0.26, 0.24],
    "night_low": [0.35, 0.30, 0.22, 0.13],
    "saturday_mixed": [0.20, 0.25, 0.29, 0.26],
    "airport_holiday_edge": [0.21, 0.26, 0.29, 0.24],
    "incident_lane_loss": [0.20, 0.27, 0.31, 0.22],
}

CBD_LEISURE_ROADS = [
    "Hai Bà Trưng",
    "Nguyễn Thị Minh Khai",
    "Trần Cao Vân",
    "Võ Thị Sáu",
]
AIRPORT_EDGE_ROADS = ["Nam Kỳ Khởi Nghĩa", "Điện Biên Phủ"]


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    source: str
    sink: str
    source_side: str
    sink_side: str
    edges: list[str]
    travel_weight: float
    base_weight: float
    source_prior: float
    sink_prior: float
    primary_edge_count: int
    secondary_edge_count: int
    local_edge_count: int
    road_names: list[str]


def normalize_road_name(name: str) -> str:
    text = unicodedata.normalize("NFD", name or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(text.replace("Đ", "D").replace("đ", "d").lower().split())


NORMALIZED_ROAD_TIERS = {
    tier: [normalize_road_name(name) for name in spec["roads"]]
    for tier, spec in ROAD_TIER_PRIORS.items()
}


def road_tier(name: str) -> str:
    normalized = normalize_road_name(name)
    for tier, roads in NORMALIZED_ROAD_TIERS.items():
        if any(road and road in normalized for road in roads):
            return tier
    return "local"


def road_tier_weight(name: str) -> float:
    return float(ROAD_TIER_PRIORS.get(road_tier(name), ROAD_TIER_PRIORS["local"])["weight"])


def edge_demand_prior(edge: dict[str, Any], boundary_score: float = 1.0) -> float:
    tier_weight = road_tier_weight(str(edge.get("name", "")))
    lane_factor = min(1.60, 0.70 + 0.25 * max(1, int(edge.get("lane_count", 1))))
    priority_factor = 0.75 + min(0.45, max(0.0, float(edge.get("priority", 0))) / 30.0)
    boundary_factor = min(1.35, max(0.70, math.sqrt(max(0.01, boundary_score)) / 5.0))
    return max(0.05, tier_weight * lane_factor * priority_factor * boundary_factor)


def weighted_edge_graph(parsed: ParsedNet) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = {edge_id: [] for edge_id in parsed.edges}
    for conn in parsed.connections:
        to_edge = parsed.edges[conn["to"]]
        speed = max(1.0, float(to_edge.get("speed", 13.89)))
        length = max(1.0, float(to_edge.get("length", 1.0)))
        road_type = str(to_edge.get("type", ""))
        local_penalty = 4.0 if any(k in road_type for k in ("residential", "service", "living_street")) else 0.0
        priority_bonus = max(0.0, 12.0 - float(to_edge.get("priority", 0))) * 0.08
        graph.setdefault(conn["from"], []).append((conn["to"], length / speed + local_penalty + priority_bonus))
    return graph


def shortest_path(graph: dict[str, list[tuple[str, float]]], source: str, sink: str) -> tuple[list[str], float] | None:
    if source == sink:
        return None
    queue: list[tuple[float, str]] = [(0.0, source)]
    dist: dict[str, float] = {source: 0.0}
    parent: dict[str, str] = {}
    while queue:
        cost, edge = heapq.heappop(queue)
        if edge == sink:
            path = [sink]
            while path[-1] != source:
                path.append(parent[path[-1]])
            path.reverse()
            return path, cost
        if cost > dist.get(edge, math.inf):
            continue
        for nxt, weight in graph.get(edge, []):
            new_cost = cost + weight
            if new_cost < dist.get(nxt, math.inf):
                dist[nxt] = new_cost
                parent[nxt] = edge
                heapq.heappush(queue, (new_cost, nxt))
    return None


def build_route_candidates(parsed: ParsedNet, metadata: dict[str, Any]) -> list[RouteCandidate]:
    graph = weighted_edge_graph(parsed)
    candidates: list[RouteCandidate] = []
    sources = metadata["boundary_sources"]
    sinks = metadata["boundary_sinks"]
    for src in sources:
        for dst in sinks:
            source = src["id"]
            sink = dst["id"]
            result = shortest_path(graph, source, sink)
            if result is None:
                continue
            path, weight = result
            if not all(edge in parsed.edges for edge in path):
                continue
            source_score = max(0.1, float(src.get("score", 1.0)))
            sink_score = max(0.1, float(dst.get("score", 1.0)))
            base_weight = min(1.50, max(0.35, math.sqrt(source_score * sink_score) / 8.0))
            route_edges = [parsed.edges[edge_id] for edge_id in path]
            tiers = [road_tier(str(edge.get("name", ""))) for edge in route_edges]
            road_names = sorted({
                str(edge.get("name", "")).strip()
                for edge in route_edges
                if str(edge.get("name", "")).strip()
            })
            route_id = f"r_{len(candidates):05d}"
            candidates.append(RouteCandidate(
                route_id=route_id,
                source=source,
                sink=sink,
                source_side=src.get("side", "unknown"),
                sink_side=dst.get("side", "unknown"),
                edges=path,
                travel_weight=weight,
                base_weight=base_weight,
                source_prior=edge_demand_prior(parsed.edges[source], source_score),
                sink_prior=edge_demand_prior(parsed.edges[sink], sink_score),
                primary_edge_count=tiers.count("primary"),
                secondary_edge_count=tiers.count("secondary"),
                local_edge_count=tiers.count("local"),
                road_names=road_names,
            ))
    return candidates


def family_multiplier(family: str) -> float:
    return FAMILY_DEMAND.get(family, 1.0)


def bin_profile(family: str, duration: int) -> list[tuple[float, float, float]]:
    bins: list[tuple[float, float, float]] = []
    profile = TIME_BIN_PROFILES.get(family, TIME_BIN_PROFILES["base_midday"])
    bin_count = max(1, int(math.ceil(duration / 900.0)))
    t = 0.0
    idx = 0
    while t < duration:
        end = min(float(duration), t + 900.0)
        if bin_count == len(profile):
            weight = profile[idx]
        else:
            position = (idx / max(1, bin_count - 1)) * (len(profile) - 1) if bin_count > 1 else 0.0
            lower = int(math.floor(position))
            upper = min(len(profile) - 1, lower + 1)
            frac = position - lower
            weight = profile[lower] * (1.0 - frac) + profile[upper] * frac
        bins.append((t, end, max(0.05, weight)))
        t = end
        idx += 1
    total = sum(w for _, _, w in bins) or 1.0
    return [(start, end, weight / total) for start, end, weight in bins]


def route_uses(route: RouteCandidate, road_name: str) -> bool:
    target = normalize_road_name(road_name)
    return any(target in normalize_road_name(name) for name in route.road_names)


def route_uses_any(route: RouteCandidate, road_names: list[str]) -> bool:
    return any(route_uses(route, name) for name in road_names)


def route_is_local_only(route: RouteCandidate) -> bool:
    return route.primary_edge_count == 0 and route.secondary_edge_count == 0


def route_uses_local_collector(route: RouteCandidate) -> bool:
    return route.local_edge_count > 0 and route.primary_edge_count == 0


def arterial_bonus(route: RouteCandidate) -> float:
    return min(1.35, 1.0 + 0.08 * route.primary_edge_count)


def direction_modifier(route: RouteCandidate, family: str) -> float:
    modifier = 1.0
    if family == "am_peak":
        if route.sink_side in {"east", "north"}:
            modifier *= 1.20
        if route.source_side in {"east", "north"} and route.sink_side in {"west", "south"}:
            modifier *= 0.90
    if family == "pm_peak":
        if route.source_side in {"east", "north"}:
            modifier *= 1.20
        if route.sink_side in {"east", "north"}:
            modifier *= 0.92
    if family == "base_midday" and route.sink_side in {"east", "north"}:
        modifier *= 1.05
    return modifier


def corridor_family_modifier(route: RouteCandidate, family: str) -> float:
    modifier = 1.0
    if family == "friday_pm_leisure":
        if route_uses_any(route, CBD_LEISURE_ROADS):
            modifier *= 1.20
    elif family == "holiday_low":
        if route_uses(route, "Nam Kỳ Khởi Nghĩa"):
            modifier *= 1.20
        if route_uses_any(route, CBD_LEISURE_ROADS):
            modifier *= 1.10
        if route_is_local_only(route):
            modifier *= 0.75
    elif family == "night_low":
        if route_uses_any(route, CBD_LEISURE_ROADS):
            modifier *= 1.20
        if route_is_local_only(route):
            modifier *= 0.75
    elif family in {"mild_rain", "rainy_peak"}:
        modifier *= 0.85 if route_uses_local_collector(route) else 0.95
    elif family == "stress_overload":
        if route.primary_edge_count > 0:
            modifier *= 1.15
    elif family == "airport_holiday_edge":
        if route_uses(route, "Nam Kỳ Khởi Nghĩa"):
            modifier *= 1.35
        if route_uses(route, "Điện Biên Phủ"):
            modifier *= 1.15
    elif family == "incident_lane_loss":
        if route_uses_any(route, AIRPORT_EDGE_ROADS + CBD_LEISURE_ROADS):
            modifier *= 1.10
    return modifier


def route_weight(route: RouteCandidate, family: str, rng: random.Random) -> float:
    weight = (
        route.base_weight
        * route.source_prior
        * route.sink_prior
        * arterial_bonus(route)
        * direction_modifier(route, family)
        * corridor_family_modifier(route, family)
        * rng.uniform(0.92, 1.08)
    )
    return min(3.0, max(0.05, weight))


def choose_weighted(rng: random.Random, items: list[Any], weights: list[float]) -> Any:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return items[rng.randrange(len(items))]
    target = rng.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += max(0.0, weight)
        if acc >= target:
            return item
    return items[-1]


def choose_vehicle_type(rng: random.Random) -> str:
    items = list(VEHICLE_MIX)
    weights = [VEHICLE_MIX[k] for k in items]
    return str(choose_weighted(rng, items, weights))


def scenario_vehicle_count(
    family: str,
    duration: int,
    rng: random.Random,
    base_hourly: float = DEFAULT_BASE_HOURLY,
) -> int:
    jitter = rng.uniform(0.92, 1.08)
    return max(1, int(round(base_hourly * (duration / 3600.0) * family_multiplier(family) * jitter)))


def vtype_specs_for_family(family: str) -> dict[str, dict[str, str]]:
    specs = {key: dict(value) for key, value in VTYPE_SPECS.items()}
    if family not in {"mild_rain", "rainy_peak"}:
        return specs
    for vtype_id, attrs in specs.items():
        vclass = attrs.get("vClass", "")
        speed_factor = float(attrs.get("speedFactor", "1.0"))
        sigma = float(attrs.get("sigma", "0.5"))
        tau = float(attrs.get("tau", "1.0"))
        min_gap = float(attrs.get("minGap", "1.0"))
        if vclass == "motorcycle":
            attrs["speedFactor"] = f"{speed_factor * 0.90:.2f}"
            attrs["sigma"] = f"{min(1.0, sigma + 0.08):.2f}"
            attrs["tau"] = f"{tau * 1.08:.2f}"
            attrs["minGap"] = f"{min_gap * 1.08:.2f}"
        elif vclass in {"passenger", "taxi", "delivery", "truck", "bus"}:
            attrs["speedFactor"] = f"{speed_factor * 0.93:.2f}"
            attrs["sigma"] = f"{min(1.0, sigma + 0.05):.2f}"
            attrs["tau"] = f"{tau * 1.06:.2f}"
            attrs["minGap"] = f"{min_gap * 1.05:.2f}"
    return specs


def write_routes_file(path: str | Path, routes: list[RouteCandidate], vehicles: list[dict[str, Any]], family: str) -> None:
    root = ET.Element("routes")
    for vtype_id, attrs in vtype_specs_for_family(family).items():
        ET.SubElement(root, "vType", id=vtype_id, **attrs)
    for route in routes:
        ET.SubElement(root, "route", id=route.route_id, edges=" ".join(route.edges))
    for vehicle in vehicles:
        ET.SubElement(
            root,
            "vehicle",
            id=vehicle["id"],
            type=vehicle["type"],
            route=vehicle["route"],
            depart=f"{vehicle['depart']:.2f}",
            departLane="best",
            departSpeed="max",
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)


def major_incident_lane(parsed: ParsedNet) -> tuple[str, float]:
    best_edge = max(
        parsed.edges.values(),
        key=lambda e: (e.get("lane_count", 1), e.get("priority", 0), e.get("length", 0.0), e["id"]),
    )
    lane = best_edge["lanes"][0]
    return lane["id"], float(lane.get("speed", best_edge.get("speed", 13.89)))


def write_incident_file(path: str | Path, parsed: ParsedNet, duration: int, rng: random.Random) -> str:
    lane_id, base_speed = major_incident_lane(parsed)
    start = int(duration * (0.35 + 0.05 * rng.random()))
    end = int(duration * (0.62 + 0.05 * rng.random()))
    factor = 0.30 + 0.30 * rng.random()
    root = ET.Element("additional")
    vss = ET.SubElement(root, "variableSpeedSign", id="incident_lane_loss", lanes=lane_id)
    ET.SubElement(vss, "step", time="0", speed=f"{base_speed:.2f}")
    ET.SubElement(vss, "step", time=str(start), speed=f"{max(1.0, base_speed * factor):.2f}")
    ET.SubElement(vss, "step", time=str(max(start + 1, end)), speed=f"{base_speed:.2f}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    out = Path(path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out.name


def write_sumocfg(path: str | Path, net_file: str | Path, routes_file: str | Path, additional_files: list[str], duration: int) -> None:
    cfg_path = Path(path)
    cfg_dir = cfg_path.parent
    rel_net = Path(net_file)
    if rel_net.is_absolute():
        rel_net = Path("../../../../net/network.net.xml")
    else:
        rel_net = Path("../../../../net/network.net.xml")
    root = ET.Element("configuration")
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "net-file", value=str(rel_net))
    ET.SubElement(input_elem, "route-files", value=Path(routes_file).name)
    if additional_files:
        ET.SubElement(input_elem, "additional-files", value=",".join(additional_files))
    time_elem = ET.SubElement(root, "time")
    ET.SubElement(time_elem, "begin", value="0")
    ET.SubElement(time_elem, "end", value=str(duration))
    processing_elem = ET.SubElement(root, "processing")
    ET.SubElement(processing_elem, "time-to-teleport", value="600")
    report_elem = ET.SubElement(root, "report")
    ET.SubElement(report_elem, "verbose", value="false")
    ET.SubElement(report_elem, "no-warnings", value="true")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    tree.write(cfg_path, encoding="utf-8", xml_declaration=True)


def generate_single_scenario(
    parsed: ParsedNet,
    metadata: dict[str, Any],
    split: str,
    family: str,
    seed: int,
    scenario_dir: str | Path,
    duration: int,
    base_hourly: float = DEFAULT_BASE_HOURLY,
    route_candidates: list[RouteCandidate] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    routes = route_candidates if route_candidates is not None else build_route_candidates(parsed, metadata)
    if not routes:
        requested = scenario_vehicle_count(family, duration, rng, base_hourly)
        return {
            "demand_model_version": DEMAND_MODEL_VERSION,
            "split": split,
            "family": family,
            "seed": seed,
            "sumocfg": "",
            "routes": "",
            "additional_files": [],
            "duration": duration,
            "base_hourly": base_hourly,
            "family_demand_multiplier": family_multiplier(family),
            "requested_vehicles": requested,
            "routed_vehicles": 0,
            "route_rate": 0.0,
            "road_tier_summary": {},
            "time_bin_counts": {},
            "direction_split_summary": {},
        }

    count = scenario_vehicle_count(family, duration, rng, base_hourly)
    bins = bin_profile(family, duration)
    bin_items = list(range(len(bins)))
    bin_weights = [b[2] for b in bins]
    raw_route_weights = [route_weight(route, family, rng) for route in routes]
    route_weight_total = sum(raw_route_weights) or 1.0
    route_weights = [weight / route_weight_total for weight in raw_route_weights]

    vehicles: list[dict[str, Any]] = []
    used_route_ids: set[str] = set()
    time_bin_counts: Counter[str] = Counter()
    direction_split_summary: Counter[str] = Counter()
    road_tier_summary: Counter[str] = Counter()
    for idx in range(count):
        bin_idx = int(choose_weighted(rng, bin_items, bin_weights))
        start, end, _ = bins[bin_idx]
        route = choose_weighted(rng, routes, route_weights)
        used_route_ids.add(route.route_id)
        time_bin_counts[f"{int(start)}_{int(end)}"] += 1
        direction_split_summary[f"{route.source_side}->{route.sink_side}"] += 1
        road_tier_summary["primary_edge_exposures"] += route.primary_edge_count
        road_tier_summary["secondary_edge_exposures"] += route.secondary_edge_count
        road_tier_summary["local_edge_exposures"] += route.local_edge_count
        if route.primary_edge_count > 0:
            road_tier_summary["vehicles_using_primary"] += 1
        elif route.secondary_edge_count > 0:
            road_tier_summary["vehicles_using_secondary"] += 1
        else:
            road_tier_summary["vehicles_local_only"] += 1
        vehicles.append({
            "id": f"{family}_{seed}_{idx:06d}",
            "type": choose_vehicle_type(rng),
            "route": route.route_id,
            "depart": rng.uniform(start, max(start + 0.1, end)),
        })
    vehicles.sort(key=lambda v: (v["depart"], v["id"]))
    used_routes = [r for r in routes if r.route_id in used_route_ids]

    scenario_path = Path(scenario_dir)
    scenario_path.mkdir(parents=True, exist_ok=True)
    routes_path = scenario_path / "routes.rou.xml"
    sumocfg_path = scenario_path / "scenario.sumocfg"
    additional_files: list[str] = []
    if family == "incident_lane_loss":
        additional_files.append(write_incident_file(scenario_path / "incident.add.xml", parsed, duration, rng))
    write_routes_file(routes_path, used_routes, vehicles, family)
    write_sumocfg(sumocfg_path, parsed.net_file, routes_path, additional_files, duration)

    routed = len(vehicles)
    record = {
        "demand_model_version": DEMAND_MODEL_VERSION,
        "split": split,
        "family": family,
        "seed": seed,
        "sumocfg": str(sumocfg_path),
        "routes": str(routes_path),
        "additional_files": [str(scenario_path / name) for name in additional_files],
        "duration": duration,
        "base_hourly": base_hourly,
        "family_demand_multiplier": family_multiplier(family),
        "requested_vehicles": count,
        "routed_vehicles": routed,
        "route_rate": routed / max(1, count),
        "road_tier_summary": dict(sorted(road_tier_summary.items())),
        "time_bin_counts": dict(sorted(time_bin_counts.items())),
        "direction_split_summary": dict(sorted(direction_split_summary.items())),
    }
    (scenario_path / "scenario.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
