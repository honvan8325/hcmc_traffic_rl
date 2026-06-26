from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParsedNet:
    net_file: Path
    location: dict[str, str]
    edges: dict[str, dict[str, Any]]
    lanes: dict[str, dict[str, Any]]
    junctions: dict[str, dict[str, Any]]
    connections: list[dict[str, Any]]
    tl_logics: dict[str, dict[str, Any]]


def _as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _is_internal_edge(edge_id: str, attrs: dict[str, str]) -> bool:
    return edge_id.startswith(":") or attrs.get("function") == "internal"


def _split_floats(value: str | None) -> list[float]:
    if not value:
        return []
    return [_as_float(part) for part in value.split(",")]


def _lane_id(edges: dict[str, dict[str, Any]], edge_id: str, lane_index: str | int | None) -> str:
    if edge_id not in edges:
        return f"{edge_id}_{lane_index or 0}"
    idx = _as_int(str(lane_index), 0)
    for lane in edges[edge_id]["lanes"]:
        if lane["index"] == idx:
            return lane["id"]
    return f"{edge_id}_{idx}"


def parse_net(net_file: str | Path) -> ParsedNet:
    path = Path(net_file)
    if not path.exists():
        raise FileNotFoundError(f"SUMO network not found: {path}")

    root = ET.parse(path).getroot()
    location_elem = root.find("location")
    location = dict(location_elem.attrib) if location_elem is not None else {}

    type_defaults: dict[str, dict[str, Any]] = {}
    for type_elem in root.findall("type"):
        type_defaults[type_elem.get("id", "")] = {
            "priority": _as_int(type_elem.get("priority"), 0),
            "numLanes": _as_int(type_elem.get("numLanes"), 1),
            "speed": _as_float(type_elem.get("speed"), 13.89),
        }

    edges: dict[str, dict[str, Any]] = {}
    lanes: dict[str, dict[str, Any]] = {}
    for edge_elem in root.findall("edge"):
        edge_id = edge_elem.get("id", "")
        if _is_internal_edge(edge_id, edge_elem.attrib):
            continue
        edge_type = edge_elem.get("type", "")
        defaults = type_defaults.get(edge_type, {})
        edge_lanes: list[dict[str, Any]] = []
        for lane_elem in edge_elem.findall("lane"):
            lane = {
                "id": lane_elem.get("id", ""),
                "edge": edge_id,
                "index": _as_int(lane_elem.get("index"), len(edge_lanes)),
                "speed": _as_float(lane_elem.get("speed"), _as_float(str(defaults.get("speed", 13.89)))),
                "length": _as_float(lane_elem.get("length"), 1.0),
                "shape": lane_elem.get("shape", ""),
            }
            lanes[lane["id"]] = lane
            edge_lanes.append(lane)
        if not edge_lanes:
            edge_lanes = [{
                "id": f"{edge_id}_0",
                "edge": edge_id,
                "index": 0,
                "speed": _as_float(str(defaults.get("speed", 13.89))),
                "length": 1.0,
                "shape": "",
            }]
            lanes[edge_lanes[0]["id"]] = edge_lanes[0]
        length = sum(l["length"] for l in edge_lanes) / max(1, len(edge_lanes))
        speed = sum(l["speed"] for l in edge_lanes) / max(1, len(edge_lanes))
        edges[edge_id] = {
            "id": edge_id,
            "from": edge_elem.get("from", ""),
            "to": edge_elem.get("to", ""),
            "priority": _as_int(edge_elem.get("priority"), _as_int(str(defaults.get("priority", 0)))),
            "type": edge_type,
            "name": edge_elem.get("name", edge_elem.get("origId", "")),
            "speed": speed,
            "length": length,
            "lane_count": len(edge_lanes),
            "lanes": edge_lanes,
        }

    junctions: dict[str, dict[str, Any]] = {}
    for junction_elem in root.findall("junction"):
        jid = junction_elem.get("id", "")
        if jid.startswith(":"):
            continue
        junctions[jid] = {
            "id": jid,
            "type": junction_elem.get("type", ""),
            "x": _as_float(junction_elem.get("x"), 0.0),
            "y": _as_float(junction_elem.get("y"), 0.0),
            "incLanes": junction_elem.get("incLanes", "").split(),
            "intLanes": junction_elem.get("intLanes", "").split(),
        }

    connections: list[dict[str, Any]] = []
    for conn_elem in root.findall("connection"):
        from_edge = conn_elem.get("from", "")
        to_edge = conn_elem.get("to", "")
        if from_edge not in edges or to_edge not in edges:
            continue
        conn = {
            "from": from_edge,
            "to": to_edge,
            "fromLane": _as_int(conn_elem.get("fromLane"), 0),
            "toLane": _as_int(conn_elem.get("toLane"), 0),
            "from_lane_id": _lane_id(edges, from_edge, conn_elem.get("fromLane")),
            "to_lane_id": _lane_id(edges, to_edge, conn_elem.get("toLane")),
            "tl": conn_elem.get("tl"),
            "linkIndex": _as_int(conn_elem.get("linkIndex"), -1),
            "dir": conn_elem.get("dir", ""),
            "state": conn_elem.get("state", ""),
            "via": conn_elem.get("via", ""),
        }
        connections.append(conn)

    tl_logics: dict[str, dict[str, Any]] = {}
    for tl_elem in root.findall("tlLogic"):
        tl_id = tl_elem.get("id", "")
        phases: list[dict[str, Any]] = []
        for index, phase_elem in enumerate(tl_elem.findall("phase")):
            phases.append({
                "index": index,
                "duration": _as_float(phase_elem.get("duration"), 0.0),
                "state": phase_elem.get("state", ""),
                "minDur": _as_float(phase_elem.get("minDur"), 0.0),
                "maxDur": _as_float(phase_elem.get("maxDur"), 0.0),
                "name": phase_elem.get("name", ""),
            })
        tl_logics[tl_id] = {
            "id": tl_id,
            "type": tl_elem.get("type", ""),
            "programID": tl_elem.get("programID", ""),
            "offset": _as_float(tl_elem.get("offset"), 0.0),
            "phases": phases,
        }

    return ParsedNet(path, location, edges, lanes, junctions, connections, tl_logics)


def _road_name_prior(name: str) -> float:
    n = name.lower()
    hints = ("nguyen", "tran", "le ", "vo ", "dien bien", "cach mang", "pasteur", "nam ky", "ly ")
    return 2.0 if any(h in n for h in hints) else 0.0


def _conv_boundary(location: dict[str, str], junctions: dict[str, dict[str, Any]]) -> tuple[float, float, float, float]:
    vals = _split_floats(location.get("convBoundary"))
    if len(vals) == 4:
        return vals[0], vals[1], vals[2], vals[3]
    if not junctions:
        return 0.0, 0.0, 1.0, 1.0
    xs = [j["x"] for j in junctions.values()]
    ys = [j["y"] for j in junctions.values()]
    return min(xs), min(ys), max(xs), max(ys)


def _boundary_side(x: float, y: float, bounds: tuple[float, float, float, float]) -> str:
    xmin, ymin, xmax, ymax = bounds
    distances = {
        "west": abs(x - xmin),
        "east": abs(x - xmax),
        "south": abs(y - ymin),
        "north": abs(y - ymax),
    }
    return min(distances, key=distances.get)


def _edge_endpoint(parsed: ParsedNet, edge: dict[str, Any], endpoint: str) -> tuple[float, float]:
    jid = edge["from"] if endpoint == "from" else edge["to"]
    j = parsed.junctions.get(jid)
    if j:
        return float(j["x"]), float(j["y"])
    lane = edge["lanes"][0]
    shape = lane.get("shape", "")
    if shape:
        parts = shape.split()
        point = parts[0] if endpoint == "from" else parts[-1]
        xy = point.split(",")
        if len(xy) >= 2:
            return _as_float(xy[0]), _as_float(xy[1])
    return 0.0, 0.0


def _is_near_boundary(x: float, y: float, bounds: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = bounds
    width = max(1.0, xmax - xmin)
    height = max(1.0, ymax - ymin)
    margin = max(35.0, 0.08 * min(width, height))
    return x <= xmin + margin or x >= xmax - margin or y <= ymin + margin or y >= ymax - margin


def _candidate_score(edge: dict[str, Any], side: str) -> float:
    road_type = edge.get("type", "")
    local_penalty = -1.5 if any(x in road_type for x in ("residential", "service", "living_street")) else 0.0
    side_bias = {"west": 0.4, "east": 0.4, "north": 0.2, "south": 0.2}.get(side, 0.0)
    return (
        4.0 * edge.get("lane_count", 1)
        + 0.45 * edge.get("priority", 0)
        + 0.015 * min(edge.get("length", 0.0), 500.0)
        + _road_name_prior(edge.get("name", ""))
        + local_penalty
        + side_bias
    )


def _select_with_side_diversity(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda x: (-x["score"], x["id"]))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for side in ("west", "east", "north", "south"):
        side_best = [c for c in ranked if c["side"] == side and c["id"] not in used]
        if side_best and len(selected) < limit:
            selected.append(side_best[0])
            used.add(side_best[0]["id"])
    for cand in ranked:
        if len(selected) >= limit:
            break
        if cand["id"] not in used:
            selected.append(cand)
            used.add(cand["id"])
    return selected


def detect_boundary_edges(parsed: ParsedNet, limit: int = 16) -> dict[str, list[dict[str, Any]]]:
    out_degree: Counter[str] = Counter()
    in_degree: Counter[str] = Counter()
    for conn in parsed.connections:
        out_degree[conn["from"]] += 1
        in_degree[conn["to"]] += 1

    bounds = _conv_boundary(parsed.location, parsed.junctions)
    source_candidates: list[dict[str, Any]] = []
    sink_candidates: list[dict[str, Any]] = []
    for edge_id, edge in parsed.edges.items():
        fx, fy = _edge_endpoint(parsed, edge, "from")
        tx, ty = _edge_endpoint(parsed, edge, "to")
        from_side = _boundary_side(fx, fy, bounds)
        to_side = _boundary_side(tx, ty, bounds)
        source_ok = _is_near_boundary(fx, fy, bounds) or in_degree[edge_id] <= 0
        sink_ok = _is_near_boundary(tx, ty, bounds) or out_degree[edge_id] <= 0
        if source_ok:
            source_candidates.append({
                "id": edge_id,
                "side": from_side,
                "score": round(_candidate_score(edge, from_side), 6),
                "lane_count": edge["lane_count"],
                "priority": edge["priority"],
                "length": round(edge["length"], 3),
                "name": edge.get("name", ""),
            })
        if sink_ok:
            sink_candidates.append({
                "id": edge_id,
                "side": to_side,
                "score": round(_candidate_score(edge, to_side), 6),
                "lane_count": edge["lane_count"],
                "priority": edge["priority"],
                "length": round(edge["length"], 3),
                "name": edge.get("name", ""),
            })

    if len(source_candidates) < 2:
        for edge_id, edge in parsed.edges.items():
            fx, fy = _edge_endpoint(parsed, edge, "from")
            side = _boundary_side(fx, fy, bounds)
            source_candidates.append({
                "id": edge_id,
                "side": side,
                "score": round(_candidate_score(edge, side), 6),
                "lane_count": edge["lane_count"],
                "priority": edge["priority"],
                "length": round(edge["length"], 3),
                "name": edge.get("name", ""),
            })
    if len(sink_candidates) < 2:
        for edge_id, edge in parsed.edges.items():
            tx, ty = _edge_endpoint(parsed, edge, "to")
            side = _boundary_side(tx, ty, bounds)
            sink_candidates.append({
                "id": edge_id,
                "side": side,
                "score": round(_candidate_score(edge, side), 6),
                "lane_count": edge["lane_count"],
                "priority": edge["priority"],
                "length": round(edge["length"], 3),
                "name": edge.get("name", ""),
            })

    return {
        "sources": _select_with_side_diversity(source_candidates, limit),
        "sinks": _select_with_side_diversity(sink_candidates, limit),
    }


def _green_actions(tl_logic: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    actions: list[dict[str, Any]] = []
    for phase in tl_logic["phases"]:
        state = phase["state"]
        if ("y" in state.lower()) or not any(ch in "Gg" for ch in state):
            continue
        if state in seen:
            continue
        seen.add(state)
        actions.append({
            "action": len(actions),
            "phase_index": phase["index"],
            "state": state,
            "duration": phase["duration"],
        })
    return actions


def _served_links_for_state(state: str, connections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    served: list[dict[str, Any]] = []
    for conn in connections:
        link_index = conn["linkIndex"]
        if link_index < 0 or link_index >= len(state):
            continue
        if state[link_index] in "Gg":
            served.append({
                "link_index": link_index,
                "from_edge": conn["from"],
                "to_edge": conn["to"],
                "from_lane": conn["from_lane_id"],
                "to_lane": conn["to_lane_id"],
                "direction": conn["dir"],
            })
    return served


def _edge_graph(connections: list[dict[str, Any]]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for conn in connections:
        if conn["to"] not in graph[conn["from"]]:
            graph[conn["from"]].append(conn["to"])
    return dict(graph)


def _downstream_adjacency(agents: list[dict[str, Any]], connections: list[dict[str, Any]], max_depth: int = 8) -> list[list[int]]:
    n = len(agents)
    incoming_edge_agents: dict[str, set[int]] = defaultdict(set)
    for idx, agent in enumerate(agents):
        for edge_id in agent["incoming_edges"]:
            incoming_edge_agents[edge_id].add(idx)

    graph = _edge_graph(connections)
    adjacency = [[0 for _ in range(n)] for _ in range(n)]
    for i, agent in enumerate(agents):
        starts = set(agent["outgoing_edges"])
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((edge, 0) for edge in starts)
        while queue:
            edge_id, depth = queue.popleft()
            if edge_id in visited or depth > max_depth:
                continue
            visited.add(edge_id)
            for j in incoming_edge_agents.get(edge_id, set()):
                if i != j:
                    adjacency[i][j] = 1
            for nxt in graph.get(edge_id, []):
                if nxt not in visited:
                    queue.append((nxt, depth + 1))
    return adjacency


def build_network_metadata(net_file: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    parsed = parse_net(net_file)
    by_tl: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for conn in parsed.connections:
        if conn.get("tl"):
            by_tl[str(conn["tl"])].append(conn)

    agents: list[dict[str, Any]] = []
    static_tls: list[dict[str, Any]] = []
    for tl_id in sorted(parsed.tl_logics):
        controlled = sorted(by_tl.get(tl_id, []), key=lambda c: (c["linkIndex"], c["from"], c["to"]))
        actions = _green_actions(parsed.tl_logics[tl_id])
        if not controlled or len(actions) < 2:
            static_tls.append({
                "id": tl_id,
                "reason": "insufficient_green_actions" if controlled else "no_controlled_connections",
                "num_valid_actions": len(actions),
                "num_controlled_connections": len(controlled),
            })
            continue

        all_links: list[dict[str, Any]] = []
        for action in actions:
            served = _served_links_for_state(action["state"], controlled)
            action["served_links"] = served
            all_links.extend(served)

        incoming_lanes = sorted({link["from_lane"] for link in all_links})
        outgoing_lanes = sorted({link["to_lane"] for link in all_links})
        incoming_edges = sorted({link["from_edge"] for link in all_links})
        outgoing_edges = sorted({link["to_edge"] for link in all_links})
        agents.append({
            "id": tl_id,
            "tl_id": tl_id,
            "index": len(agents),
            "num_actions": len(actions),
            "incoming_lanes": incoming_lanes,
            "outgoing_lanes": outgoing_lanes,
            "incoming_edges": incoming_edges,
            "outgoing_edges": outgoing_edges,
            "controlled_link_indices": sorted({c["linkIndex"] for c in controlled if c["linkIndex"] >= 0}),
            "action_states": [a["state"] for a in actions],
            "action_phase_indices": [a["phase_index"] for a in actions],
            "actions": actions,
        })

    p_max = max((agent["num_actions"] for agent in agents), default=0)
    action_mask = []
    for agent in agents:
        action_mask.append([1 if i < agent["num_actions"] else 0 for i in range(p_max)])

    adjacency = _downstream_adjacency(agents, parsed.connections) if agents else []
    boundary = detect_boundary_edges(parsed)

    lane_count = sum(edge["lane_count"] for edge in parsed.edges.values())
    edge_type_counts = Counter(edge["type"] or "unknown" for edge in parsed.edges.values())
    metadata = {
        "net_file": str(Path(net_file)),
        "net_sha256": sha256_file(net_file),
        "num_agents": len(agents),
        "num_static_tls": len(static_tls),
        "num_tls_total": len(parsed.tl_logics),
        "agents": agents,
        "static_tls": static_tls,
        "p_max": p_max,
        "action_mask": action_mask,
        "adjacency": adjacency,
        "boundary_sources": boundary["sources"],
        "boundary_sinks": boundary["sinks"],
        "edge_summary": {
            "num_edges": len(parsed.edges),
            "num_lanes": lane_count,
            "total_lane_km": round(sum(edge["length"] * edge["lane_count"] for edge in parsed.edges.values()) / 1000.0, 6),
            "edge_type_counts": dict(sorted(edge_type_counts.items())),
            "source_candidate_count": len(boundary["sources"]),
            "sink_candidate_count": len(boundary["sinks"]),
        },
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_metadata(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Metadata not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_demand_edges(metadata: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    payload = {
        "net_file": metadata["net_file"],
        "net_sha256": metadata.get("net_sha256"),
        "sources": metadata["boundary_sources"],
        "sinks": metadata["boundary_sinks"],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

