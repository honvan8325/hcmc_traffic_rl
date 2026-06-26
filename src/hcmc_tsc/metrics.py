from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _float_attr(elem: ET.Element, name: str, default: float = 0.0) -> float:
    try:
        return float(elem.get(name, default))
    except (TypeError, ValueError):
        return default


def _intish(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_xml(path: Path, attempts: int = 5, delay: float = 0.2) -> ET.Element:
    last_error: ET.ParseError | None = None
    for attempt in range(attempts):
        try:
            return ET.parse(path).getroot()
        except ET.ParseError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def parse_tripinfo(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {
            "avg_travel_time": 0.0,
            "avg_waiting_time": 0.0,
            "avg_time_loss": 0.0,
            "avg_depart_delay": 0.0,
            "p95_travel_time": 0.0,
            "p95_waiting_time": 0.0,
            "finished_trips": 0,
        }
    root = _parse_xml(p)
    durations: list[float] = []
    waits: list[float] = []
    losses: list[float] = []
    depart_delays: list[float] = []
    for trip in root.findall("tripinfo"):
        durations.append(_float_attr(trip, "duration"))
        waits.append(_float_attr(trip, "waitingTime"))
        losses.append(_float_attr(trip, "timeLoss"))
        depart_delays.append(_float_attr(trip, "departDelay"))
    return {
        "avg_travel_time": float(np.mean(durations)) if durations else 0.0,
        "avg_waiting_time": float(np.mean(waits)) if waits else 0.0,
        "avg_time_loss": float(np.mean(losses)) if losses else 0.0,
        "avg_depart_delay": float(np.mean(depart_delays)) if depart_delays else 0.0,
        "p95_travel_time": float(np.percentile(durations, 95)) if durations else 0.0,
        "p95_waiting_time": float(np.percentile(waits, 95)) if waits else 0.0,
        "finished_trips": len(durations),
    }


def parse_summary(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    root = _parse_xml(p)
    steps = root.findall("step")
    if not steps:
        return {}
    last = steps[-1]
    keys = ("loaded", "inserted", "running", "waiting", "ended", "arrived", "teleports")
    return {key: _float_attr(last, key) for key in keys if key in last.attrib}


def parse_statistic(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    root = _parse_xml(p)
    result: dict[str, float] = {}
    for elem in root.iter():
        for key, value in elem.attrib.items():
            lower = key.lower()
            if lower in {"teleports", "collisions", "inserted", "running", "waiting", "arrived"}:
                result[lower] = _float_attr(elem, key)
    return result


def parse_sumo_outputs(
    tripinfo_path: str | Path,
    summary_path: str | Path,
    statistic_path: str | Path,
    departed: int,
    arrived: int,
) -> dict[str, float]:
    trip = parse_tripinfo(tripinfo_path)
    summary = parse_summary(summary_path)
    statistic = parse_statistic(statistic_path)
    finished = int(trip.get("finished_trips", 0))
    departed_runtime = max(_intish(summary.get("inserted"), 0), int(departed), finished)
    arrived_runtime = max(_intish(summary.get("arrived"), 0), int(arrived), finished)
    unfinished = max(0, departed_runtime - arrived_runtime)
    metrics = {
        **trip,
        "unfinished_trips": unfinished,
        "departed": departed_runtime,
        "arrived": arrived_runtime,
        "completion_rate": arrived_runtime / max(1, departed_runtime),
        "teleports": float(max(_intish(summary.get("teleports"), 0), _intish(statistic.get("teleports"), 0))),
        "inserted": float(max(_intish(summary.get("inserted"), 0), _intish(statistic.get("inserted"), 0), departed_runtime)),
        "running": float(max(_intish(summary.get("running"), 0), _intish(statistic.get("running"), 0))),
        "waiting": float(max(_intish(summary.get("waiting"), 0), _intish(statistic.get("waiting"), 0))),
    }
    return metrics


def aggregate_metrics(per_scenario: pd.DataFrame) -> dict[str, dict[str, float]]:
    keys = [
        "avg_travel_time",
        "avg_waiting_time",
        "avg_time_loss",
        "queue_mean_step",
        "teleports",
        "completion_rate",
        "wall_time_seconds",
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for key in keys:
        if key not in per_scenario:
            continue
        values = pd.to_numeric(per_scenario[key], errors="coerce").dropna()
        aggregate[key] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
