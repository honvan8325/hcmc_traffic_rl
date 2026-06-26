from __future__ import annotations

import os
import inspect
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import traci

from .net import load_metadata


OBS_FEATURES = [
    "incoming_queue",
    "incoming_vehicle_count",
    "incoming_waiting_time",
    "downstream_queue",
    "downstream_occupancy",
    "pressure",
    "served_link_count_norm",
    "right_turn_ratio",
    "straight_ratio",
    "left_turn_ratio",
    "downstream_spillback_indicator",
    "current_action",
    "elapsed_green_norm",
    "valid_action",
    "existence_bias",
]


@dataclass
class EnvConfig:
    net_file: str | Path
    metadata_path: str | Path
    sumo_binary: str = "sumo"
    gui: bool = False
    gui_delay_ms: int = 0
    control_interval: int = 10
    min_green: int = 20
    max_green: int = 90
    yellow_time: int = 3
    all_red_time: int = 1
    sim_max_time: int = 7200
    step_length: float = 1.0
    time_to_teleport: int = 600
    seed: int = 42
    output_dir: str | Path | None = None
    write_xml: bool = True
    no_warnings: bool = True
    sumo_threads: int = 1


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


class SumoTSCEnv:
    def __init__(self, config: EnvConfig, scenario: dict[str, Any] | str | Path | None = None):
        self.config = config
        self.metadata = load_metadata(config.metadata_path)
        self.agents = self.metadata["agents"]
        self.num_agents = int(self.metadata["num_agents"])
        self.p_max = int(self.metadata["p_max"])
        self.base_action_mask = np.asarray(self.metadata["action_mask"], dtype=np.float32)
        self.adjacency = np.asarray(self.metadata["adjacency"], dtype=np.float32)
        self.obs_dim = len(OBS_FEATURES)
        self.max_links_per_action = max(
            (
                len(action.get("served_links", []))
                for agent in self.agents
                for action in agent.get("actions", [])
            ),
            default=1,
        )
        self.scenario = scenario
        self.conn: Any | None = None
        self.label = f"hcmc_tsc_{os.getpid()}_{int(time.time() * 1_000_000)}"
        self.current_actions = np.zeros(self.num_agents, dtype=np.int64)
        self.elapsed_green = np.zeros(self.num_agents, dtype=np.float32)
        self._departed = 0
        self._arrived = 0
        self._teleports = 0
        self._collisions = 0
        self._last_switch_count = 0
        self._last_invalid_count = 0
        self._tripinfo_path: Path | None = None
        self._summary_path: Path | None = None
        self._statistic_path: Path | None = None
        self._stdout_path: Path | None = None
        self._stderr_path: Path | None = None

    def reset(self, scenario: dict[str, Any] | str | Path | None = None) -> dict[str, np.ndarray]:
        if scenario is not None:
            self.scenario = scenario
        if self.scenario is None:
            raise ValueError("A scenario record or scenario.sumocfg path is required.")
        self.close()
        self._start_sumo()
        self.current_actions[:] = 0
        self.elapsed_green[:] = 0.0
        self._departed = 0
        self._arrived = 0
        self._teleports = 0
        self._collisions = 0
        self._last_switch_count = 0
        self._last_invalid_count = 0
        for idx, agent in enumerate(self.agents):
            state = agent["action_states"][0]
            self.conn.trafficlight.setRedYellowGreenState(agent["tl_id"], state)
            self.current_actions[idx] = 0
        return self.get_state()

    def _scenario_sumocfg(self) -> Path:
        if isinstance(self.scenario, dict):
            return Path(str(self.scenario["sumocfg"]))
        return Path(str(self.scenario))

    def _start_sumo(self) -> None:
        sumocfg = self._scenario_sumocfg()
        if not sumocfg.exists():
            raise FileNotFoundError(f"Scenario config not found: {sumocfg}")
        scenario_dir = sumocfg.parent
        output_dir = Path(self.config.output_dir) if self.config.output_dir is not None else scenario_dir
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._tripinfo_path = output_dir / "tripinfo.xml"
        self._summary_path = output_dir / "summary.xml"
        self._statistic_path = output_dir / "statistic.xml"
        self._stdout_path = output_dir / "sumo_stdout.log"
        self._stderr_path = output_dir / "sumo_stderr.log"

        binary = self.config.sumo_binary
        if self.config.gui and binary == "sumo":
            binary = "sumo-gui"
        cmd = [
            binary,
            "-c",
            sumocfg.name,
            "--step-length",
            str(self.config.step_length),
            "--time-to-teleport",
            str(self.config.time_to_teleport),
            "--seed",
            str(self.config.seed),
            "--quit-on-end",
            "true",
            "--no-step-log",
            "true",
        ]
        if self.config.no_warnings:
            cmd.extend(["--no-warnings", "true"])
        if self.config.sumo_threads and self.config.sumo_threads > 1:
            cmd.extend(["--threads", str(self.config.sumo_threads)])
        if self.config.gui:
            cmd.append("--start")
            if self.config.gui_delay_ms > 0:
                cmd.extend(["--delay", str(self.config.gui_delay_ms)])
        if self.config.write_xml:
            cmd.extend([
                "--tripinfo-output",
                str(self._tripinfo_path),
                "--summary-output",
                str(self._summary_path),
                "--statistic-output",
                str(self._statistic_path),
            ])

        port = free_port()
        old_cwd = Path.cwd()
        try:
            with self._stdout_path.open("w", encoding="utf-8") as stdout, self._stderr_path.open("w", encoding="utf-8") as stderr:
                os.chdir(scenario_dir)
                start_kwargs = {"port": port, "label": self.label, "stdout": stdout}
                if "stderr" in inspect.signature(traci.start).parameters:
                    start_kwargs["stderr"] = stderr
                else:
                    stderr.write("TraCI start() does not support stderr redirection in this installed version.\n")
                traci.start(cmd, **start_kwargs)
            self.conn = traci.getConnection(self.label)
        except Exception as exc:
            stdout_tail = _tail(self._stdout_path)
            stderr_tail = _tail(self._stderr_path)
            raise RuntimeError(
                "Failed to start SUMO.\n"
                f"Command: {' '.join(cmd)}\n"
                f"cwd: {scenario_dir}\n"
                f"stdout:\n{stdout_tail}\n"
                f"stderr:\n{stderr_tail}"
            ) from exc
        finally:
            os.chdir(old_cwd)

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close(False)
            except Exception:
                pass
            finally:
                self.conn = None

    def __enter__(self) -> "SumoTSCEnv":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _lane_queue(self, lane_id: str) -> float:
        try:
            return float(self.conn.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            return 0.0

    def _lane_vehicle_count(self, lane_id: str) -> float:
        try:
            return float(self.conn.lane.getLastStepVehicleNumber(lane_id))
        except Exception:
            return 0.0

    def _lane_waiting(self, lane_id: str) -> float:
        try:
            return float(self.conn.lane.getWaitingTime(lane_id))
        except Exception:
            return 0.0

    def _lane_occupancy(self, lane_id: str) -> float:
        try:
            return float(self.conn.lane.getLastStepOccupancy(lane_id)) / 100.0
        except Exception:
            return 0.0

    def _action_lanes(self, agent: dict[str, Any], action: int) -> tuple[list[str], list[str]]:
        if action >= len(agent["actions"]):
            return [], []
        links = agent["actions"][action].get("served_links", [])
        incoming = sorted({link["from_lane"] for link in links})
        outgoing = sorted({link["to_lane"] for link in links})
        return incoming, outgoing

    def _action_observation(self, agent_idx: int, action: int, mask: np.ndarray) -> np.ndarray:
        agent = self.agents[agent_idx]
        if action >= agent["num_actions"]:
            return np.zeros(self.obs_dim, dtype=np.float32)
        incoming, outgoing = self._action_lanes(agent, action)
        incoming_queue = sum(self._lane_queue(lane) for lane in incoming)
        incoming_count = sum(self._lane_vehicle_count(lane) for lane in incoming)
        incoming_wait = sum(self._lane_waiting(lane) for lane in incoming)
        downstream_queue = sum(self._lane_queue(lane) for lane in outgoing)
        downstream_occ = np.mean([self._lane_occupancy(lane) for lane in outgoing], dtype=np.float32) if outgoing else 0.0
        pressure = incoming_queue - downstream_queue
        links = agent["actions"][action].get("served_links", [])
        served_link_count = len(links)
        direction_count = max(1, served_link_count)
        right_turn_ratio = sum(1 for link in links if link.get("direction") == "r") / direction_count
        straight_ratio = sum(1 for link in links if link.get("direction") == "s") / direction_count
        left_turn_ratio = sum(1 for link in links if link.get("direction") in {"l", "L"}) / direction_count
        spillback_indicator = 1.0 if downstream_occ >= 0.75 or downstream_queue >= 10.0 else 0.0
        return np.asarray([
            incoming_queue / 25.0,
            incoming_count / 35.0,
            incoming_wait / 600.0,
            downstream_queue / 25.0,
            downstream_occ,
            pressure / 25.0,
            served_link_count / max(1.0, float(self.max_links_per_action)),
            right_turn_ratio,
            straight_ratio,
            left_turn_ratio,
            spillback_indicator,
            1.0 if self.current_actions[agent_idx] == action else 0.0,
            min(1.5, self.elapsed_green[agent_idx] / max(1.0, float(self.config.max_green))),
            float(mask[agent_idx, action]),
            1.0,
        ], dtype=np.float32)

    def action_mask(self) -> np.ndarray:
        mask = self.base_action_mask.copy()
        for idx in range(self.num_agents):
            current = int(self.current_actions[idx])
            valid = np.flatnonzero(self.base_action_mask[idx] > 0.5)
            if len(valid) == 0:
                mask[idx, current] = 1.0
                continue
            if self.elapsed_green[idx] < self.config.min_green:
                mask[idx, :] = 0.0
                mask[idx, current] = 1.0
            elif self.elapsed_green[idx] >= self.config.max_green and len(valid) > 1:
                mask[idx, current] = 0.0
            if mask[idx].sum() <= 0:
                fallback = current if current in valid else int(valid[0])
                mask[idx, fallback] = 1.0
        return mask

    def get_state(self) -> dict[str, np.ndarray]:
        mask = self.action_mask()
        obs = np.zeros((self.num_agents, self.p_max, self.obs_dim), dtype=np.float32)
        for i in range(self.num_agents):
            for a in range(self.p_max):
                obs[i, a] = self._action_observation(i, a, mask)
        return {
            "obs": obs,
            "action_mask": mask.astype(np.float32),
            "adjacency": self.adjacency.astype(np.float32),
            "current_actions": self.current_actions.copy(),
        }

    def _yellow_state(self, old_state: str, new_state: str) -> str:
        length = max(len(old_state), len(new_state))
        chars: list[str] = []
        for idx in range(length):
            old = old_state[idx] if idx < len(old_state) else "r"
            new = new_state[idx] if idx < len(new_state) else "r"
            old_green = old in "Gg"
            new_green = new in "Gg"
            if old_green and new_green:
                chars.append(old)
            elif old_green and not new_green:
                chars.append("y")
            else:
                chars.append("r")
        return "".join(chars)

    def _simulate_for(self, seconds: float) -> None:
        if self.conn is None:
            raise RuntimeError("SUMO is not running.")
        steps = max(0, int(round(seconds / max(self.config.step_length, 1e-6))))
        for _ in range(steps):
            self.conn.simulationStep()
            try:
                self._departed += int(self.conn.simulation.getDepartedNumber())
                self._arrived += int(self.conn.simulation.getArrivedNumber())
                self._teleports += int(self.conn.simulation.getEndingTeleportNumber())
            except Exception:
                pass
            try:
                self._collisions += int(self.conn.simulation.getCollidingVehiclesNumber())
            except Exception:
                pass
            self.elapsed_green += float(self.config.step_length)
            if self._is_done():
                break

    def _is_done(self) -> bool:
        if self.conn is None:
            return True
        sim_time = float(self.conn.simulation.getTime())
        if sim_time >= float(self.config.sim_max_time):
            return True
        return int(self.conn.simulation.getMinExpectedNumber()) <= 0

    def _sanitize_actions(self, actions: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        sanitized = actions.copy()
        invalid = 0
        for idx, action in enumerate(actions):
            current = int(self.current_actions[idx])
            if action < 0 or action >= self.p_max or mask[idx, action] <= 0.0:
                sanitized[idx] = current
                invalid += 1
        return sanitized, invalid

    def step(self, actions: np.ndarray | list[int]) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        if self.conn is None:
            raise RuntimeError("Call reset before step.")
        mask = self.action_mask()
        target_actions, invalid_count = self._sanitize_actions(np.asarray(actions), mask)
        switch_indices = [idx for idx, action in enumerate(target_actions) if int(action) != int(self.current_actions[idx])]
        switch_count = len(switch_indices)
        tele_before = self._teleports
        collision_before = self._collisions

        if switch_indices:
            for idx in switch_indices:
                agent = self.agents[idx]
                old_state = agent["action_states"][int(self.current_actions[idx])]
                new_state = agent["action_states"][int(target_actions[idx])]
                self.conn.trafficlight.setRedYellowGreenState(agent["tl_id"], self._yellow_state(old_state, new_state))
            self._simulate_for(self.config.yellow_time)
            for idx in switch_indices:
                agent = self.agents[idx]
                new_state = agent["action_states"][int(target_actions[idx])]
                self.conn.trafficlight.setRedYellowGreenState(agent["tl_id"], "r" * len(new_state))
            self._simulate_for(self.config.all_red_time)
            for idx in switch_indices:
                agent = self.agents[idx]
                new_state = agent["action_states"][int(target_actions[idx])]
                self.conn.trafficlight.setRedYellowGreenState(agent["tl_id"], new_state)
                self.current_actions[idx] = int(target_actions[idx])
                self.elapsed_green[idx] = 0.0
            remaining = max(0.0, float(self.config.control_interval - self.config.yellow_time - self.config.all_red_time))
        else:
            remaining = float(self.config.control_interval)
        self._simulate_for(remaining)

        reward, reward_parts = self._reward(
            switch_count=switch_count,
            teleport_delta=self._teleports - tele_before,
            collision_delta=self._collisions - collision_before,
        )
        self._last_switch_count = switch_count
        self._last_invalid_count = invalid_count
        done = self._is_done()
        state = self.get_state()
        info = self._info(reward_parts, switch_count, invalid_count)
        return state, reward, done, info

    def _reward(self, switch_count: int, teleport_delta: int, collision_delta: int) -> tuple[float, dict[str, float]]:
        incoming_lanes = sorted({lane for agent in self.agents for lane in agent["incoming_lanes"]})
        outgoing_lanes = sorted({lane for agent in self.agents for lane in agent["outgoing_lanes"]})
        queue_values = [self._lane_queue(lane) for lane in incoming_lanes]
        waiting_values = [self._lane_waiting(lane) for lane in incoming_lanes]
        downstream_occ = [self._lane_occupancy(lane) for lane in outgoing_lanes]
        pressure_values = []
        for idx, agent in enumerate(self.agents):
            incoming, outgoing = self._action_lanes(agent, int(self.current_actions[idx]))
            pressure_values.append(sum(self._lane_queue(l) for l in incoming) - sum(self._lane_queue(l) for l in outgoing))
        queue_mean = float(np.mean(queue_values)) if queue_values else 0.0
        waiting_mean = float(np.mean(waiting_values)) if waiting_values else 0.0
        pressure_abs = float(np.mean(np.abs(pressure_values))) if pressure_values else 0.0
        spillback = float(np.mean(downstream_occ)) if downstream_occ else 0.0
        reward = -(
            0.08 * queue_mean
            + 0.002 * waiting_mean
            + 0.04 * pressure_abs
            + 0.25 * spillback
            + 0.12 * switch_count
            + 4.0 * teleport_delta
            + 8.0 * collision_delta
        )
        return float(reward), {
            "queue_total": float(sum(queue_values)),
            "waiting_total": float(sum(waiting_values)),
            "pressure_total": float(sum(abs(v) for v in pressure_values)),
            "queue_mean": queue_mean,
            "pressure_mean": pressure_abs,
            "spillback_mean": spillback,
        }

    def _info(self, reward_parts: dict[str, float], switch_count: int, invalid_action_count: int) -> dict[str, Any]:
        sim_time = float(self.conn.simulation.getTime()) if self.conn is not None else 0.0
        min_expected = int(self.conn.simulation.getMinExpectedNumber()) if self.conn is not None else 0
        return {
            "sim_time": sim_time,
            "min_expected": min_expected,
            "departed": self._departed,
            "arrived": self._arrived,
            "running": max(0, min_expected),
            "teleports": self._teleports,
            "collisions": self._collisions,
            "queue_total": reward_parts["queue_total"],
            "waiting_total": reward_parts["waiting_total"],
            "pressure_total": reward_parts["pressure_total"],
            "switch_count": switch_count,
            "invalid_action_count": invalid_action_count,
            "tripinfo_path": str(self._tripinfo_path) if self._tripinfo_path else "",
            "summary_path": str(self._summary_path) if self._summary_path else "",
            "statistic_path": str(self._statistic_path) if self._statistic_path else "",
        }
