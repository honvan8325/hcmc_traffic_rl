from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import ActorCritic
from .net import load_metadata
from .sumo_env import OBS_FEATURES


class FixedPolicy:
    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        self.num_agents = int(metadata["num_agents"])
        self.p_max = int(metadata["p_max"])
        self.target_duration = np.zeros((self.num_agents, self.p_max), dtype=np.float32)
        for idx, agent in enumerate(metadata.get("agents", [])):
            for action_idx, action in enumerate(agent.get("actions", [])):
                if action_idx >= self.p_max:
                    continue
                try:
                    self.target_duration[idx, action_idx] = max(0.0, float(action.get("duration", 0.0)))
                except (TypeError, ValueError):
                    self.target_duration[idx, action_idx] = 0.0

    def _next_allowed(self, mask_row: np.ndarray, current: int) -> int:
        allowed = np.flatnonzero(mask_row > 0.5)
        if len(allowed) == 0:
            return current
        if len(allowed) == 1:
            return int(allowed[0])
        nxt = (current + 1) % self.p_max
        for _ in range(self.p_max):
            if mask_row[nxt] > 0.5:
                return int(nxt)
            nxt = (nxt + 1) % self.p_max
        return int(allowed[0])

    def act(self, state: dict[str, np.ndarray], env: Any | None = None) -> np.ndarray:
        mask = state["action_mask"]
        current = state["current_actions"]
        elapsed = state.get("elapsed_green")
        if elapsed is None:
            elapsed = np.zeros(self.num_agents, dtype=np.float32)
        actions = np.zeros(self.num_agents, dtype=np.int64)
        for idx in range(self.num_agents):
            cur = int(current[idx])
            if cur < 0 or cur >= self.p_max:
                actions[idx] = self._next_allowed(mask[idx], 0)
                continue
            current_allowed = mask[idx, cur] > 0.5
            target = float(self.target_duration[idx, cur])
            if current_allowed and float(elapsed[idx]) < target:
                actions[idx] = cur
                continue
            actions[idx] = self._next_allowed(mask[idx], cur)
        return actions


class MaxPressurePolicy:
    def __init__(self, metadata: dict[str, Any], spillback_threshold: float = 0.72, switch_penalty: float = 0.35, starvation_bonus: float = 0.08):
        self.metadata = metadata
        self.num_agents = int(metadata["num_agents"])
        self.p_max = int(metadata["p_max"])
        self.spillback_threshold = spillback_threshold
        self.switch_penalty = switch_penalty
        self.starvation_bonus = starvation_bonus
        self.starvation = np.zeros((self.num_agents, self.p_max), dtype=np.float32)

    def act(self, state: dict[str, np.ndarray], env: Any | None = None) -> np.ndarray:
        obs = state["obs"]
        mask = state["action_mask"]
        current = state["current_actions"]
        actions = np.zeros(self.num_agents, dtype=np.int64)
        pressure_idx = OBS_FEATURES.index("pressure")
        spill_idx = OBS_FEATURES.index("downstream_occupancy")
        queue_idx = OBS_FEATURES.index("incoming_queue")
        wait_idx = OBS_FEATURES.index("incoming_waiting_time")
        for idx in range(self.num_agents):
            allowed = np.flatnonzero(mask[idx] > 0.5)
            if len(allowed) == 0:
                actions[idx] = int(current[idx])
                continue
            best_action = int(allowed[0])
            best_score = -np.inf
            for action in allowed:
                pressure = float(obs[idx, action, pressure_idx]) * 25.0
                incoming_queue = float(obs[idx, action, queue_idx]) * 25.0
                waiting = float(obs[idx, action, wait_idx]) * 600.0
                spill = float(obs[idx, action, spill_idx])
                spill_penalty = 4.0 * max(0.0, spill - self.spillback_threshold)
                change_penalty = self.switch_penalty if int(action) != int(current[idx]) else 0.0
                starvation = self.starvation_bonus * float(self.starvation[idx, action])
                score = pressure + 0.25 * incoming_queue + 0.002 * waiting - spill_penalty - change_penalty + starvation
                if score > best_score or (score == best_score and int(action) < best_action):
                    best_score = score
                    best_action = int(action)
            actions[idx] = best_action
        self.starvation += 1.0
        for idx, action in enumerate(actions):
            self.starvation[idx, int(action)] = 0.0
        return actions


class CheckpointPolicy:
    def __init__(
        self,
        metadata_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        self.metadata = load_metadata(metadata_path)
        self.device = torch.device(device)
        self.deterministic = deterministic
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        obs_dim = int(config.get("obs_dim", len(OBS_FEATURES)))
        hidden = int(config.get("hidden", 128))
        graph_layers = int(config.get("graph_layers", 2))
        self.model = ActorCritic(
            num_agents=int(self.metadata["num_agents"]),
            p_max=int(self.metadata["p_max"]),
            obs_dim=obs_dim,
            hidden=hidden,
            graph_layers=graph_layers,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def act(self, state: dict[str, np.ndarray], env: Any | None = None) -> np.ndarray:
        with torch.no_grad():
            obs = torch.as_tensor(state["obs"], dtype=torch.float32, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(state["action_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
            adj = torch.as_tensor(state["adjacency"], dtype=torch.float32, device=self.device).unsqueeze(0)
            actions, _, _, _ = self.model.act(obs, mask, adj, deterministic=self.deterministic)
        return actions.squeeze(0).detach().cpu().numpy().astype(np.int64)
