from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .models import ActorCritic
from .net import load_metadata, sha256_file
from .policies import MaxPressurePolicy
from .scenario import load_scenario_index
from .sumo_env import OBS_FEATURES, EnvConfig, SumoTSCEnv


@dataclass
class TrainConfig:
    map_root: str | Path = "map"
    metadata_path: str | Path = "map/metadata/network_metadata.json"
    scenario_index: str | Path = "map/scenarios/scenario_index.csv"
    output_dir: str | Path = "results/proposed/train"
    device: str = "auto"
    seed: int = 42
    total_updates: int = 500
    rollout_steps: int = 512
    lr: float = 5e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    value_coef: float = 0.1
    entropy_coef: float = 0.003
    entropy_coef_final: float = 0.001
    entropy_decay_fraction: float = 0.70
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden: int = 256
    graph_layers: int = 3
    control_interval: int = 10
    min_green: int = 20
    max_green: int = 90
    sim_max_time: int = 7200
    bc_scenarios: int = 18
    bc_epochs: int = 12
    bc_batch_size: int = 256
    bc_lr: float = 1e-3
    resume: str | Path | None = None
    overwrite: bool = False
    torch_threads: int = 1
    write_train_xml: bool = False
    sumo_binary: str = "sumo"
    gui: bool = False
    gui_delay_ms: int = 0
    yellow_time: int = 3
    all_red_time: int = 1
    checkpoint_interval: int = 10


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: str | Path, map_location: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def checkpoint_config(config: TrainConfig, obs_dim: int) -> dict[str, Any]:
    payload = asdict(config)
    payload["obs_dim"] = obs_dim
    payload["output_dir"] = str(payload["output_dir"])
    payload["map_root"] = str(payload["map_root"])
    payload["metadata_path"] = str(payload["metadata_path"])
    payload["scenario_index"] = str(payload["scenario_index"])
    payload["resume"] = str(payload["resume"]) if payload["resume"] else None
    return payload


def make_env(config: TrainConfig, metadata: dict[str, Any], scenario: dict[str, Any], output_dir: Path) -> SumoTSCEnv:
    return SumoTSCEnv(
        EnvConfig(
            net_file=metadata["net_file"],
            metadata_path=config.metadata_path,
            sumo_binary=config.sumo_binary,
            gui=config.gui,
            gui_delay_ms=config.gui_delay_ms,
            control_interval=config.control_interval,
            min_green=config.min_green,
            max_green=config.max_green,
            yellow_time=config.yellow_time,
            all_red_time=config.all_red_time,
            sim_max_time=config.sim_max_time,
            seed=int(scenario["seed"]),
            output_dir=output_dir,
            write_xml=config.write_train_xml,
        ),
        scenario,
    )


def bc_dataset_metadata(config: TrainConfig, metadata_hash: str, scenario_index_hash: str) -> dict[str, Any]:
    return {
        "metadata_hash": metadata_hash,
        "scenario_index_hash": scenario_index_hash,
        "bc_scenarios": config.bc_scenarios,
        "control_interval": config.control_interval,
        "min_green": config.min_green,
        "max_green": config.max_green,
        "sim_max_time": config.sim_max_time,
        "observation_features": OBS_FEATURES,
    }


def load_bc_dataset(path: Path, expected_meta: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(str(data["meta"].item()))
        if meta != expected_meta:
            return None
        return {
            "obs": data["obs"],
            "mask": data["mask"],
            "adj": data["adj"],
            "actions": data["actions"],
        }


def collect_bc_dataset(
    config: TrainConfig,
    metadata: dict[str, Any],
    scenarios: list[dict[str, Any]],
    metadata_hash: str,
    scenario_index_hash: str,
) -> dict[str, np.ndarray]:
    output = Path(config.output_dir)
    dataset_path = output / "bc_dataset.npz"
    expected_meta = bc_dataset_metadata(config, metadata_hash, scenario_index_hash)
    loaded = load_bc_dataset(dataset_path, expected_meta)
    if loaded is not None:
        return loaded

    rng = random.Random(config.seed)
    selected = list(scenarios)
    rng.shuffle(selected)
    selected = selected[: max(1, min(config.bc_scenarios, len(selected)))]
    teacher = MaxPressurePolicy(metadata)
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    adj_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    for record in tqdm(selected, desc="collect BC teacher"):
        rollout_dir = output / "bc_rollouts" / str(record["family"]) / f"seed_{int(record['seed'])}"
        env = make_env(config, metadata, record, rollout_dir)
        try:
            state = env.reset()
            done = False
            while not done:
                actions = teacher.act(state, env)
                obs_list.append(state["obs"].astype(np.float32))
                mask_list.append(state["action_mask"].astype(np.float32))
                adj_list.append(state["adjacency"].astype(np.float32))
                action_list.append(actions.astype(np.int64))
                state, _, done, _ = env.step(actions)
        finally:
            env.close()

    if not obs_list:
        raise RuntimeError("BC teacher collection produced no samples.")
    dataset = {
        "obs": np.stack(obs_list),
        "mask": np.stack(mask_list),
        "adj": np.stack(adj_list),
        "actions": np.stack(action_list),
    }
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_path, **dataset, meta=json.dumps(expected_meta, sort_keys=True))
    return dataset


def train_bc(
    model: ActorCritic,
    config: TrainConfig,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    metadata_hash: str,
    scenario_index_hash: str,
) -> tuple[float, float]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.bc_lr)
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(dataset["mask"], dtype=torch.float32, device=device)
    adj = torch.as_tensor(dataset["adj"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(dataset["actions"], dtype=torch.long, device=device)
    n = obs.shape[0]
    last_loss = 0.0
    last_acc = 0.0
    for _ in tqdm(range(config.bc_epochs), desc="BC warm-start"):
        perm = torch.randperm(n, device=device)
        correct = 0
        total = 0
        loss_total = 0.0
        for start in range(0, n, config.bc_batch_size):
            idx = perm[start:start + config.bc_batch_size]
            logits, _ = model(obs[idx], mask[idx], adj[idx])
            loss = F.cross_entropy(logits.reshape(-1, model.p_max), actions[idx].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct += int((pred == actions[idx]).sum().item())
                total += int(actions[idx].numel())
                loss_total += float(loss.item()) * len(idx)
        last_loss = loss_total / max(1, n)
        last_acc = correct / max(1, total)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": checkpoint_config(config, model.obs_dim),
        "metadata_hash": metadata_hash,
        "scenario_index_hash": scenario_index_hash,
        "bc_loss": last_loss,
        "bc_accuracy": last_acc,
    }
    out = Path(config.output_dir) / "checkpoints" / "bc_init.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out)
    return last_loss, last_acc


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / var_y)


def entropy_coef_for_update(config: TrainConfig, update: int) -> float:
    if config.total_updates <= 1:
        return float(config.entropy_coef_final)
    progress = (update - 1) / max(1, config.total_updates - 1)
    decay_fraction = min(1.0, max(1e-6, float(config.entropy_decay_fraction)))
    alpha = min(1.0, progress / decay_fraction)
    return float(config.entropy_coef + alpha * (config.entropy_coef_final - config.entropy_coef))


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    metadata_hash: str,
    scenario_index_hash: str,
    update: int,
    best_train_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": checkpoint_config(config, model.obs_dim),
            "metadata_hash": metadata_hash,
            "scenario_index_hash": scenario_index_hash,
            "update": update,
            "best_train_score": best_train_score,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        path,
    )


def restore_rng(checkpoint: dict[str, Any]) -> None:
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_random_state"])
    if torch.cuda.is_available() and checkpoint.get("torch_cuda_random_state"):
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state"])


def append_train_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "update",
        "reward_mean",
        "reward_sum",
        "policy_loss",
        "value_loss",
        "entropy",
        "entropy_coef",
        "approx_kl",
        "explained_variance",
        "learning_rate",
        "rollout_episodes_completed",
        "teleports_last",
        "forced_switches_last",
        "invalid_actions_last",
        "arrivals_last",
        "waiting_delta_last",
        "waiting_growth_last",
        "spillback_mean_last",
        "unserved_wait_mean_last",
        "max_action_share_mean",
        "max_action_share_max",
        "elapsed_seconds",
        "bc_loss",
        "bc_accuracy",
    ]
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != fields:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({key: old_row.get(key, "") for key in fields})
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not path.exists() or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def append_action_audit(
    path: Path,
    update: int,
    metadata: dict[str, Any],
    action_counts: np.ndarray,
    switch_counts: np.ndarray,
    mean_green_time: np.ndarray,
) -> tuple[float, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "update",
        "agent_index",
        "tl_id",
        "action_counts",
        "switch_count",
        "mean_green_time",
        "max_action_share",
    ]
    exists = path.exists()
    max_shares: list[float] = []
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        for idx, counts in enumerate(action_counts):
            total = int(np.sum(counts))
            share = float(np.max(counts) / max(1, total))
            max_shares.append(share)
            agent = metadata["agents"][idx]
            writer.writerow({
                "update": update,
                "agent_index": idx,
                "tl_id": agent["tl_id"],
                "action_counts": json.dumps([int(v) for v in counts.tolist()]),
                "switch_count": int(switch_counts[idx]),
                "mean_green_time": float(mean_green_time[idx]),
                "max_action_share": share,
            })
    return float(np.mean(max_shares)) if max_shares else 0.0, float(np.max(max_shares)) if max_shares else 0.0


def train(config: TrainConfig) -> Path:
    torch.set_num_threads(config.torch_threads)
    device = resolve_device(config.device)
    set_seeds(config.seed)
    output = Path(config.output_dir)
    if config.overwrite and config.resume:
        raise ValueError("overwrite=True cannot be used together with resume.")
    if config.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(config.metadata_path)
    metadata_hash = sha256_file(config.metadata_path)
    scenario_index_hash = sha256_file(config.scenario_index)
    scenarios = load_scenario_index(config.scenario_index, split="train")
    if not scenarios:
        raise RuntimeError("No train scenarios found. Run scripts/build_scenarios.py first.")

    obs_dim = len(OBS_FEATURES)
    model = ActorCritic(
        num_agents=int(metadata["num_agents"]),
        p_max=int(metadata["p_max"]),
        obs_dim=obs_dim,
        hidden=config.hidden,
        graph_layers=config.graph_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    start_update = 1
    best_train_score = -float("inf")
    bc_loss = ""
    bc_accuracy = ""

    if config.resume:
        checkpoint = torch_load(config.resume, map_location=device)
        if checkpoint.get("metadata_hash") != metadata_hash:
            raise RuntimeError("Checkpoint metadata_hash does not match current metadata file.")
        if checkpoint.get("scenario_index_hash") != scenario_index_hash:
            raise RuntimeError("Checkpoint scenario_index_hash does not match current scenario_index.csv.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        restore_rng(checkpoint)
        start_update = int(checkpoint.get("update", 0)) + 1
        best_train_score = float(checkpoint.get("best_train_score", best_train_score))
    else:
        dataset = collect_bc_dataset(config, metadata, scenarios, metadata_hash, scenario_index_hash)
        bc_loss, bc_accuracy = train_bc(model, config, dataset, device, metadata_hash, scenario_index_hash)

    current_record = random.choice(scenarios)
    env = make_env(config, metadata, current_record, output / "ppo_rollouts" / str(current_record["family"]) / f"seed_{int(current_record['seed'])}")
    state = env.reset()
    log_path = output / "train_log.csv"

    try:
        for update in range(start_update, config.total_updates + 1):
            update_start = time.time()
            obs_buf: list[np.ndarray] = []
            mask_buf: list[np.ndarray] = []
            adj_buf: list[np.ndarray] = []
            action_buf: list[np.ndarray] = []
            logprob_buf: list[float] = []
            value_buf: list[float] = []
            reward_buf: list[float] = []
            done_buf: list[float] = []
            episodes_completed = 0
            invalid_actions_last = 0
            forced_switches_last = 0
            arrivals_last = 0.0
            waiting_delta_last = 0.0
            waiting_growth_last = 0.0
            spillback_values: list[float] = []
            unserved_wait_values: list[float] = []
            action_counts = np.zeros((int(metadata["num_agents"]), int(metadata["p_max"])), dtype=np.int64)
            switch_counts = np.zeros(int(metadata["num_agents"]), dtype=np.int64)
            green_time_sum = np.zeros(int(metadata["num_agents"]), dtype=np.float64)
            green_time_samples = 0
            last_info: dict[str, Any] = {}

            for _ in tqdm(range(config.rollout_steps), desc=f"PPO rollout {update}", leave=False):
                green_time_sum += env.elapsed_green.astype(np.float64)
                green_time_samples += 1
                obs_t = torch.as_tensor(state["obs"], dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(state["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                adj_t = torch.as_tensor(state["adjacency"], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    actions_t, logprob_t, _, value_t = model.act(obs_t, mask_t, adj_t, deterministic=False)
                actions = actions_t.squeeze(0).detach().cpu().numpy().astype(np.int64)
                for agent_idx, action in enumerate(actions):
                    if 0 <= int(action) < action_counts.shape[1]:
                        action_counts[agent_idx, int(action)] += 1

                next_state, reward, done, info = env.step(actions)
                obs_buf.append(state["obs"].astype(np.float32))
                mask_buf.append(state["action_mask"].astype(np.float32))
                adj_buf.append(state["adjacency"].astype(np.float32))
                action_buf.append(actions)
                logprob_buf.append(float(logprob_t.item()))
                value_buf.append(float(value_t.item()))
                reward_buf.append(float(reward))
                done_buf.append(1.0 if done else 0.0)
                last_info = info
                invalid_actions_last += int(info.get("invalid_action_count", 0))
                forced_switches_last += int(info.get("forced_switch_count", 0))
                arrivals_last += float(info.get("arrived_delta", 0.0))
                waiting_delta_last += float(info.get("waiting_delta", 0.0))
                waiting_growth_last += float(info.get("waiting_growth_rate", 0.0))
                spillback_values.append(float(info.get("spillback_mean", 0.0)))
                unserved_wait_values.append(float(info.get("unserved_wait_mean", 0.0)))
                per_agent_switches = np.asarray(info.get("per_agent_switches", np.zeros_like(switch_counts)), dtype=np.int64)
                if per_agent_switches.shape == switch_counts.shape:
                    switch_counts += per_agent_switches
                state = next_state
                if done:
                    episodes_completed += 1
                    env.close()
                    current_record = random.choice(scenarios)
                    env = make_env(config, metadata, current_record, output / "ppo_rollouts" / str(current_record["family"]) / f"seed_{int(current_record['seed'])}")
                    state = env.reset()

            with torch.no_grad():
                last_obs = torch.as_tensor(state["obs"], dtype=torch.float32, device=device).unsqueeze(0)
                last_mask = torch.as_tensor(state["action_mask"], dtype=torch.float32, device=device).unsqueeze(0)
                last_adj = torch.as_tensor(state["adjacency"], dtype=torch.float32, device=device).unsqueeze(0)
                _, last_value_t = model(last_obs, last_mask, last_adj)
                last_value = float(last_value_t.item())

            rewards = np.asarray(reward_buf, dtype=np.float32)
            dones = np.asarray(done_buf, dtype=np.float32)
            values = np.asarray(value_buf + [last_value], dtype=np.float32)
            advantages = np.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(config.rollout_steps)):
                next_nonterminal = 1.0 - dones[t]
                delta = rewards[t] + config.gamma * values[t + 1] * next_nonterminal - values[t]
                last_gae = delta + config.gamma * config.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values[:-1]

            obs = torch.as_tensor(np.stack(obs_buf), dtype=torch.float32, device=device)
            mask = torch.as_tensor(np.stack(mask_buf), dtype=torch.float32, device=device)
            adj = torch.as_tensor(np.stack(adj_buf), dtype=torch.float32, device=device)
            actions = torch.as_tensor(np.stack(action_buf), dtype=torch.long, device=device)
            old_logprob = torch.as_tensor(logprob_buf, dtype=torch.float32, device=device)
            adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
            old_values = torch.as_tensor(value_buf, dtype=torch.float32, device=device)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            approx_kls: list[float] = []
            batch_size = config.rollout_steps
            current_entropy_coef = entropy_coef_for_update(config, update)
            for _ in range(config.update_epochs):
                perm = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, config.minibatch_size):
                    idx = perm[start:start + config.minibatch_size]
                    new_logprob, entropy, new_value = model.evaluate_actions(obs[idx], mask[idx], actions[idx], adj[idx])
                    logratio = new_logprob - old_logprob[idx]
                    ratio = torch.exp(logratio)
                    pg_loss1 = -adv[idx] * ratio
                    pg_loss2 = -adv[idx] * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    value_loss = 0.5 * F.mse_loss(new_value, ret[idx])
                    entropy_loss = entropy.mean()
                    loss = policy_loss + config.value_coef * value_loss - current_entropy_coef * entropy_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy_loss.item()))
                    approx_kls.append(float(approx_kl.item()))

            reward_mean = float(rewards.mean()) if len(rewards) else 0.0
            reward_sum = float(rewards.sum()) if len(rewards) else 0.0
            ev = explained_variance(old_values.detach().cpu().numpy(), returns)
            if reward_mean > best_train_score:
                best_train_score = reward_mean
                save_checkpoint(output / "checkpoints" / "best_train.pt", model, optimizer, config, metadata_hash, scenario_index_hash, update, best_train_score)
            save_checkpoint(output / "checkpoints" / "last.pt", model, optimizer, config, metadata_hash, scenario_index_hash, update, best_train_score)
            if update % config.checkpoint_interval == 0:
                save_checkpoint(output / "checkpoints" / f"update_{update:04d}.pt", model, optimizer, config, metadata_hash, scenario_index_hash, update, best_train_score)

            mean_green_time = green_time_sum / max(1, green_time_samples)
            max_action_share_mean, max_action_share_max = append_action_audit(
                output / "action_audit.csv",
                update,
                metadata,
                action_counts,
                switch_counts,
                mean_green_time,
            )
            append_train_log(log_path, {
                "update": update,
                "reward_mean": reward_mean,
                "reward_sum": reward_sum,
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
                "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "entropy_coef": current_entropy_coef,
                "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                "explained_variance": ev,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "rollout_episodes_completed": episodes_completed,
                "teleports_last": int(last_info.get("teleports", 0)),
                "forced_switches_last": forced_switches_last,
                "invalid_actions_last": invalid_actions_last,
                "arrivals_last": arrivals_last,
                "waiting_delta_last": waiting_delta_last,
                "waiting_growth_last": waiting_growth_last,
                "spillback_mean_last": float(np.mean(spillback_values)) if spillback_values else 0.0,
                "unserved_wait_mean_last": float(np.mean(unserved_wait_values)) if unserved_wait_values else 0.0,
                "max_action_share_mean": max_action_share_mean,
                "max_action_share_max": max_action_share_max,
                "elapsed_seconds": time.time() - update_start,
                "bc_loss": bc_loss if update == start_update else "",
                "bc_accuracy": bc_accuracy if update == start_update else "",
            })
    finally:
        env.close()

    return output / "checkpoints" / "last.pt"
