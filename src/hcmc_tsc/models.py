from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


class DirectedAttentionBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.ReLU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, node: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, n, hidden = node.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch, -1, -1)
        eye = torch.eye(n, dtype=torch.bool, device=node.device).unsqueeze(0)
        directed_neighborhood = (adj.transpose(1, 2) > 0.5) | eye
        q = self.q(node)
        k = self.k(node)
        v = self.v(node)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(hidden)
        scores = scores.masked_fill(~directed_neighborhood, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attended = self.out(torch.matmul(attn, v))
        node = self.norm1(node + attended)
        node = self.norm2(node + self.ff(node))
        return node


class ActorCritic(nn.Module):
    def __init__(self, num_agents: int, p_max: int, obs_dim: int, hidden: int = 128, graph_layers: int = 2):
        super().__init__()
        self.num_agents = num_agents
        self.p_max = p_max
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.graph_layers = graph_layers
        self.action_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.agent_embedding = nn.Embedding(num_agents, hidden)
        self.graph = nn.ModuleList([DirectedAttentionBlock(hidden) for _ in range(graph_layers)])
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(num_agents * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _ensure_batch(self, obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        squeezed = False
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
            adj = adj.unsqueeze(0)
            squeezed = True
        return obs, mask, adj, squeezed

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs, mask, adj, _ = self._ensure_batch(obs, mask, adj)
        batch = obs.shape[0]
        action_h = self.action_encoder(obs)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        node = (action_h * mask_f).sum(dim=2) / denom
        agent_ids = torch.arange(self.num_agents, device=obs.device)
        node = node + self.agent_embedding(agent_ids).unsqueeze(0)
        for layer in self.graph:
            node = layer(node, adj)
        action_context = torch.tanh(action_h + node.unsqueeze(2))
        logits = self.actor(action_context).squeeze(-1)
        logits = logits.masked_fill(mask <= 0.0, -1e9)
        value = self.critic(node.reshape(batch, self.num_agents * self.hidden)).squeeze(-1)
        return logits, value

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        adj: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, mask, adj, squeezed = self._ensure_batch(obs, mask, adj)
        logits, value = self.forward(obs, mask, adj)
        batch = logits.shape[0]
        dist = Categorical(logits=logits.reshape(batch * self.num_agents, self.p_max))
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample().reshape(batch, self.num_agents)
        log_prob = dist.log_prob(actions.reshape(batch * self.num_agents)).reshape(batch, self.num_agents).sum(dim=-1)
        entropy = dist.entropy().reshape(batch, self.num_agents).sum(dim=-1)
        if squeezed:
            return actions.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(0)
        return actions, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs, mask, adj, squeezed = self._ensure_batch(obs, mask, adj)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        logits, value = self.forward(obs, mask, adj)
        batch = logits.shape[0]
        dist = Categorical(logits=logits.reshape(batch * self.num_agents, self.p_max))
        log_prob = dist.log_prob(actions.reshape(batch * self.num_agents)).reshape(batch, self.num_agents).sum(dim=-1)
        entropy = dist.entropy().reshape(batch, self.num_agents).sum(dim=-1)
        if squeezed:
            return log_prob.squeeze(0), entropy.squeeze(0), value.squeeze(0)
        return log_prob, entropy, value

