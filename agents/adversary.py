"""
Adversarial Agent: Minimax adversary that perturbs environment dynamics
to discover vulnerabilities in the protagonist policy.

The adversary has a disruption budget per episode and must allocate it
across perturbation types. This prevents brute-force "shut everything down"
and forces discovery of targeted vulnerabilities.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class AdversaryPolicy(nn.Module):
    """MLP-based adversary that generates environment perturbations.
    
    Action space (continuous, bounded):
    - demand_multipliers: [0.5, 2.0] per retailer (scaled subset)
    - lead_time_multipliers: [1.0, 3.0] per edge (scaled subset)
    - edge_disable_probs: [0, 1] per edge (thresholded at 0.5)
    - capacity_multipliers: [0.3, 1.0] per node
    
    All subject to a total disruption budget constraint.
    """

    def __init__(self, obs_dim: int, num_demand_targets: int = 5,
                 num_edge_targets: int = 5, num_capacity_targets: int = 5,
                 hidden_dim: int = 128, disruption_budget: float = 5.0):
        super().__init__()

        self.num_demand = num_demand_targets
        self.num_edges = num_edge_targets
        self.num_capacity = num_capacity_targets
        self.disruption_budget = disruption_budget

        # Total action dimension
        self.action_dim = num_demand_targets + num_edge_targets + num_capacity_targets

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim) - 1.0)

    def forward(self, obs: torch.Tensor):
        """Generate perturbation parameters."""
        h = self.encoder(obs)
        mean = torch.sigmoid(self.mean_head(h))  # Raw [0, 1]
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_disruption(self, obs: torch.Tensor,
                       retailer_ids: list, edge_keys: list,
                       node_ids: list) -> dict:
        """Generate a disruption dict compatible with SupplyChainEnv.inject_disruption().
        
        Applies disruption budget: total perturbation magnitude capped.
        """
        with torch.no_grad():
            mean, log_std = self.forward(obs)
            std = log_std.exp()
            dist = Normal(mean, std)
            raw_action = dist.sample()
            raw_action = torch.clamp(raw_action, 0.0, 1.0).numpy().flatten()

        # Split into perturbation types
        demand_raw = raw_action[:self.num_demand]
        edge_raw = raw_action[self.num_demand:self.num_demand + self.num_edges]
        capacity_raw = raw_action[self.num_demand + self.num_edges:]

        # Compute disruption cost for budget constraint
        # Each perturbation type has a "cost" proportional to severity
        demand_costs = np.abs(demand_raw - 0.5) * 2  # Deviation from neutral
        edge_costs = (edge_raw > 0.5).astype(float) * 1.5  # Disabling an edge costs more
        capacity_costs = (1.0 - capacity_raw) * 1.0  # Reducing capacity

        total_cost = demand_costs.sum() + edge_costs.sum() + capacity_costs.sum()

        # Scale down if over budget
        if total_cost > self.disruption_budget:
            scale = self.disruption_budget / (total_cost + 1e-8)
            demand_raw = 0.5 + (demand_raw - 0.5) * scale
            edge_raw = edge_raw * scale
            capacity_raw = 1.0 - (1.0 - capacity_raw) * scale

        # Convert to disruption dict
        disruption = {
            "demand_multipliers": {},
            "disabled_edges": [],
            "capacity_multipliers": {},
            "lead_time_multipliers": {},
        }

        # Demand multipliers: map [0, 1] → [0.5, 2.0]
        for i, rid in enumerate(retailer_ids[:self.num_demand]):
            mult = 0.5 + demand_raw[i] * 1.5  # [0.5, 2.0]
            if abs(mult - 1.0) > 0.1:  # Only apply non-trivial perturbations
                disruption["demand_multipliers"][rid] = float(mult)

        # Edge disabling: threshold at 0.5
        for i, ekey in enumerate(edge_keys[:self.num_edges]):
            if edge_raw[i] > 0.5:
                parts = ekey.split("->")
                if len(parts) == 2:
                    disruption["disabled_edges"].append(parts)

        # Capacity multipliers: map [0, 1] → [0.3, 1.0]
        for i, nid in enumerate(node_ids[:self.num_capacity]):
            mult = 0.3 + capacity_raw[i] * 0.7  # [0.3, 1.0]
            if mult < 0.9:  # Only apply meaningful reductions
                disruption["capacity_multipliers"][nid] = float(mult)

        return disruption

    def get_action_and_logprob(self, obs: torch.Tensor, action=None):
        """For training: get action and log probability."""
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()
            action = torch.clamp(action, 0.0, 1.0)

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy
