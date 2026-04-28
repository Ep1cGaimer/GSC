"""
Adversarial Agent: Minimax adversary that perturbs environment dynamics
to discover vulnerabilities in the protagonist policy.

The adversary has a disruption budget per episode and must allocate it
across perturbation types. This prevents brute-force "shut everything down"
and forces discovery of targeted vulnerabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
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
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        self.alpha_head = nn.Linear(hidden_dim, self.action_dim)
        self.beta_head = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, obs: torch.Tensor):
        """Generate perturbation parameters."""
        h = self.encoder(obs)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        return alpha, beta

    def action_to_disruption(self, action: torch.Tensor,
                             retailer_ids: list, edge_keys: list,
                             node_ids: list) -> dict:
        """Convert a raw action tensor into a disruption dict.
        
        This method does NOT resample — it uses the exact action passed in,
        ensuring the gradient flows through the correct action.
        """
        raw_action = action.detach().cpu().numpy().flatten()

        # Split into perturbation types
        demand_raw = raw_action[:self.num_demand]
        edge_raw = raw_action[self.num_demand:self.num_demand + self.num_edges]
        capacity_raw = raw_action[self.num_demand + self.num_edges:]

        # Compute disruption cost for budget constraint
        demand_costs = np.abs(demand_raw - 0.5) * 2
        edge_costs = (edge_raw > 0.5).astype(float) * 1.5
        capacity_costs = (1.0 - capacity_raw) * 1.0

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

        for i, rid in enumerate(retailer_ids[:self.num_demand]):
            mult = 0.5 + demand_raw[i] * 1.5  # [0.5, 2.0]
            if abs(mult - 1.0) > 0.1:
                disruption["demand_multipliers"][rid] = float(mult)

        for i, ekey in enumerate(edge_keys[:self.num_edges]):
            if edge_raw[i] > 0.5:
                parts = ekey.split("->")
                if len(parts) == 2:
                    disruption["disabled_edges"].append(parts)

        for i, nid in enumerate(node_ids[:self.num_capacity]):
            mult = 0.3 + capacity_raw[i] * 0.7  # [0.3, 1.0]
            if mult < 0.9:
                disruption["capacity_multipliers"][nid] = float(mult)

        return disruption

    def get_action_and_logprob(self, obs: torch.Tensor, action=None, deterministic: bool = False):
        """For training: get action and log probability."""
        alpha, beta = self.forward(obs)
        dist = Beta(alpha, beta)

        if action is None:
            if deterministic:
                action = alpha / (alpha + beta)
            else:
                action = dist.rsample()

        action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy
