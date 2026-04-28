"""
Baseline policies for supply chain optimization benchmarking.

Provides rule-based heuristics that the RL policy must beat to prove its value:
- BaseStockPolicy: Classic inventory management using base-stock levels
- RandomPolicy: Uniform random actions (lower bound for comparison)
"""

import numpy as np


class RandomPolicy:
    """Uniform random policy — establishes the floor for RL performance."""

    def __init__(self, action_dim: int, seed: int = 42):
        self.action_dim = action_dim
        self.rng = np.random.RandomState(seed)

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None) -> np.ndarray:
        action = self.rng.uniform(0.0, 1.0, self.action_dim).astype(np.float32)
        if action_mask is not None:
            action = action * action_mask
        return action


class BaseStockPolicy:
    """Base-stock inventory policy for warehouse agents.

    For each warehouse:
      1. Computes base-stock level from downstream retailer demand distribution
         base_stock = mean_daily_demand * lead_time_days + z_score * std_demand * sqrt(lead_time)
      2. Orders up to base-stock from cheapest available suppliers first
      3. Allocates outbound proportionally to expected retailer demand

    The policy outputs actions in the same format as GNNActor, making it a
    drop-in replacement for evaluation comparisons.
    """

    def __init__(self, env, z_score: float = 1.64):
        """
        Args:
            env: SupplyChainEnv instance (used to read topology data)
            z_score: Safety stock z-score (1.64 = 95% service level, 2.33 = 99%)
        """
        self.env = env
        self.z_score = z_score
        self.gb = env.graph_builder
        self.agents = env.possible_agents
        self.max_inbound = env._max_inbound
        self.max_outbound = env._max_outbound

        self._base_stock_levels = {}
        self._inbound_suppliers = {}
        self._outbound_retailers = {}

        for agent_id in self.agents:
            self._compute_params(agent_id)

    def _compute_params(self, agent_id: str):
        inbound = self.gb.get_edges_to(agent_id)
        outbound = self.gb.get_edges_from(agent_id)

        total_daily_demand = 0.0
        total_demand_var = 0.0
        max_lead_time = 0.0
        retailers = []

        for edge in outbound:
            dest = self.gb.get_node_data(edge["to"])
            if dest.get("type") != "retailer":
                continue
            mean = float(dest.get("demand_mean", 100))
            std = float(dest.get("demand_std", 20))
            total_daily_demand += mean
            total_demand_var += std ** 2
            max_lead_time = max(max_lead_time, float(edge.get("lead_time_days", 1)))
            retailers.append((edge, mean))

        total_demand_std = np.sqrt(total_demand_var)

        base_stock = (
            total_daily_demand * max_lead_time
            + self.z_score * total_demand_std * np.sqrt(max(max_lead_time, 1))
        )

        self._base_stock_levels[agent_id] = base_stock
        self._outbound_retailers[agent_id] = retailers

        suppliers = []
        for edge in inbound:
            src = self.gb.get_node_data(edge["from"])
            suppliers.append((edge, float(src.get("capacity", 100)),
                              float(edge.get("cost_per_unit", 1.0))))
        suppliers.sort(key=lambda x: x[2])
        self._inbound_suppliers[agent_id] = suppliers

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray = None,
                   agent_id: str = None) -> np.ndarray:
        """Compute base-stock action for a single agent.

        Args:
            obs: Flat observation vector [obs_dim]
            action_mask: Optional valid action mask [action_dim]
            agent_id: Agent ID string (required for topology lookups)

        Returns:
            Action vector [action_dim] in [0, 1]
        """
        if agent_id is None:
            raise ValueError("agent_id is required for BaseStockPolicy")

        action = np.zeros(self.max_inbound + self.max_outbound, dtype=np.float32)

        current_inventory = float(obs[0]) * 5000.0
        base_stock = self._base_stock_levels.get(agent_id, 1000.0)
        needed = max(0.0, base_stock - current_inventory)

        suppliers = self._inbound_suppliers.get(agent_id, [])
        total_capacity = sum(s[1] for s in suppliers)
        if needed > 0 and total_capacity > 0:
            for i, (edge, capacity, _cost) in enumerate(suppliers):
                if i >= self.max_inbound:
                    break
                edge_key = (edge["from"], edge["to"])
                is_disabled = False
                for d in self.env.disruptions.get("disabled_edges", []):
                    if (d[0], d[1]) == edge_key:
                        is_disabled = True
                        break
                if is_disabled:
                    continue

                cap_mult = self.env.disruptions.get("capacity_multipliers", {}).get(
                    edge["from"], 1.0
                )
                available = capacity * cap_mult
                order_qty = min(needed * (capacity / total_capacity), available)
                action[i] = order_qty / max(available, 1.0)
                needed -= order_qty

        retailers = self._outbound_retailers.get(agent_id, [])
        total_demand = sum(r[1] for r in retailers)
        if total_demand > 0:
            for i, (_edge, demand_mean) in enumerate(retailers):
                if i >= self.max_outbound:
                    break
                route_idx = self.max_inbound + i
                action[route_idx] = demand_mean / total_demand  # proportional alloc

        if action_mask is not None:
            action = action * action_mask

        return action.astype(np.float32)

    def get_actions(self, observations: dict, action_masks: dict = None) -> dict:
        """Compute actions for all agents.

        Args:
            observations: Dict mapping agent_id -> obs_vector
            action_masks: Optional dict mapping agent_id -> mask_vector

        Returns:
            Dict mapping agent_id -> action_vector
        """
        actions = {}
        for agent_id in self.agents:
            obs = observations[agent_id]
            mask = action_masks.get(agent_id) if action_masks else None
            actions[agent_id] = self.get_action(obs, mask, agent_id=agent_id)
        return actions