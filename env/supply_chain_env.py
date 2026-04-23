"""
Supply Chain Environment: PettingZoo ParallelEnv for multi-agent supply chain optimization.

Each warehouse node is an agent. Agents decide:
  - How much to order from connected suppliers (factories/ports)
  - Which outgoing route to use for fulfilling retailer demand

The environment tracks inventory, costs, CO2 emissions, and demand fulfillment.
Topology is loaded from YAML and represented as a PyG HeteroData graph.
"""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from env.graph_builder import GraphBuilder


class SupplyChainEnv(ParallelEnv):
    """Multi-agent supply chain environment.
    
    Agents: Warehouse nodes (W01, W02, ..., W10)
    Observation: Local graph features (inventory, demand, supply status) 
    Actions: [order_quantity_normalized, route_selection_logit_0, ..., route_selection_logit_k]
    Reward: Revenue from fulfilled demand - costs - penalties
    Constraints: CO2 cap, budget cap (enforced by external shield, tracked here)
    """

    metadata = {
        "name": "supply_chain_v1",
        "render_modes": ["human"],
        "is_parallelizable": True,
    }

    def __init__(self, topology_config: str, max_steps: int = 100, seed: int = 42):
        super().__init__()
        self.graph_builder = GraphBuilder(topology_config)
        self.max_steps = max_steps
        self._seed = seed
        self.rng = np.random.RandomState(seed)

        # Agents are warehouse nodes
        self.possible_agents = self.graph_builder.get_all_agent_ids()
        self.agents = list(self.possible_agents)

        # Topology data
        self.hazmat_zones = self.graph_builder.get_hazmat_zones()

        # Precompute max connections per agent for action space sizing
        self._max_inbound = 0
        self._max_outbound = 0
        for agent_id in self.agents:
            inbound = len(self.graph_builder.get_edges_to(agent_id))
            outbound = len(self.graph_builder.get_edges_from(agent_id))
            self._max_inbound = max(self._max_inbound, inbound)
            self._max_outbound = max(self._max_outbound, outbound)

        # State variables (initialized in reset)
        self.inventory = {}
        self.cumulative_cost = {}
        self.cumulative_co2 = {}
        self.cumulative_revenue = {}
        self.demand_history = {}
        self.current_step = 0
        self.disruptions = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """Observation space per agent.
        
        Features:
        - inventory (1)
        - cumulative_cost (1)
        - cumulative_co2 (1)
        - current_demand_pressure (1) — sum of pending demand from connected retailers
        - inbound_supply_available (max_inbound) — capacity available from suppliers
        - outbound_route_status (max_outbound) — 1.0 if route active, 0.0 if disrupted
        - time_remaining_normalized (1)
        """
        obs_size = 4 + self._max_inbound + self._max_outbound + 1
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """Action space per agent.
        
        Continuous actions:
        - order_quantities (max_inbound) — how much to order from each inbound supplier [0, 1] normalized
        - route_allocations (max_outbound) — allocation fraction for each outbound route [0, 1]
        """
        action_size = self._max_inbound + self._max_outbound
        return spaces.Box(low=0.0, high=1.0, shape=(action_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            self._seed = seed
        self.rng = np.random.RandomState(self._seed)
        self.agents = list(self.possible_agents)
        self.current_step = 0

        # Initialize state for each warehouse agent
        self.inventory = {}
        self.cumulative_cost = {}
        self.cumulative_co2 = {}
        self.cumulative_revenue = {}
        self.demand_history = {}

        for agent_id in self.agents:
            node_data = self.graph_builder.get_node_data(agent_id)
            # Start with 50% capacity as initial inventory
            self.inventory[agent_id] = node_data.get("capacity", 1000) * 0.5
            self.cumulative_cost[agent_id] = 0.0
            self.cumulative_co2[agent_id] = 0.0
            self.cumulative_revenue[agent_id] = 0.0
            self.demand_history[agent_id] = []

        # Reset disruptions
        self.disruptions = {
            "disabled_edges": [],
            "capacity_multipliers": {},
            "lead_time_multipliers": {},
        }

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def step(self, actions: dict):
        """Execute one timestep.
        
        Args:
            actions: dict mapping agent_id -> action array
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.current_step += 1
        rewards = {}
        terminations = {}
        truncations = {}

        # Generate demand for this timestep
        current_demands = self._generate_demands()

        for agent_id in self.agents:
            action = actions.get(agent_id)
            if action is None:
                continue

            inbound_edges = self.graph_builder.get_edges_to(agent_id)
            outbound_edges = self.graph_builder.get_edges_from(agent_id)

            # Parse action
            order_quantities = action[:self._max_inbound]
            route_allocations = action[self._max_inbound:]

            # --- Process inbound orders ---
            total_order_cost = 0.0
            total_order_co2 = 0.0
            total_incoming = 0.0

            for i, edge in enumerate(inbound_edges):
                if i >= len(order_quantities):
                    break
                # Check if edge is disabled by disruption
                edge_key = (edge["from"], edge["to"])
                if edge_key in [(d[0], d[1]) for d in self.disruptions.get("disabled_edges", [])]:
                    continue

                # Normalized order quantity * source capacity
                src_data = self.graph_builder.get_node_data(edge["from"])
                src_capacity = src_data.get("capacity", 100)
                cap_mult = self.disruptions.get("capacity_multipliers", {}).get(edge["from"], 1.0)
                available = src_capacity * cap_mult

                order_qty = float(order_quantities[i]) * available
                order_cost = order_qty * edge["cost_per_unit"]
                order_co2 = order_qty * edge["distance_km"] * edge["co2_per_km"]

                total_incoming += order_qty
                total_order_cost += order_cost
                total_order_co2 += order_co2

            # Update inventory
            self.inventory[agent_id] += total_incoming

            # --- Process outbound fulfillment ---
            total_fulfilled = 0.0
            total_fulfill_cost = 0.0
            total_fulfill_co2 = 0.0

            for i, edge in enumerate(outbound_edges):
                if i >= len(route_allocations):
                    break
                edge_key = (edge["from"], edge["to"])
                if edge_key in [(d[0], d[1]) for d in self.disruptions.get("disabled_edges", [])]:
                    continue

                dest_id = edge["to"]
                demand = current_demands.get(dest_id, 0.0)

                # Allocate from inventory based on route allocation fraction
                allocated = float(route_allocations[i]) * min(demand, self.inventory[agent_id])
                fulfilled = min(allocated, self.inventory[agent_id])

                self.inventory[agent_id] -= fulfilled
                total_fulfilled += fulfilled
                total_fulfill_cost += fulfilled * edge["cost_per_unit"]
                total_fulfill_co2 += fulfilled * edge["distance_km"] * edge["co2_per_km"]

            # --- Compute costs ---
            node_data = self.graph_builder.get_node_data(agent_id)
            storage_cost = self.inventory[agent_id] * node_data.get("storage_cost", 0.5) * 0.01
            total_cost = total_order_cost + total_fulfill_cost + storage_cost
            total_co2 = total_order_co2 + total_fulfill_co2

            self.cumulative_cost[agent_id] += total_cost
            self.cumulative_co2[agent_id] += total_co2

            # Revenue from fulfilled demand
            revenue = total_fulfilled * 5.0  # $5 per unit fulfilled
            self.cumulative_revenue[agent_id] += revenue

            # --- Reward ---
            # Revenue - costs - holding penalty - stockout penalty
            total_demand = sum(
                current_demands.get(e["to"], 0.0)
                for e in outbound_edges
            )
            unfulfilled = max(0, total_demand - total_fulfilled)
            stockout_penalty = unfulfilled * 2.0

            reward = revenue - total_cost - stockout_penalty
            rewards[agent_id] = reward

            # Track demand for info
            self.demand_history[agent_id].append(total_demand)

        # Check termination
        done = self.current_step >= self.max_steps
        for agent_id in self.agents:
            terminations[agent_id] = done
            truncations[agent_id] = False

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        if done:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _generate_demands(self) -> dict:
        """Generate stochastic demand for all retailer nodes."""
        demands = {}
        for node in self.graph_builder.config["nodes"]:
            if node["type"] == "retailer":
                mean = node.get("demand_mean", 100)
                std = node.get("demand_std", 20)
                demand_mult = self.disruptions.get("demand_multipliers", {}).get(node["id"], 1.0)
                demands[node["id"]] = max(0, self.rng.normal(mean * demand_mult, std))
        return demands

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Build observation vector for an agent."""
        inbound_edges = self.graph_builder.get_edges_to(agent_id)
        outbound_edges = self.graph_builder.get_edges_from(agent_id)

        features = [
            self.inventory[agent_id] / 5000.0,                      # Normalized inventory
            self.cumulative_cost[agent_id] / 100000.0,               # Normalized cost
            self.cumulative_co2[agent_id] / 10000.0,                 # Normalized CO2
            float(self.current_step) / float(self.max_steps),        # Time remaining
        ]

        # Inbound supply availability (padded to max_inbound)
        for i in range(self._max_inbound):
            if i < len(inbound_edges):
                edge = inbound_edges[i]
                src_data = self.graph_builder.get_node_data(edge["from"])
                cap_mult = self.disruptions.get("capacity_multipliers", {}).get(edge["from"], 1.0)
                features.append(src_data.get("capacity", 100) * cap_mult / 10000.0)
            else:
                features.append(0.0)

        # Outbound route status (padded to max_outbound)
        disabled = set()
        for e in self.disruptions.get("disabled_edges", []):
            disabled.add((e[0], e[1]))

        for i in range(self._max_outbound):
            if i < len(outbound_edges):
                edge = outbound_edges[i]
                active = 1.0 if (edge["from"], edge["to"]) not in disabled else 0.0
                features.append(active)
            else:
                features.append(0.0)

        # Time remaining
        features.append(1.0 - self.current_step / self.max_steps)

        return np.array(features, dtype=np.float32)

    def _get_info(self, agent_id: str) -> dict:
        """Get info dict for an agent (used by shield, dashboard, logging)."""
        return {
            "agent_id": agent_id,
            "inventory": self.inventory[agent_id],
            "cumulative_cost": self.cumulative_cost[agent_id],
            "cumulative_co2": self.cumulative_co2[agent_id],
            "cumulative_revenue": self.cumulative_revenue[agent_id],
            "step": self.current_step,
            "max_steps": self.max_steps,
        }

    def inject_disruption(self, disruption: dict):
        """Inject a disruption into the environment (used by adversary agent).
        
        Args:
            disruption: dict with optional keys:
                - 'disabled_edges': list of [from_id, to_id]
                - 'capacity_multipliers': dict {node_id: multiplier}
                - 'lead_time_multipliers': dict {edge_key: multiplier}
                - 'demand_multipliers': dict {node_id: multiplier}
        """
        for key in disruption:
            if key in self.disruptions:
                if isinstance(self.disruptions[key], list):
                    self.disruptions[key].extend(disruption[key])
                elif isinstance(self.disruptions[key], dict):
                    self.disruptions[key].update(disruption[key])
            else:
                self.disruptions[key] = disruption[key]

    def clear_disruptions(self):
        """Clear all active disruptions."""
        self.disruptions = {
            "disabled_edges": [],
            "capacity_multipliers": {},
            "lead_time_multipliers": {},
        }

    def get_graph_state(self) -> dict:
        """Get the current graph state for GNN encoding."""
        return {
            "graph": self.graph_builder.build(
                inventory_state=self.inventory,
                disruptions=self.disruptions
            ),
            "inventory": dict(self.inventory),
            "disruptions": dict(self.disruptions),
        }
