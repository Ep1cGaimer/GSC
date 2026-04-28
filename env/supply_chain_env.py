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

    def __init__(
        self,
        topology_config: str,
        max_steps: int = 100,
        seed: int = 42,
        traffic_enabled: bool = True,
        stochastic_traffic: bool = True,
    ):
        super().__init__()
        self.graph_builder = GraphBuilder(topology_config)
        self.max_steps = max_steps
        self._seed = seed
        self.rng = np.random.RandomState(seed)
        self.traffic_enabled = traffic_enabled
        self.stochastic_traffic = stochastic_traffic

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

        # In-transit shipments: {(from, to): {"units": float, "steps_remaining": int}}
        self.in_transit = {}
        # Track what happened in each step for animation playback
        self.flow_history = []
        # Flow data for the last step (inbound/outbound shipments)
        self.last_step_flows = {"inbound": [], "outbound": [], "deliveries": []}

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
        # The second outbound block is the route traffic pressure. This makes
        # congestion visible to the policy instead of only affecting the UI.
        obs_size = 4 + self._max_inbound + self._max_outbound + self._max_outbound + 1
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
        elif not hasattr(self, 'rng'):
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
            "demand_multipliers": {},
            "traffic_multipliers": {},
        }

        # Reset in-transit shipments and flow tracking
        self.in_transit = {}
        self.flow_history = []
        self.last_step_flows = {"inbound": [], "outbound": [], "deliveries": []}

        if self.traffic_enabled and self.stochastic_traffic:
            self._refresh_stochastic_traffic()

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
        infos = {}

        # Initialize flow tracking for this step
        step_inbound = []
        step_outbound = []
        step_deliveries = []

        # Process deliveries first: items that have arrived from in_transit
        delivered_keys = []
        for edge_key, transit_data in list(self.in_transit.items()):
            if transit_data["steps_remaining"] <= 0:
                # Deliver the goods to the destination
                dest_id = edge_key[1]
                units = transit_data["units"]
                if units > 0 and dest_id in self.inventory:
                    self.inventory[dest_id] += units
                    step_deliveries.append({
                        "from": edge_key[0],
                        "to": dest_id,
                        "units": round(units, 1)
                    })
                delivered_keys.append(edge_key)

        # Remove delivered items from in_transit
        for key in delivered_keys:
            del self.in_transit[key]

        # Generate demand for this timestep using the traffic state the agent observed.
        current_demands = self._generate_demands()

        for agent_id in self.agents:
            action = actions.get(agent_id)
            if action is None:
                continue

            inbound_edges = self.graph_builder.get_edges_to(agent_id)
            outbound_edges = self.graph_builder.get_edges_from(agent_id)
            node_data = self.graph_builder.get_node_data(agent_id)
            warehouse_capacity = float(node_data.get("capacity", 1000.0))

            # Parse action
            order_quantities = np.clip(action[:self._max_inbound], 0.0, 1.0)
            route_allocations = np.clip(action[self._max_inbound:], 0.0, 1.0)
            active_route_allocations = route_allocations[:len(outbound_edges)].astype(np.float32, copy=True)
            alloc_sum = float(active_route_allocations.sum())
            if alloc_sum > 1.0:
                active_route_allocations /= alloc_sum

            # --- Process inbound orders (add to in_transit, don't arrive instantly) ---
            total_order_cost = 0.0
            total_order_co2 = 0.0
            total_ordered = 0.0

            # Build disabled edges set once
            disabled_edges = set()
            for d in self.disruptions.get("disabled_edges", []):
                disabled_edges.add((d[0], d[1]))

            for i, edge in enumerate(inbound_edges):
                if i >= len(order_quantities):
                    break
                # Check if edge is disabled by disruption
                edge_key = (edge["from"], edge["to"])
                if edge_key in disabled_edges:
                    continue

                # Normalized order quantity * source capacity
                src_data = self.graph_builder.get_node_data(edge["from"])
                src_capacity = src_data.get("capacity", 100)
                cap_mult = self.disruptions.get("capacity_multipliers", {}).get(edge["from"], 1.0)
                available = src_capacity * cap_mult

                order_qty = float(order_quantities[i]) * available
                metrics = self._effective_edge_metrics(edge)
                order_cost = order_qty * metrics["cost_per_unit"]
                order_co2 = order_qty * metrics["distance_km"] * metrics["co2_per_km"]

                total_ordered += order_qty
                total_order_cost += order_cost
                total_order_co2 += order_co2

                # Add to in_transit with lead_time_days as steps_remaining
                if order_qty > 0:
                    lead_time = max(1, int(edge.get("lead_time_days", 1)))
                    if edge_key in self.in_transit:
                        self.in_transit[edge_key]["units"] += order_qty
                        self.in_transit[edge_key]["steps_remaining"] = max(
                            self.in_transit[edge_key]["steps_remaining"], lead_time
                        )
                    else:
                        self.in_transit[edge_key] = {
                            "units": order_qty,
                            "steps_remaining": lead_time
                        }
                    step_inbound.append({
                        "from": edge["from"],
                        "to": agent_id,
                        "units": round(order_qty, 1),
                        "mode": edge.get("mode", "road")
                    })

            # Note: inventory is NOT updated here - goods are in transit
            # They will be added to inventory when steps_remaining reaches 0

            # --- Process outbound fulfillment ---
            # Compute all allocations from a snapshot of inventory
            total_fulfilled = 0.0
            total_fulfill_cost = 0.0
            total_fulfill_co2 = 0.0

            # First pass: compute desired fulfillment for each route
            route_fulfills = []
            available_inventory = self.inventory[agent_id]
            total_desired = 0.0

            for i, edge in enumerate(outbound_edges):
                if i >= len(active_route_allocations):
                    break
                edge_key = (edge["from"], edge["to"])
                if edge_key in disabled_edges:
                    route_fulfills.append(0.0)
                    continue

                dest_id = edge["to"]
                demand = current_demands.get(dest_id, 0.0)
                desired = float(active_route_allocations[i]) * demand
                route_fulfills.append(desired)
                total_desired += desired

            # Scale down if total desired exceeds available inventory
            if total_desired > available_inventory and total_desired > 0:
                scale = available_inventory / total_desired
                route_fulfills = [r * scale for r in route_fulfills]

            # Second pass: apply fulfillment
            for i, edge in enumerate(outbound_edges):
                if i >= len(route_fulfills):
                    break
                fulfilled = route_fulfills[i]
                if fulfilled <= 0:
                    continue

                self.inventory[agent_id] -= fulfilled
                total_fulfilled += fulfilled
                metrics = self._effective_edge_metrics(edge)
                total_fulfill_cost += fulfilled * metrics["cost_per_unit"]
                total_fulfill_co2 += fulfilled * metrics["distance_km"] * metrics["co2_per_km"]

                step_outbound.append({
                    "from": agent_id,
                    "to": edge["to"],
                    "units": round(fulfilled, 1),
                    "mode": edge.get("mode", "road")
                })

            # --- Compute costs ---
            storage_cost = self.inventory[agent_id] * node_data.get("storage_cost", 0.5) * 0.01
            # Add holding cost for goods in transit targeting this warehouse
            in_transit_to_agent = sum(
                v["units"] for k, v in self.in_transit.items() if k[1] == agent_id
            )
            holding_cost = in_transit_to_agent * node_data.get("storage_cost", 0.5) * 0.005
            overflow_penalty = 0.0  # Simplified - no overflow for now
            total_cost = total_order_cost + total_fulfill_cost + storage_cost + holding_cost + overflow_penalty
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
            infos[agent_id] = {
                **self._get_info(agent_id),
                "incoming_units": 0,  # No instant incoming - goods are in transit
                "in_transit_incoming": round(in_transit_to_agent, 1),
                "fulfilled_units": total_fulfilled,
                "demand_units": total_demand,
                "unfulfilled_units": unfulfilled,
                "overflow_units": 0,
                "overflow_penalty": 0,
                "stockout_penalty": stockout_penalty,
                "storage_cost": storage_cost,
                "holding_cost": holding_cost,
                "step_cost": total_cost,
                "step_co2": total_co2,
                "step_revenue": revenue,
            }

        # Decrement steps_remaining for all in-transit items
        for edge_key in self.in_transit:
            self.in_transit[edge_key]["steps_remaining"] -= 1

        # Store flow data for this step
        self.last_step_flows = {
            "inbound": step_inbound,
            "outbound": step_outbound,
            "deliveries": step_deliveries
        }
        self.flow_history.append(self.last_step_flows)
        # Keep only last 50 steps in history
        if len(self.flow_history) > 50:
            self.flow_history = self.flow_history[-50:]

        # Check termination
        done = self.current_step >= self.max_steps
        for agent_id in self.agents:
            terminations[agent_id] = done
            truncations[agent_id] = False

        if not done and self.traffic_enabled and self.stochastic_traffic:
            self._refresh_stochastic_traffic()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        for agent in self.agents:
            infos.setdefault(agent, self._get_info(agent))

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
            self.cumulative_cost[agent_id] / 1e6,                   # Normalized cost
            self.cumulative_co2[agent_id] / 2e7,                    # Normalized CO2
            self._get_expected_demand_pressure(agent_id),           # Demand pressure
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

        # Outbound traffic pressure (1.0 normal, >1.0 congested), padded.
        traffic = self.disruptions.get("traffic_multipliers", {})
        for i in range(self._max_outbound):
            if i < len(outbound_edges):
                edge = outbound_edges[i]
                if edge.get("mode") == "road":
                    features.append(float(traffic.get(self._edge_key(edge), 1.0)) / 3.0)
                else:
                    features.append(1.0 / 3.0)
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

    def get_action_mask(self, agent_id: str, obs_vec: np.ndarray | None = None) -> np.ndarray:
        """Binary mask for valid action dimensions based on the current observation layout."""
        if obs_vec is None:
            obs_vec = self._get_obs(agent_id)

        in_start = 4
        in_end = 4 + self._max_inbound
        out_start = in_end
        out_end = in_end + self._max_outbound

        in_mask = (obs_vec[in_start:in_end] > 0).astype(np.float32)
        out_mask = (obs_vec[out_start:out_end] > 0).astype(np.float32)
        return np.concatenate([in_mask, out_mask], axis=0)

    def _get_expected_demand_pressure(self, agent_id: str) -> float:
        """Expected downstream demand for this warehouse based on connected retailers."""
        pressure = 0.0
        demand_mults = self.disruptions.get("demand_multipliers", {})
        for edge in self.graph_builder.get_edges_from(agent_id):
            dest = self.graph_builder.get_node_data(edge["to"])
            if dest.get("type") != "retailer":
                continue
            base_demand = float(dest.get("demand_mean", 100.0))
            pressure += base_demand * float(demand_mults.get(dest["id"], 1.0))
        return pressure / 1000.0

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
            "demand_multipliers": {},
            "traffic_multipliers": {},
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

    def get_graph_obs(self, agent_id: str) -> dict:
        """Get a homogeneous subgraph observation for a single agent."""
        return self.graph_builder.build_homogeneous_subgraph(
            center_id=agent_id,
            k_hops=2,
            inventory_state=self.inventory,
            disruptions=self.disruptions,
        )

    def get_all_graph_obs(self) -> list:
        """Get homogeneous subgraph observations for all warehouse agents."""
        return [self.get_graph_obs(a) for a in self.agents]

    def update_traffic(self, traffic_multipliers: dict):
        """Update live road traffic multipliers from Maps observations.

        Keys are "FROM->TO" edge ids. Values are duration_in_traffic/base_duration
        clamped to a sane range so a bad API response cannot dominate rewards.
        """
        if not self.traffic_enabled:
            return
        current = self.disruptions.setdefault("traffic_multipliers", {})
        valid_edges = {self._edge_key(edge) for edge in self.graph_builder.config.get("edges", [])}
        for key, value in traffic_multipliers.items():
            if key not in valid_edges:
                continue
            try:
                current[key] = float(np.clip(float(value), 0.75, 3.0))
            except (TypeError, ValueError):
                continue

    def _edge_key(self, edge: dict) -> str:
        return f"{edge['from']}->{edge['to']}"

    def _effective_edge_metrics(self, edge: dict) -> dict:
        metrics = dict(edge)
        key = self._edge_key(edge)
        lead_mult = self.disruptions.get("lead_time_multipliers", {}).get(key, 1.0)
        traffic_mult = self.disruptions.get("traffic_multipliers", {}).get(key, 1.0)
        if edge.get("mode") != "road":
            traffic_mult = 1.0

        metrics["lead_time_days"] = float(edge.get("lead_time_days", 0.0)) * float(lead_mult) * float(traffic_mult)
        # Congestion mostly affects operating cost; CO2 rises more gently.
        metrics["cost_per_unit"] = float(edge.get("cost_per_unit", 0.0)) * (1.0 + 0.35 * (float(traffic_mult) - 1.0))
        metrics["co2_per_km"] = float(edge.get("co2_per_km", 0.0)) * (1.0 + 0.15 * (float(traffic_mult) - 1.0))
        return metrics

    def _refresh_stochastic_traffic(self):
        traffic = {}
        for edge in self.graph_builder.config.get("edges", []):
            if edge.get("mode") != "road":
                continue
            # Most roads are normal; some become moderately/heavily congested.
            draw = self.rng.lognormal(mean=0.0, sigma=0.22)
            traffic[self._edge_key(edge)] = float(np.clip(draw, 0.85, 2.2))
        self.disruptions["traffic_multipliers"] = traffic

    def get_shipment_state(self) -> dict:
        """Get current in-transit shipments for animation.
        
        Returns:
            dict mapping edge keys to shipment data with units and steps remaining.
        """
        result = {}
        for edge_key, data in self.in_transit.items():
            key = f"{edge_key[0]}->{edge_key[1]}"
            result[key] = {
                "from": edge_key[0],
                "to": edge_key[1],
                "units": round(data["units"], 1),
                "steps_remaining": data["steps_remaining"],
                "total_steps": data.get("total_steps", 1)
            }
        return result

    def get_step_flows(self) -> dict:
        """Get what moved in the last step for animation playback.
        
        Returns:
            dict with inbound, outbound, and deliveries from last step.
        """
        return self.last_step_flows

    def get_all_flows(self) -> list:
        """Get the complete flow history for replay or animation.
        
        Returns:
            list of step flow dictionaries.
        """
        return self.flow_history
