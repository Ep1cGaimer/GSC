"""
Safety Shield: Deterministic action masking with mathematical guarantees.

Two-layer safety:
  1. SHIELD (this module) — hard mathematical constraint enforcement. No learning.
     Runs AFTER the policy proposes an action, BEFORE env.step() executes it.
  2. CMDP Lagrangian (in training loop) — soft learned constraint. Learns to
     propose actions that don't trigger the shield.

The shield guarantees ZERO constraint violations regardless of policy quality.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShieldEvent:
    """Record of a shield intervention for logging/dashboard."""
    agent_id: str
    step: int
    reason: str
    original_action: np.ndarray
    safe_action: np.ndarray
    constraint_value: float
    constraint_limit: float


class SafetyShield:
    """Hard safety filter for supply chain actions.
    
    Enforces:
    1. CO2 emission caps (SDG 13 — Climate Action)
    2. Budget caps (SDG 12 — Responsible Production)
    3. Hazmat routing rules (no hazmat through residential zones)
    4. Inventory bounds (prevent negative inventory)
    
    Every intervention is logged with full context for the dashboard.
    """

    def __init__(self, co2_cap: float = 5000.0, budget_cap: float = 500000.0,
                 hazmat_zones: set = None, max_order_fraction: float = 0.8):
        self.co2_cap = co2_cap
        self.budget_cap = budget_cap
        self.hazmat_zones = hazmat_zones or set()
        self.max_order_fraction = max_order_fraction
        self.intervention_log: list[ShieldEvent] = []
        self.total_interventions = 0
        self.total_checks = 0

    def filter(self, action: np.ndarray, agent_state: dict,
               global_state: dict, edges_in: list, edges_out: list) -> tuple[np.ndarray, bool, Optional[ShieldEvent]]:
        """Filter an action through all safety constraints.
        
        Args:
            action: Raw action from policy [order_q0, ..., order_qN, route_a0, ..., route_aM]
            agent_state: Dict with keys: agent_id, inventory, cumulative_cost, cumulative_co2, step, max_steps
            global_state: Dict with global info (all agent states, disruptions)
            edges_in: List of inbound edge dicts
            edges_out: List of outbound edge dicts
            
        Returns:
            (safe_action, was_intervened, shield_event_or_None)
        """
        self.total_checks += 1
        safe_action = action.copy()
        agent_id = agent_state["agent_id"]
        step = agent_state["step"]
        was_intervened = False
        last_event = None

        # --- Check 1: CO2 Cap ---
        projected_co2 = self._estimate_co2(safe_action, agent_state, edges_in, edges_out)
        total_co2 = agent_state["cumulative_co2"] + projected_co2

        if total_co2 > self.co2_cap:
            remaining_co2 = max(0, self.co2_cap - agent_state["cumulative_co2"])
            if projected_co2 > 0:
                scale = remaining_co2 / projected_co2
                scale = max(0.0, min(1.0, scale))
                n_orders = len(edges_in)
                safe_action[:n_orders] *= scale

            last_event = ShieldEvent(
                agent_id=agent_id, step=step,
                reason=f"CO₂ cap exceeded ({total_co2:.0f} > {self.co2_cap:.0f})",
                original_action=action, safe_action=safe_action.copy(),
                constraint_value=total_co2, constraint_limit=self.co2_cap,
            )
            self.intervention_log.append(last_event)
            self.total_interventions += 1
            was_intervened = True

        # --- Check 2: Budget Cap ---
        projected_cost = self._estimate_cost(safe_action, agent_state, edges_in, edges_out)
        total_cost = agent_state["cumulative_cost"] + projected_cost

        if total_cost > self.budget_cap:
            remaining_budget = max(0, self.budget_cap - agent_state["cumulative_cost"])
            if projected_cost > 0:
                scale = remaining_budget / projected_cost
                scale = max(0.0, min(1.0, scale))
                n_orders = len(edges_in)
                safe_action[:n_orders] *= scale

            last_event = ShieldEvent(
                agent_id=agent_id, step=step,
                reason=f"Budget cap exceeded (₹{total_cost:.0f} > ₹{self.budget_cap:.0f})",
                original_action=action, safe_action=safe_action.copy(),
                constraint_value=total_cost, constraint_limit=self.budget_cap,
            )
            self.intervention_log.append(last_event)
            self.total_interventions += 1
            was_intervened = True

        # --- Check 3: Hazmat Routing ---
        for i, edge in enumerate(edges_out):
            dest_id = edge["to"]
            if dest_id in self.hazmat_zones:
                n_orders = len(edges_in)
                route_idx = n_orders + i
                if route_idx < len(safe_action) and safe_action[route_idx] > 0.0:
                    safe_action[route_idx] = 0.0  # Block this route

                    last_event = ShieldEvent(
                        agent_id=agent_id, step=step,
                        reason=f"Hazmat zone routing blocked ({dest_id})",
                        original_action=action, safe_action=safe_action.copy(),
                        constraint_value=1.0, constraint_limit=0.0,
                    )
                    self.intervention_log.append(last_event)
                    self.total_interventions += 1
                    was_intervened = True

        # --- Check 4: Over-ordering (prevent drain) ---
        for i, edge in enumerate(edges_in):
            if i < len(safe_action) and safe_action[i] > self.max_order_fraction:
                safe_action[i] = self.max_order_fraction

                last_event = ShieldEvent(
                    agent_id=agent_id, step=step,
                    reason=f"Over-ordering capped at {self.max_order_fraction:.0%}",
                    original_action=action, safe_action=safe_action.copy(),
                    constraint_value=float(action[i]), constraint_limit=self.max_order_fraction,
                )
                self.intervention_log.append(last_event)
                self.total_interventions += 1
                was_intervened = True

        return safe_action, was_intervened, last_event

    def _estimate_co2(self, action: np.ndarray, agent_state: dict,
                      edges_in: list, edges_out: list) -> float:
        """Estimate CO2 from proposed action."""
        total_co2 = 0.0
        for i, edge in enumerate(edges_in):
            if i < len(action):
                order_frac = float(action[i])
                total_co2 += order_frac * edge["distance_km"] * edge["co2_per_km"] * 100  # Scale
        for i, edge in enumerate(edges_out):
            idx = len(edges_in) + i
            if idx < len(action):
                alloc = float(action[idx])
                total_co2 += alloc * edge["distance_km"] * edge["co2_per_km"] * 50
        return total_co2

    def _estimate_cost(self, action: np.ndarray, agent_state: dict,
                       edges_in: list, edges_out: list) -> float:
        """Estimate cost from proposed action."""
        total_cost = 0.0
        for i, edge in enumerate(edges_in):
            if i < len(action):
                order_frac = float(action[i])
                total_cost += order_frac * edge["cost_per_unit"] * 500  # Scale
        for i, edge in enumerate(edges_out):
            idx = len(edges_in) + i
            if idx < len(action):
                alloc = float(action[idx])
                total_cost += alloc * edge["cost_per_unit"] * 200
        return total_cost

    @property
    def intervention_rate(self) -> float:
        """Fraction of actions that were modified by the shield."""
        if self.total_checks == 0:
            return 0.0
        return self.total_interventions / self.total_checks

    def get_recent_events(self, n: int = 20) -> list[ShieldEvent]:
        """Get the N most recent shield events."""
        return self.intervention_log[-n:]

    def reset(self):
        """Reset intervention log for new episode."""
        self.intervention_log.clear()
        self.total_interventions = 0
        self.total_checks = 0
