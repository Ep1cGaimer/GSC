"""
Local Flask server for ChainGuard AI dashboard.

Serves the dashboard frontend and provides API endpoints for:
  - /api/step   — run one environment step with the trained protagonist
  - /api/state  — get current environment state for the map
  - /api/disrupt — inject adversarial disruption
  - /api/resolve — Gemini + KG signal grounding
"""

import os
import sys
import time
import glob

# Add the project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor
from agents.shield import SafetyShield
from agents.conformal import ConformalEscalation

# --- App Setup ---
DASHBOARD_ROOT = os.path.join(PROJECT_ROOT, "dashboard")
DASHBOARD_DIST_DIR = os.path.join(DASHBOARD_ROOT, "dist")
DASHBOARD_DIR = DASHBOARD_DIST_DIR if os.path.exists(DASHBOARD_DIST_DIR) else DASHBOARD_ROOT
app = Flask(__name__, static_folder=DASHBOARD_DIR)
CORS(app)

# --- Initialize Environment ---
topology_path = os.path.join(PROJECT_ROOT, "env", "topology_configs", "india_50_nodes.yaml")
env = SupplyChainEnv(topology_config=topology_path, stochastic_traffic=False)
agents = env.possible_agents
num_agents = len(agents)
obs_dim = env.observation_space(agents[0]).shape[0]
action_dim = env.action_space(agents[0]).shape[0]

# --- Load Trained Models ---
device = torch.device("cpu")
actor = GNNActor(obs_dim, action_dim).to(device)

# Find and load the latest trained actor checkpoint
models_dir = os.path.join(PROJECT_ROOT, "models")
actor_files = glob.glob(os.path.join(models_dir, "*_actor.pt"))
if actor_files:
    latest_actor = max(actor_files, key=os.path.getctime)
    try:
        actor.load_state_dict(torch.load(latest_actor, map_location=device, weights_only=True))
        print(f"[OK] Loaded trained actor: {os.path.basename(latest_actor)}")
    except RuntimeError as e:
        print(f"[WARNING] Trained actor incompatible with current observation shape; running with random policy. {e}")
else:
    print("[WARNING] No trained actor found — running with random policy")
actor.eval()

# --- Safety Shield ---
shield = SafetyShield(co2_cap=5000, budget_cap=500000)

# --- Conformal Escalation (uncalibrated — will default to fail-safe escalation) ---
conformal = ConformalEscalation()

# --- Signal Resolver (lazy-loaded — only init if APIs are available) ---
_resolver = None
def get_resolver():
    global _resolver
    if _resolver is None:
        if not os.getenv("GEMINI_API_KEY"):
            print("[WARNING] SignalResolver unavailable (missing GEMINI_API_KEY)")
            return None
        try:
            from kg.signal_resolver import SignalResolver
            _resolver = SignalResolver()
        except Exception as e:
            print(f"[WARNING] SignalResolver unavailable (Gemini/Neo4j not configured): {e}")
    return _resolver

# --- State Management ---
current_obs, _ = env.reset()
step_count = 0


def rebuild_current_obs():
    """Rebuild observations after live data changes without advancing the env."""
    global current_obs
    current_obs = {agent: env._get_obs(agent) for agent in env.agents}
    return current_obs


def compute_quality_flags():
    metadata = env.graph_builder.metadata or {}
    config = env.graph_builder.config
    expected_name = metadata.get("name", "")
    expected_nodes = 50 if "50" in expected_name else None
    expected_edges = 72 if "50" in expected_name else None

    flags = []
    if expected_nodes is not None and len(config.get("nodes", [])) != expected_nodes:
        flags.append({
            "type": "topology_mismatch",
            "severity": "warning",
            "message": f"Topology metadata suggests {expected_nodes} nodes but config contains {len(config.get('nodes', []))}.",
        })
    if expected_edges is not None and len(config.get("edges", [])) != expected_edges:
        flags.append({
            "type": "topology_mismatch",
            "severity": "warning",
            "message": f"Topology metadata suggests {expected_edges} edges but config contains {len(config.get('edges', []))}.",
        })
    return flags


def build_state_payload():
    return {
        "step": env.current_step,
        "max_steps": env.max_steps,
        "inventory": {k: round(v, 1) for k, v in env.inventory.items()},
        "revenue": {k: round(v, 1) for k, v in env.cumulative_revenue.items()},
        "co2": {k: round(v, 1) for k, v in env.cumulative_co2.items()},
        "cost": {k: round(v, 1) for k, v in env.cumulative_cost.items()},
        "total_revenue": round(sum(env.cumulative_revenue.values()), 1),
        "total_co2": round(sum(env.cumulative_co2.values()), 1),
        "total_cost": round(sum(env.cumulative_cost.values()), 1),
        "disruptions": env.disruptions,
        "traffic": env.disruptions.get("traffic_multipliers", {}),
        "shipments": env.get_shipment_state(),
        "flows": env.get_step_flows(),
        "quality_flags": compute_quality_flags(),
    }

# --- Get topology data for the map ---
def get_topology_for_map():
    """Extract node positions and edges from topology for frontend rendering."""
    nodes = []
    for node in env.graph_builder.config["nodes"]:
        nodes.append({
            "id": node["id"],
            "type": node["type"],
            "name": node.get("name", node["id"]),
            "lat": node.get("lat", 20.0 + np.random.uniform(-5, 5)),
            "lng": node.get("lng", 78.0 + np.random.uniform(-8, 8)),
            "capacity": node.get("capacity", 0),
        })
    edges = []
    traffic = env.disruptions.get("traffic_multipliers", {})
    for edge in env.graph_builder.config["edges"]:
        effective = env._effective_edge_metrics(edge)
        key = f"{edge['from']}->{edge['to']}"
        edges.append({
            "from": edge["from"],
            "to": edge["to"],
            "mode": edge.get("mode", "road"),
            "distance_km": edge.get("distance_km", 0),
            "lead_time_days": edge.get("lead_time_days", 0),
            "cost_per_unit": edge.get("cost_per_unit", 0),
            "co2_per_km": edge.get("co2_per_km", 0),
            "traffic_multiplier": round(float(traffic.get(key, 1.0)), 3),
            "effective_lead_time_days": round(effective.get("lead_time_days", 0), 3),
            "effective_cost_per_unit": round(effective.get("cost_per_unit", 0), 3),
            "effective_co2_per_km": round(effective.get("co2_per_km", 0), 5),
        })
    metadata = dict(env.graph_builder.metadata or {})
    metadata["node_count"] = len(nodes)
    metadata["edge_count"] = len(edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
        "quality_flags": compute_quality_flags(),
    }


# ==================== ROUTES ====================

@app.route('/')
def index():
    return send_from_directory(DASHBOARD_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(DASHBOARD_DIR, filename)


@app.route('/api/topology', methods=['GET'])
def get_topology():
    """Return full graph topology for map rendering."""
    return jsonify(get_topology_for_map())


@app.route('/api/config', methods=['GET'])
def get_config():
    """Expose browser-safe runtime config for the dashboard."""
    return jsonify({
        "maps_api_key": os.getenv("MAPS_API_KEY", ""),
        "google_maps_map_id": os.getenv("GOOGLE_MAPS_MAP_ID", ""),
        "maps_enabled": bool(os.getenv("MAPS_API_KEY")),
        "topology_config": os.path.basename(topology_path),
        "traffic_runtime": "live_maps_push",
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get the current environment state."""
    return jsonify(build_state_payload())


@app.route('/api/step', methods=['POST'])
def step():
    """Execute one environment step with the trained protagonist."""
    global current_obs, step_count

    # Check if episode is done
    if not env.agents:
        current_obs, _ = env.reset()
        shield.reset()
        step_count = 0

    # 1. Actor chooses actions
    obs_tensor = torch.tensor(
        np.array([current_obs[a] for a in agents]), dtype=torch.float32
    )
    with torch.no_grad():
        actions_tensor, _, _ = actor.get_action_and_value(obs_tensor)

    actions = {agents[i]: actions_tensor[i].numpy() for i in range(num_agents)}

    # 2. Apply Safety Shield
    shield_events = []
    safe_actions = {}
    for agent_id, action in actions.items():
        edges_in = [env._effective_edge_metrics(edge) for edge in env.graph_builder.get_edges_to(agent_id)]
        edges_out = [env._effective_edge_metrics(edge) for edge in env.graph_builder.get_edges_from(agent_id)]
        info = {
            "agent_id": agent_id,
            "inventory": env.inventory[agent_id],
            "cumulative_cost": env.cumulative_cost[agent_id],
            "cumulative_co2": env.cumulative_co2[agent_id],
            "step": env.current_step,
            "max_steps": env.max_steps,
        }

        safe_action, intervened, event = shield.filter(action, info, {}, edges_in, edges_out)
        safe_actions[agent_id] = safe_action
        if intervened:
            shield_events.append({
                "agent_id": agent_id,
                "reason": event.reason,
                "intervened": True,
                "timestamp": time.time()
            })

    # 3. Step Environment
    next_obs, rewards, terminations, truncations, infos = env.step(safe_actions)
    current_obs = next_obs
    step_count += 1

    # 4. Uncertainty Check
    mean_obs = obs_tensor.mean(dim=0).numpy()
    escalate, width, threshold = conformal.should_escalate(mean_obs)

    episode_done = any(terminations.values()) if terminations else False

    return jsonify({
        "status": "success",
        "step": env.current_step,
        "max_steps": env.max_steps,
        "episode_done": episode_done,
        "rewards": {k: round(v, 2) for k, v in rewards.items()},
        "total_reward": round(sum(rewards.values()), 2),
        "shield_events": shield_events,
        "shield_intervention_rate": round(shield.intervention_rate, 4),
        "uncertainty": {
            "width": None if width == float('inf') else round(float(width), 4),
            "threshold": round(float(threshold), 4),
            "escalate": escalate,
        },
        "inventory": {k: round(v, 1) for k, v in env.inventory.items()},
        "total_revenue": round(sum(env.cumulative_revenue.values()), 1),
        "total_co2": round(sum(env.cumulative_co2.values()), 1),
        "disruptions": env.disruptions,
        "traffic": env.disruptions.get("traffic_multipliers", {}),
        "shipments": env.get_shipment_state(),
        "flows": env.get_step_flows(),
        "quality_flags": compute_quality_flags(),
    })


@app.route('/api/traffic', methods=['POST'])
def update_traffic():
    """Ingest live Maps route traffic ratios and make them affect env steps."""
    data = request.get_json(silent=True) or {}
    multipliers = {}

    for item in data.get("observations", []):
        key = item.get("edge_key") or item.get("key")
        base = float(item.get("base_duration_seconds") or 0)
        traffic = float(item.get("traffic_duration_seconds") or 0)
        if key and base > 0 and traffic > 0:
            multipliers[key] = traffic / base

    multipliers.update(data.get("traffic_multipliers", {}))
    env.update_traffic(multipliers)
    rebuild_current_obs()

    return jsonify({
        "status": "traffic_updated",
        "traffic": env.disruptions.get("traffic_multipliers", {}),
        "updated": sorted(multipliers.keys()),
    })


@app.route('/api/disrupt', methods=['POST'])
def disrupt():
    """Inject a disruption into the environment."""
    data = request.get_json(silent=True) or {}

    # Use provided disruption or a default demo disruption
    disruption = data.get("disruption", {
        "disabled_edges": [["P01", "W01"]],
        "capacity_multipliers": {"F01": 0.5},
        "demand_multipliers": {"R03": 2.0},
    })
    env.inject_disruption(disruption)
    return jsonify({"status": "disrupted", "disruption": disruption})


@app.route('/api/resolve', methods=['POST'])
def resolve_signal():
    """Run Gemini + KG signal resolution on raw text."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    resolver = get_resolver()
    if resolver is None:
        # Fallback mock response when APIs aren't configured
        return jsonify({
            "raw_text": text,
            "results": [{
                "entity": {"name": "UnknownEvent", "type": "HAZARD",
                           "description": f"Detected from: {text[:80]}"},
                "grounded": False,
                "protocols": [],
                "affected_locations": [],
            }],
            "overall_confidence": 0.5,
            "note": "Gemini/Neo4j not configured — using fallback"
        })

    try:
        result = resolver.resolve(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset_env():
    """Reset the environment to initial state."""
    global current_obs, step_count
    current_obs, _ = env.reset()
    shield.reset()
    step_count = 0
    return jsonify({"status": "reset", "step": 0})


# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("  ChainGuard AI — Local Dashboard Server")
    print(f"  Dashboard: http://localhost:5000")
    print(f"  API:       http://localhost:5000/api/state")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
