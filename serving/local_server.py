from flask import Flask, request, jsonify
from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor
from agents.shield import SafetyShield
from agents.conformal import ConformalEscalation
from kg.signal_resolver import SignalResolver
import torch
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize components
topology_path = "env/topology_configs/india_50_nodes.yaml"
env = SupplyChainEnv(topology_config=topology_path)
agents = env.possible_agents
num_agents = len(agents)
obs_dim = env.observation_space(agents[0]).shape[0]
action_dim = env.action_space(agents[0]).shape[0]

# Load models (dummy for demo if files don't exist)
device = torch.device("cpu")
actor = GNNActor(obs_dim, action_dim).to(device)
shield = SafetyShield(co2_cap=5000, budget_cap=500000)
conformal = ConformalEscalation()
resolver = SignalResolver()

# State management for demo
current_obs, _ = env.reset()

@app.route('/api/step', methods=['POST'])
def step():
    global current_obs
    
    # 1. Protagonist chooses actions
    obs_tensor = torch.tensor(np.array([current_obs[a] for a in agents]), dtype=torch.float32)
    with torch.no_grad():
        actions_tensor, _, _ = actor.get_action_and_value(obs_tensor)
    
    actions = {agents[i]: actions_tensor[i].numpy() for i in range(num_agents)}
    
    # 2. Apply Safety Shield
    shield_results = []
    safe_actions = {}
    for agent_id, action in actions.items():
        # Get edges for shield context
        edges_in = env.graph_builder.get_edges_to(agent_id)
        edges_out = env.graph_builder.get_edges_from(agent_id)
        info = {
            "agent_id": agent_id,
            "inventory": env.inventory[agent_id],
            "cumulative_cost": env.cumulative_cost[agent_id],
            "cumulative_co2": env.cumulative_co2[agent_id],
            "step": env.current_step
        }
        
        safe_action, intervened, event = shield.filter(action, info, {}, edges_in, edges_out)
        safe_actions[agent_id] = safe_action
        if intervened:
            shield_results.append({
                "agent_id": agent_id,
                "reason": event.reason,
                "intervened": True,
                "timestamp": time.time()
            })

    # 3. Step Env
    next_obs, rewards, terminations, truncations, infos = env.step(safe_actions)
    current_obs = next_obs
    
    # 4. Uncertainty Check
    mean_obs = obs_tensor.mean(dim=0).numpy()
    escalate, width, threshold = conformal.should_escalate(mean_obs)

    return jsonify({
        "status": "success",
        "shield_events": shield_results,
        "uncertainty": {"width": float(width), "threshold": float(threshold), "escalate": escalate},
        "inventory": env.inventory,
        "revenue": sum(env.cumulative_revenue.values())
    })

@app.route('/api/resolve', methods=['POST'])
def resolve_signal():
    data = request.json
    text = data.get("text", "")
    result = resolver.resolve(text)
    return jsonify(result)

@app.route('/api/disrupt', methods=['POST'])
def disrupt():
    # Inject a random disruption for demo
    disruption = {
        "disabled_edges": [["P01", "W01"]],
        "capacity_multipliers": {"F01": 0.5},
        "demand_multipliers": {"R03": 2.0}
    }
    env.inject_disruption(disruption)
    return jsonify({"status": "disrupted"})

if __name__ == '__main__':
    import time
    app.run(port=5000)
