import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

from env.supply_chain_env import SupplyChainEnv

# Test the environment
topology_path = os.path.join(PROJECT_ROOT, "env", "topology_configs", "india_50_nodes.yaml")
print(f"Topology path: {topology_path}")
print(f"File exists: {os.path.exists(topology_path)}")

env = SupplyChainEnv(topology_config=topology_path, max_steps=100, seed=42)

print("Env created")
print(f"Agents: {env.possible_agents}")

obs, info = env.reset()
print("Reset done")
print(f"Obs keys: {list(obs.keys())}")
print(f"Obs shape for first agent: {obs[env.possible_agents[0]].shape}")

# Try a few steps
import numpy as np
for i in range(5):
    # Create random actions
    actions = {}
    for agent in env.possible_agents:
        action_dim = env.action_space(agent).shape[0]
        actions[agent] = np.random.uniform(0, 1, action_dim)

    obs, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Step {i}: rewards={rewards[env.possible_agents[0]]:.2f}, done={terminations[env.possible_agents[0]]}")

    if terminations[env.possible_agents[0]]:
        break

print("Test completed")
