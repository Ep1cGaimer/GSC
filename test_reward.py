import numpy as np
import torch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from env.supply_chain_env import SupplyChainEnv

env = SupplyChainEnv(topology_config=os.path.join(PROJECT_ROOT, "env", "topology_configs", "india_50_nodes.yaml"), max_steps=100, seed=42)

for ep in range(5):
    obs, _ = env.reset()
    ep_reward = {a: 0.0 for a in env.possible_agents}
    ep_rev = 0.0
    for step in range(100):
        actions = {a: np.zeros(env.action_space(a).shape[0], dtype=np.float32) for a in env.possible_agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        for a in env.possible_agents:
            ep_reward[a] += rewards.get(a, 0.0)
        if any(term.values()) or any(trunc.values()):
            break
    
    total_rev = sum(env.cumulative_revenue.values())
    avg_reward = np.mean(list(ep_reward.values()))
    print(f"Ep {ep}: Rev: {total_rev}, Avg Reward: {avg_reward}")
