import os
import time
import torch
import torch.optim as optim
import numpy as np
from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor
from agents.adversary import AdversaryPolicy
from dotenv import load_dotenv

load_dotenv()

def train_adversary():
    """Minimax training loop for the adversarial agent.
    
    1. Load a pre-trained protagonist (GNN-MAPPO).
    2. Freeze protagonist weights.
    3. Train adversary to minimize protagonist reward.
    4. Alternate (Optional for more robust minimax).
    """
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 1024
    learning_rate = 5e-4
    total_timesteps = 500000
    
    # Env
    topology_path = "env/topology_configs/india_50_nodes.yaml"
    env = SupplyChainEnv(topology_config=topology_path)
    agents = env.possible_agents
    num_agents = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    # Protagonist (Frozen)
    protagonist = GNNActor(obs_dim, action_dim).to(device)
    # Load checkpoint if available
    # protagonist.load_state_dict(torch.load("models/mappo_checkpoint.pt"))
    for param in protagonist.parameters():
        param.requires_grad = False
    protagonist.eval()

    # Adversary
    # Targets for disruption (subset of nodes/edges for demo focus)
    retailer_ids = [n["id"] for n in env.graph_builder.config["nodes"] if n["type"] == "retailer"]
    edge_keys = [f"{e['from']}->{e['to']}" for e in env.graph_builder.config["edges"]]
    node_ids = env.possible_agents

    adversary = AdversaryPolicy(obs_dim, 
                                num_demand_targets=10, 
                                num_edge_targets=10, 
                                num_capacity_targets=10).to(device)
    
    adv_optimizer = optim.Adam(adversary.parameters(), lr=learning_rate)

    # Training Loop
    next_obs_dict, _ = env.reset()
    next_obs = torch.tensor(np.array([next_obs_dict[a] for a in agents]), dtype=torch.float32).to(device)
    
    global_step = 0
    while global_step < total_timesteps:
        # 1. Adversary chooses disruption based on current state (mean state)
        mean_obs = next_obs.mean(dim=0).unsqueeze(0) # Adversary sees global aggregate
        disruption = adversary.get_disruption(mean_obs, retailer_ids, edge_keys, node_ids)
        
        # 2. Inject disruption into env
        env.clear_disruptions()
        env.inject_disruption(disruption)
        
        # 3. Collect rollouts for protagonist under disruption
        # Adversary's reward is negative of protagonist's average reward
        total_protagonist_reward = 0
        
        for step in range(num_steps):
            with torch.no_grad():
                # Protagonist acts
                action, _, _ = protagonist.get_action_and_value(next_obs)
            
            action_dict = {agents[i]: action[i].cpu().numpy() for i in range(num_agents)}
            next_obs_dict, reward_dict, terminations, truncations, _ = env.step(action_dict)
            
            total_protagonist_reward += np.mean(list(reward_dict.values()))
            next_obs = torch.tensor(np.array([next_obs_dict[a] for a in agents]), dtype=torch.float32).to(device)
            
            if any(terminations.values()) or any(truncations.values()):
                next_obs_dict, _ = env.reset()
                next_obs = torch.tensor(np.array([next_obs_dict[a] for a in agents]), dtype=torch.float32).to(device)
                break
        
        # 4. Adversary Update
        # Simple policy gradient approach for adversary
        adv_reward = -total_protagonist_reward / num_steps
        
        # In a real minimax, we'd store rollouts and do PPO for adversary too.
        # For prototype, we use a simplified update.
        _, adv_logprob, _ = adversary.get_action_and_logprob(mean_obs)
        adv_loss = -adv_logprob * adv_reward
        
        adv_optimizer.zero_grad()
        adv_loss.backward()
        adv_optimizer.step()
        
        global_step += num_steps
        if global_step % 10000 == 0:
            print(f"Step: {global_step}, Adv Reward: {adv_reward:.2f}")

    # Save adversary
    os.makedirs("models", exist_ok=True)
    torch.save(adversary.state_dict(), "models/adversary_checkpoint.pt")
    print("Adversary training finished.")

if __name__ == "__main__":
    train_adversary()
