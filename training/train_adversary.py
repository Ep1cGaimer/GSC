import os
import sys
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor
from agents.adversary import AdversaryPolicy
from training.training_utils import stack_agent_obs, compute_team_reward
from dotenv import load_dotenv

load_dotenv()


def find_latest_model(models_dir, suffix="_actor.pt"):
    search_pattern = os.path.join(models_dir, f"*{suffix}")
    files = glob.glob(search_pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


class AdversaryCritic(nn.Module):
    """Value baseline for the adversary — estimates expected adversary reward
    from the mean agent observation before disruption is injected."""

    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def train_adversary():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = int(os.getenv("ADVERSARY_NUM_STEPS", 100))
    learning_rate = float(os.getenv("ADVERSARY_LR", 3e-4))
    total_timesteps = int(os.getenv("ADVERSARY_TOTAL_TIMESTEPS", 500_000))
    batch_size = int(os.getenv("ADVERSARY_BATCH_SIZE", 64))
    num_epochs = int(os.getenv("ADVERSARY_NUM_EPOCHS", 6))
    num_minibatches = int(os.getenv("ADVERSARY_NUM_MINIBATCHES", 4))
    clip_range = float(os.getenv("ADVERSARY_CLIP_RANGE", 0.2))
    entropy_coef = float(os.getenv("ADVERSARY_ENTROPY_COEF", 0.02))
    vf_coef = float(os.getenv("ADVERSARY_VF_COEF", 0.5))
    max_grad_norm = float(os.getenv("ADVERSARY_MAX_GRAD_NORM", 0.5))
    reward_scale = float(os.getenv("ADVERSARY_REWARD_SCALE", 1e-3))

    run_name = f"adversary__{int(time.time())}"
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(runs_dir, run_name))

    print(f"Using device: {device}")
    print(f"Run: {run_name}")

    topology_path = os.path.join(PROJECT_ROOT, "env", "topology_configs", "india_50_nodes.yaml")
    env = SupplyChainEnv(topology_config=topology_path, max_steps=num_steps, seed=42)
    agents = env.possible_agents
    num_agents = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    protagonist = GNNActor(obs_dim, action_dim).to(device)
    latest_actor_path = find_latest_model(models_dir, "_actor_best.pt")
    if latest_actor_path is None:
        latest_actor_path = find_latest_model(models_dir, "_actor.pt")

    if latest_actor_path:
        print(f"Loading protagonist: {latest_actor_path}")
        protagonist.load_state_dict(torch.load(latest_actor_path, map_location=device, weights_only=True))
    else:
        print("WARNING: No trained protagonist found. Adversary trains against random policy.")

    for param in protagonist.parameters():
        param.requires_grad = False
    protagonist.eval()

    retailer_ids = [n["id"] for n in env.graph_builder.config["nodes"] if n["type"] == "retailer"]
    edge_keys = [f"{e['from']}->{e['to']}" for e in env.graph_builder.config["edges"]]
    node_ids = env.possible_agents

    # Use actual topology counts instead of hardcoded values
    num_demand_targets = min(len(retailer_ids), 20)
    num_edge_targets = min(len(edge_keys), 20)
    num_capacity_targets = min(len(node_ids), 20)

    adversary = AdversaryPolicy(
        obs_dim,
        num_demand_targets=num_demand_targets,
        num_edge_targets=num_edge_targets,
        num_capacity_targets=num_capacity_targets,
        disruption_budget=20.0,
    ).to(device)

    adversary_critic = AdversaryCritic(obs_dim).to(device)

    adv_optimizer = optim.Adam(adversary.parameters(), lr=learning_rate, eps=1e-5)
    critic_optimizer = optim.Adam(adversary_critic.parameters(), lr=learning_rate, eps=1e-5)

    global_step = 0
    episode_count = 0

    print("Starting PPO Adversary Training...")
    print("-" * 60)

    while global_step < total_timesteps:
        obs_buffer = []
        action_buffer = []
        logprob_buffer = []
        reward_buffer = []
        value_buffer = []

        for _ in range(batch_size):
            next_obs_dict, _ = env.reset()
            next_obs = torch.tensor(
                stack_agent_obs(next_obs_dict, agents), dtype=torch.float32
            ).to(device)

            mean_obs = next_obs.mean(dim=0).unsqueeze(0)

            adv_action, adv_logprob, entropy = adversary.get_action_and_logprob(mean_obs)
            disruption = adversary.action_to_disruption(
                adv_action, retailer_ids, edge_keys, node_ids
            )

            with torch.no_grad():
                adv_value = adversary_critic(mean_obs).item()

            env.clear_disruptions()
            env.inject_disruption(disruption)

            total_protagonist_reward = 0.0
            episode_steps = 0

            for step in range(num_steps):
                with torch.no_grad():
                    action, _, _ = protagonist.get_action_and_value(next_obs)

                action_dict = {agents[i]: action[i].cpu().numpy() for i in range(num_agents)}
                next_obs_dict, reward_dict, terminations, truncations, infos = env.step(action_dict)

                team_reward, _ = compute_team_reward(reward_dict, infos, agents)
                total_protagonist_reward += team_reward
                episode_steps += 1

                next_obs = torch.tensor(
                    stack_agent_obs(next_obs_dict, agents), dtype=torch.float32
                ).to(device)

                if any(terminations.values()) or any(truncations.values()):
                    break

            adv_reward = -total_protagonist_reward * reward_scale

            obs_buffer.append(mean_obs.cpu().numpy().squeeze(0))
            action_buffer.append(adv_action.detach().cpu().numpy().squeeze(0))
            logprob_buffer.append(adv_logprob.item())
            reward_buffer.append(adv_reward)
            value_buffer.append(adv_value)

            episode_count += 1
            global_step += episode_steps

            writer.add_scalar("adversary/protagonist_reward", total_protagonist_reward, global_step)
            writer.add_scalar("adversary/adversary_reward", adv_reward, global_step)

        # Compute discounted returns for lower variance
        gamma_adv = float(os.getenv("ADVERSARY_GAMMA", 0.99))
        rewards_tensor = torch.tensor(reward_buffer, dtype=torch.float32).to(device)
        values_tensor = torch.tensor(value_buffer, dtype=torch.float32).to(device)
        # For single-step (bandit) formulation, discount doesn't change the value,
        # but we normalize advantages for stability
        advantages_tensor = rewards_tensor - values_tensor

        adv_mean = float(advantages_tensor.mean())
        adv_std = float(advantages_tensor.std()) + 1e-8
        advantages_tensor = (advantages_tensor - adv_mean) / adv_std

        obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(device)
        old_logprobs_tensor = torch.tensor(logprob_buffer, dtype=torch.float32).to(device)
        old_actions_tensor = torch.tensor(np.array(action_buffer), dtype=torch.float32).to(device)

        total_samples = batch_size
        mb_size = max(1, total_samples // num_minibatches)

        for epoch in range(num_epochs):
            indices = np.random.permutation(total_samples)

            for start in range(0, total_samples, mb_size):
                mb_idx = indices[start:start + mb_size]

                mb_obs = obs_tensor[mb_idx]
                mb_actions = old_actions_tensor[mb_idx]
                mb_old_logprobs = old_logprobs_tensor[mb_idx]
                mb_advantages = advantages_tensor[mb_idx]
                mb_returns = rewards_tensor[mb_idx]
                mb_old_values = values_tensor[mb_idx]

                _, new_logprobs, entropy = adversary.get_action_and_logprob(
                    mb_obs, action=mb_actions
                )

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

                policy_loss = -torch.min(
                    ratio * mb_advantages, clipped_ratio * mb_advantages
                ).mean()
                entropy_loss = entropy.mean()

                actor_loss = policy_loss - entropy_coef * entropy_loss

                adv_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(adversary.parameters(), max_grad_norm)
                adv_optimizer.step()

                mb_values = adversary_critic(mb_obs)
                v_clipped = mb_old_values + torch.clamp(
                    mb_values - mb_old_values, -clip_range, clip_range
                )
                v_loss_unclipped = (mb_values - mb_returns) ** 2
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                value_loss = vf_coef * 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(adversary_critic.parameters(), max_grad_norm)
                critic_optimizer.step()

        writer.add_scalar("losses/adversary_policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/adversary_value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/adversary_entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/adversary_advantage_mean", adv_mean, global_step)

        if episode_count % (batch_size * 2) == 0:
            print(
                f"Ep {episode_count:4d} | Step {global_step:7d} | "
                f"Avg Protag Reward: {-rewards_tensor.mean().item() / reward_scale:8.1f} | "
                f"Policy Loss: {policy_loss.item():.4f} | "
                f"Val Loss: {value_loss.item():.4f}"
            )

    adversary_path = os.path.join(models_dir, f"{run_name}_adversary.pt")
    torch.save(adversary.state_dict(), adversary_path)
    torch.save(adversary_critic.state_dict(),
               os.path.join(models_dir, f"{run_name}_advcritic.pt"))
    writer.close()

    print("-" * 60)
    print(f"Adversary training finished. Saved to: {adversary_path}")


if __name__ == "__main__":
    train_adversary()
