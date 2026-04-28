import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor, GNNCritic
from training.training_utils import stack_agent_obs, stack_action_masks, compute_team_reward
from dotenv import load_dotenv

load_dotenv()


def linear_schedule(start, end, current, total):
    """Linear decay from start to end over total steps."""
    fraction = min(float(current) / float(max(1, total)), 1.0)
    return start + fraction * (end - start)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", 1_000_000))
    num_steps = int(os.getenv("ROLLOUT_STEPS", 200))
    lr_start = float(os.getenv("LR_START", 3e-4))
    lr_end = float(os.getenv("LR_END", 5e-5))
    num_epochs = int(os.getenv("NUM_EPOCHS", 5))
    num_minibatches = int(os.getenv("NUM_MINIBATCHES", 8))
    gamma = float(os.getenv("GAMMA", 0.99))
    gae_lambda = float(os.getenv("GAE_LAMBDA", 0.95))
    clip_range = float(os.getenv("CLIP_RANGE", 0.2))
    ent_start = float(os.getenv("ENT_START", 0.05))
    ent_end = float(os.getenv("ENT_END", 0.001))
    # Entropy anneals over this many steps, NOT over total_timesteps.
    # This prevents entropy from staying high for hundreds of thousands of steps
    # when total_timesteps is large (e.g. 2M), which blocks exploitation.
    ent_anneal_steps = int(os.getenv("ENT_ANNEAL_STEPS", 100_000))
    vf_coef = float(os.getenv("VF_COEF", 1.0))
    max_grad_norm = float(os.getenv("MAX_GRAD_NORM", 0.5))
    reward_scale = float(os.getenv("REWARD_SCALE", 1e-3))
    target_kl = float(os.getenv("TARGET_KL", 0.015))

    run_name = f"mappo_v2__{int(time.time())}"
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(runs_dir, run_name))

    print(f"Using device: {device}")
    print(f"Run: {run_name}")
    print(f"ent_coef: {ent_start:.3f}->{ent_end:.3f} (over {ent_anneal_steps:,} steps), lr: {lr_start:.0e}->{lr_end:.0e}")
    print(f"num_steps: {num_steps}, epochs: {num_epochs}, minibatches: {num_minibatches}, reward_scale: {reward_scale}")

    topology_path = os.path.join(PROJECT_ROOT, "env", "topology_configs", "india_50_nodes.yaml")
    print(f"DEBUG: topology_path = {topology_path}")
    print(f"DEBUG: File exists = {os.path.exists(topology_path)}")
    env = SupplyChainEnv(topology_config=topology_path, max_steps=100, seed=42)
    agents = env.possible_agents
    num_agents = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    print(f"Agents: {num_agents}, Obs dim: {obs_dim}, Action dim: {action_dim}")

    actor = GNNActor(obs_dim, action_dim).to(device)
    critic = GNNCritic(obs_dim, num_agents).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_start, eps=1e-5)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_start, eps=1e-5)

    global_step = 0
    episode_count = 0
    episode_rewards = []
    best_mean_reward = -float("inf")
    rollout_count = 0

    next_obs_dict, _ = env.reset()
    next_obs = torch.tensor(stack_agent_obs(next_obs_dict, agents), dtype=torch.float32).to(device)

    print("Starting PPO Training (v2)...")
    print("-" * 60)

    while global_step < total_timesteps:
        # Anneal learning rate over full training, entropy over a fixed window
        lr = linear_schedule(lr_start, lr_end, global_step, total_timesteps)
        entropy_coef = linear_schedule(ent_start, ent_end, global_step, ent_anneal_steps)
        for pg in actor_optimizer.param_groups:
            pg["lr"] = lr
        for pg in critic_optimizer.param_groups:
            pg["lr"] = lr

        concat_obs_buffer = []
        reward_buffer = []
        done_buffer = []
        value_buffer = []
        obs_buffer = []
        action_buffer = []
        logprob_buffer = []
        mask_buffer = []
        episode_start_idx = 0  # Track where current episode starts in the buffer

        for step in range(num_steps):
            concat_obs = next_obs.reshape(-1)
            concat_obs_buffer.append(concat_obs.cpu().numpy())
            obs_buffer.append(next_obs.cpu().numpy())

            action_masks = torch.tensor(
                stack_action_masks(env, next_obs_dict, agents), dtype=torch.float32
            ).to(device)
            mask_buffer.append(action_masks.cpu().numpy())

            with torch.no_grad():
                actions, logprobs, _ = actor.get_action_and_value(next_obs, action_mask=action_masks)
                critic_input = concat_obs.unsqueeze(0)
                value = critic(critic_input).item()

            action_buffer.append(actions.cpu().numpy())
            logprob_buffer.append(logprobs.cpu().numpy())
            value_buffer.append(value)

            action_dict = {agents[i]: actions[i].cpu().numpy() for i in range(num_agents)}
            next_obs_dict, reward_dict, terminations, truncations, infos = env.step(action_dict)

            team_reward, step_metrics = compute_team_reward(reward_dict, infos, agents)
            reward_buffer.append(team_reward * reward_scale)

            done = any(terminations.values()) or any(truncations.values())
            done_buffer.append(done)

            next_obs = torch.tensor(stack_agent_obs(next_obs_dict, agents), dtype=torch.float32).to(device)
            global_step += 1

            if done:
                episode_count += 1
                ep_total_reward = sum(reward_buffer[episode_start_idx:]) / reward_scale
                episode_rewards.append(ep_total_reward)
                if len(episode_rewards) > 50:
                    episode_rewards.pop(0)
                writer.add_scalar("charts/episode_reward", ep_total_reward, global_step)
                mean_rew = float(np.mean(episode_rewards))
                if mean_rew > best_mean_reward:
                    best_mean_reward = mean_rew
                    torch.save(actor.state_dict(), os.path.join(models_dir, f"{run_name}_actor_best.pt"))
                    torch.save(critic.state_dict(), os.path.join(models_dir, f"{run_name}_critic_best.pt"))
                next_obs_dict, _ = env.reset()
                next_obs = torch.tensor(stack_agent_obs(next_obs_dict, agents), dtype=torch.float32).to(device)
                episode_start_idx = step + 1  # Next episode starts at the next buffer index

        # GAE computation
        concat_obs_final = next_obs.reshape(1, -1)
        with torch.no_grad():
            next_value = critic(concat_obs_final).item()

        returns = np.zeros(num_steps, dtype=np.float32)
        advantages = np.zeros(num_steps, dtype=np.float32)
        gae = 0.0

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_non_terminal = 1.0 - float(done_buffer[step])
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(done_buffer[step])
                next_val = value_buffer[step + 1]

            delta = reward_buffer[step] + gamma * next_val * next_non_terminal - value_buffer[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            returns[step] = gae + value_buffer[step]
            advantages[step] = gae

        adv_mean = float(advantages.mean())
        adv_std = float(advantages.std()) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Convert to tensors
        concat_obs_tensor = torch.tensor(np.array(concat_obs_buffer), dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        obs_tensor = torch.tensor(np.array(obs_buffer), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(np.array(action_buffer), dtype=torch.float32).to(device)
        logprobs_tensor = torch.tensor(np.array(logprob_buffer), dtype=torch.float32).to(device)
        masks_tensor = torch.tensor(np.array(mask_buffer), dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)

        mb_critic_size = max(1, num_steps // num_minibatches)

        for epoch in range(num_epochs):
            step_indices = np.random.permutation(num_steps)

            for mb_start in range(0, num_steps, mb_critic_size):
                mb_step_idx = step_indices[mb_start:mb_start + mb_critic_size]

                # --- Actor update ---
                mb_obs = obs_tensor[mb_step_idx].reshape(-1, obs_dim)
                mb_actions = actions_tensor[mb_step_idx].reshape(-1, action_dim)
                mb_old_logprobs = logprobs_tensor[mb_step_idx].reshape(-1)
                mb_masks = masks_tensor[mb_step_idx].reshape(-1, action_dim)
                mb_advantages = (
                    advantages_tensor[mb_step_idx]
                    .unsqueeze(-1)
                    .expand(-1, num_agents)
                    .reshape(-1)
                )

                _, new_logprobs, entropy = actor.get_action_and_value(
                    mb_obs, action=mb_actions, action_mask=mb_masks
                )

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()

                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                entropy_loss = entropy.mean()

                actor_loss = policy_loss - entropy_coef * entropy_loss

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optimizer.step()

                # --- Critic update (no value clipping — it was killing the signal) ---
                mb_concat_obs = concat_obs_tensor[mb_step_idx]
                mb_returns = returns_tensor[mb_step_idx]

                mb_values = critic(mb_concat_obs).squeeze(-1)
                value_loss = vf_coef * 0.5 * ((mb_values - mb_returns) ** 2).mean()

                critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()

            # Early stop if KL exceeds target
            if approx_kl > target_kl * 1.5:
                break

        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", float(approx_kl), global_step)
        writer.add_scalar("charts/advantage_mean", adv_mean, global_step)
        writer.add_scalar("charts/entropy_coef", entropy_coef, global_step)
        writer.add_scalar("charts/lr", lr, global_step)
        if episode_rewards:
            writer.add_scalar("charts/mean_episode_reward", np.mean(episode_rewards), global_step)

        rollout_count += 1
        if rollout_count % 5 == 0:
            mean_rew = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            print(
                f"Ep {episode_count:4d} | Step {global_step:7d} | "
                f"Mean Rew: {mean_rew:8.4f} | "
                f"P Loss: {policy_loss.item():.4f} | "
                f"V Loss: {value_loss.item():.6f} | "
                f"Ent: {entropy_loss.item():.3f} | "
                f"KL: {float(approx_kl):.4f}"
            )

    torch.save(actor.state_dict(), os.path.join(models_dir, f"{run_name}_actor.pt"))
    torch.save(critic.state_dict(), os.path.join(models_dir, f"{run_name}_critic.pt"))
    writer.close()

    print("-" * 60)
    print(f"Training finished. Best mean reward: {best_mean_reward:.4f}")


if __name__ == "__main__":
    train()
