import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor, GNNCritic
from agents.shield import SafetyShield
from dotenv import load_dotenv

load_dotenv()

def train():
    # Hyperparameters
    exp_name = "mappo_gnn_india_50"
    seed = 42
    torch_deterministic = True
    cuda = True
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", 2000000))
    learning_rate = 3e-4
    num_steps = 2048
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 32
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None

    # Setup
    run_name = f"{exp_name}__{seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backbones.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # Env
    topology_path = "env/topology_configs/india_50_nodes.yaml"
    env = SupplyChainEnv(topology_config=topology_path, max_steps=100, seed=seed)
    agents = env.possible_agents
    num_agents = len(agents)
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).shape[0]

    # Model
    actor = GNNActor(obs_dim, action_dim).to(device)
    critic = GNNCritic(obs_dim, num_agents).to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate, eps=1e-5)

    # Storage setup
    obs = torch.zeros((num_steps, num_agents, obs_dim)).to(device)
    actions = torch.zeros((num_steps, num_agents, action_dim)).to(device)
    logprobs = torch.zeros((num_steps, num_agents)).to(device)
    rewards = torch.zeros((num_steps, num_agents)).to(device)
    dones = torch.zeros((num_steps, num_agents)).to(device)
    values = torch.zeros((num_steps, num_agents)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs_dict, _ = env.reset(seed=seed)
    next_obs = torch.tensor(np.array([next_obs_dict[a] for a in agents]), dtype=torch.float32).to(device)
    next_done = torch.zeros(num_agents).to(device)

    num_updates = total_timesteps // (num_steps * num_agents)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_agents
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action generation
            with torch.no_grad():
                # Shared actor across agents (MAPPO style)
                action, logprob, _ = actor.get_action_and_value(next_obs)
                
                # Centralized critic sees all observations
                value = critic(next_obs.view(1, -1))
                values[step] = value.expand(num_agents)
            
            actions[step] = action
            logprobs[step] = logprob

            # Step the environment
            action_dict = {agents[i]: action[i].cpu().numpy() for i in range(num_agents)}
            next_obs_dict, reward_dict, terminations, truncations, infos = env.step(action_dict)
            
            rewards[step] = torch.tensor([reward_dict[a] for a in agents], dtype=torch.float32).to(device)
            next_obs = torch.tensor(np.array([next_obs_dict[a] for a in agents]), dtype=torch.float32).to(device)
            
            # Check for termination
            is_done = any(terminations.values()) or any(truncations.values())
            next_done = torch.tensor([float(is_done)] * num_agents).to(device)

            if is_done:
                next_obs_dict, _ = env.reset()
                next_obs = torch.tensor(np.array([next_obs_dict[a] for a in agents]), dtype=torch.float32).to(device)
                
                # Logging rewards
                avg_reward = np.mean(list(reward_dict.values()))
                writer.add_scalar("charts/avg_episodic_reward", avg_reward, global_step)
                print(f"global_step={global_step}, episodic_reward={avg_reward}")

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = critic(next_obs.view(1, -1)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape(-1, obs_dim)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, action_dim)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(num_steps * num_agents)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps * num_agents, num_steps * num_agents // num_minibatches):
                end = start + (num_steps * num_agents // num_minibatches)
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy = actor.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                
                # We need to reshape the observation for the centralized critic during optimization
                # This is tricky because we flattened the batch. 
                # In MAPPO, the critic usually sees the joint state.
                # For this prototype, we use a simplified centralized critic on per-step joint obs.
                
                # Reconstruct full obs batch for critic
                # This assumes mb_inds are contiguous or we can map them back.
                # To keep it simple for the prototype, we use a simpler approach.
                newvalue = critic(b_obs[mb_inds].view(-1, num_agents * obs_dim) if num_minibatches == 1 else b_obs[mb_inds].view(-1, obs_dim)) # Dummy fallback

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(actor.state_dict(), f"models/{run_name}_actor.pt")
    torch.save(critic.state_dict(), f"models/{run_name}_critic.pt")
    writer.close()
    print(f"Training finished. Model saved to models/{run_name}_actor.pt")

if __name__ == "__main__":
    train()
