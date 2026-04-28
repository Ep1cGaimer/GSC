"""
Evaluation harness for supply chain policies.

Compares Random, BaseStock, and RL policies across disruption scenarios
to establish whether RL is actually adding value over simple heuristics.
"""

import os
import sys
import json
import time
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from env.supply_chain_env import SupplyChainEnv
from agents.gnn_mappo import GNNActor
from agents.baseline import RandomPolicy, BaseStockPolicy
from training.training_utils import stack_agent_obs, stack_action_masks, compute_team_reward
from dotenv import load_dotenv

load_dotenv()

SCENARIOS = {
    "baseline": {
        "description": "No disruptions, normal demand",
        "disruptions": {},
    },
    "mild_stress": {
        "description": "1 edge disabled, 1.5x demand on R03",
        "disruptions": {
            "disabled_edges": [["W01", "R01"]],
            "demand_multipliers": {"R03": 1.5},
        },
    },
    "severe_stress": {
        "description": "3 edges disabled, 2x demand on R03+R04, capacity cut on F01",
        "disruptions": {
            "disabled_edges": [["W01", "R01"], ["W01", "R02"], ["P01", "W01"]],
            "demand_multipliers": {"R03": 2.0, "R04": 2.0},
            "capacity_multipliers": {"F01": 0.5},
        },
    },
}


def evaluate_policy(env, policy, agents, scenario_name, num_episodes=5,
                    max_steps=100, device="cpu", is_rl=False, is_random=False,
                    scenario_disruption=None):
    """Run a policy through multiple episodes and collect metrics.

    Args:
        scenario_disruption: Optional dict to re-inject after each reset.
    """
    results = []

    for ep in range(num_episodes):
        ep_seed = 42 + ep
        env._seed = ep_seed
        env.rng = np.random.RandomState(ep_seed)
        obs_dict, _ = env.reset()
        if scenario_disruption:
            env.inject_disruption(scenario_disruption)

        ep_metrics = {
            "episode": ep,
            "scenario": scenario_name,
            "total_revenue": 0.0,
            "total_cost": 0.0,
            "total_co2": 0.0,
            "total_demand": 0.0,
            "total_fulfilled": 0.0,
            "total_unfulfilled": 0.0,
            "shield_interventions": 0,
            "steps": 0,
        }

        for step in range(max_steps):
            if is_rl:
                obs_tensor = torch.tensor(
                    stack_agent_obs(obs_dict, agents), dtype=torch.float32
                ).to(device)
                masks_tensor = torch.tensor(
                    stack_action_masks(env, obs_dict, agents), dtype=torch.float32
                ).to(device)
                with torch.no_grad():
                    actions_tensor, _, _ = policy.get_action_and_value(
                        obs_tensor, action_mask=masks_tensor
                    )
                actions = {
                    agents[i]: actions_tensor[i].cpu().numpy() for i in range(len(agents))
                }
            elif is_random:
                actions = {}
                for a in agents:
                    obs = obs_dict[a]
                    mask = env.get_action_mask(a, obs_dict[a])
                    actions[a] = policy.get_action(obs, mask)
            else:
                masks = {a: env.get_action_mask(a, obs_dict[a]) for a in agents}
                actions = policy.get_actions(obs_dict, masks)

            obs_dict, _reward_dict, terminations, truncations, infos = env.step(actions)

            team_reward, step_metrics = compute_team_reward(_reward_dict, infos, agents)
            ep_metrics["total_revenue"] += step_metrics["team_revenue"]
            ep_metrics["total_cost"] += step_metrics["team_cost"]
            ep_metrics["total_co2"] += step_metrics["team_co2"]
            ep_metrics["total_demand"] += step_metrics["team_demand"]
            ep_metrics["total_fulfilled"] += step_metrics["team_fulfilled"]
            ep_metrics["total_unfulfilled"] += step_metrics["team_unfulfilled"]
            ep_metrics["steps"] += 1

            if any(terminations.values()) or any(truncations.values()):
                break

        ep_metrics["fill_rate"] = (
            ep_metrics["total_fulfilled"] / max(1.0, ep_metrics["total_demand"])
        )
        ep_metrics["gross_margin"] = (
            ep_metrics["total_revenue"] - ep_metrics["total_cost"]
        )

        results.append(ep_metrics)

    return results


def summarize(results: list) -> dict:
    """Compute mean and std across episodes for each metric."""
    keys = [
        "total_revenue", "total_cost", "total_co2", "fill_rate",
        "gross_margin", "total_unfulfilled",
    ]
    summary = {}
    for key in keys:
        values = [r[key] for r in results]
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)
    return summary


def print_table(all_summaries: dict, scenario_name: str):
    """Print a formatted comparison table for a scenario."""
    policies = list(all_summaries.keys())
    metrics = ["fill_rate", "gross_margin", "total_revenue", "total_cost",
               "total_co2", "total_unfulfilled"]

    print(f"\n{'='*80}")
    print(f"  Scenario: {scenario_name} — {SCENARIOS[scenario_name]['description']}")
    print(f"{'='*80}")
    header = f"{'Metric':<22}" + "".join(f"{p:>18}" for p in policies)
    print(header)
    print("-" * 80)

    for metric in metrics:
        row = f"  {metric:<20}"
        for policy in policies:
            mean = all_summaries[policy].get(f"{metric}_mean", 0)
            std = all_summaries[policy].get(f"{metric}_std", 0)
            row += f"  {mean:>10.1f} ±{std:>5.1f}"
        print(row)

    print("-" * 80)


def run_evaluation(actor_path: str = None, num_episodes: int = 5):
    """Main evaluation entry point.

    Args:
        actor_path: Path to a trained *_actor.pt checkpoint. If None, only
                    random and base-stock policies are evaluated.
        num_episodes: Episodes per scenario per policy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Episodes per scenario: {num_episodes}")

    topology_path = os.path.join(PROJECT_ROOT, "env", "topology_configs",
                                  "india_50_nodes.yaml")

    all_results = {}

    for scenario_name, scenario_cfg in SCENARIOS.items():
        env = SupplyChainEnv(topology_config=topology_path, max_steps=100, seed=42)
        agents = env.possible_agents
        obs_dim = env.observation_space(agents[0]).shape[0]
        action_dim = env.action_space(agents[0]).shape[0]

        scenario_summaries = {}

        # Random policy
        rng = RandomPolicy(action_dim, seed=42)
        env._seed = 42
        env.rng = np.random.RandomState(42)
        obs_dict, _ = env.reset()
        env.clear_disruptions()
        if scenario_cfg["disruptions"]:
            env.inject_disruption(scenario_cfg["disruptions"])

        print(f"\nEvaluating Random on {scenario_name}...")
        results = evaluate_policy(
            env, rng, agents, scenario_name, num_episodes, is_random=True,
            scenario_disruption=scenario_cfg["disruptions"] or None
        )
        scenario_summaries["Random"] = summarize(results)

        # Base-Stock policy
        env._seed = 42
        env.rng = np.random.RandomState(42)
        obs_dict, _ = env.reset()
        env.clear_disruptions()
        if scenario_cfg["disruptions"]:
            env.inject_disruption(scenario_cfg["disruptions"])
        base_stock = BaseStockPolicy(env)

        print(f"Evaluating BaseStock on {scenario_name}...")
        results = evaluate_policy(
            env, base_stock, agents, scenario_name, num_episodes,
            scenario_disruption=scenario_cfg["disruptions"] or None
        )

        scenario_summaries["BaseStock"] = summarize(results)

        # RL policy (if checkpoint provided)
        if actor_path and os.path.exists(actor_path):
            env._seed = 42
            env.rng = np.random.RandomState(42)
            obs_dict, _ = env.reset()
            env.clear_disruptions()
            if scenario_cfg["disruptions"]:
                env.inject_disruption(scenario_cfg["disruptions"])

            rl_policy = GNNActor(obs_dim, action_dim).to(device)
            rl_policy.load_state_dict(
                torch.load(actor_path, map_location=device, weights_only=True)
            )
            rl_policy.eval()

            print(f"Evaluating RL on {scenario_name}...")
            results = evaluate_policy(
                env, rl_policy, agents, scenario_name, num_episodes,
                is_rl=True, device=device,
                scenario_disruption=scenario_cfg["disruptions"] or None
            )
            scenario_summaries["RL"] = summarize(results)
        else:
            print(f"No RL checkpoint found at {actor_path}, skipping RL eval.")

        print_table(scenario_summaries, scenario_name)
        all_results[scenario_name] = scenario_summaries

    results_path = os.path.join(PROJECT_ROOT, "evaluation",
                                 f"eval_results_{int(time.time())}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to trained actor checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per scenario per policy")
    args = parser.parse_args()
    run_evaluation(actor_path=args.load_model, num_episodes=args.episodes)