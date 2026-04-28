import numpy as np


def stack_agent_obs(obs_dict, agents):
    return np.array([obs_dict[a] for a in agents], dtype=np.float32)


def stack_action_masks(env, obs_dict, agents):
    return np.array([env.get_action_mask(a, obs_dict[a]) for a in agents], dtype=np.float32)


def compute_team_reward(reward_dict, infos, agents):
    total_reward = 0.0
    total_revenue = 0.0
    total_cost = 0.0
    total_co2 = 0.0
    total_demand = 0.0
    total_fulfilled = 0.0
    total_unfulfilled = 0.0
    total_overflow = 0.0
    total_stockout_penalty = 0.0

    for agent_id in agents:
        total_reward += float(reward_dict.get(agent_id, 0.0))
        info = infos.get(agent_id, {})
        total_revenue += float(info.get("step_revenue", 0.0))
        total_cost += float(info.get("step_cost", 0.0))
        total_co2 += float(info.get("step_co2", 0.0))
        total_demand += float(info.get("demand_units", 0.0))
        total_fulfilled += float(info.get("fulfilled_units", 0.0))
        total_unfulfilled += float(info.get("unfulfilled_units", 0.0))
        total_overflow += float(info.get("overflow_units", 0.0))
        total_stockout_penalty += float(info.get("stockout_penalty", 0.0))

    fill_rate = total_fulfilled / max(1.0, total_demand)
    gross_margin = total_revenue - total_cost
    mean_reward = total_reward / max(1, len(agents))

    metrics = {
        "team_reward": mean_reward,
        "team_reward_sum": total_reward,
        "gross_margin": gross_margin,
        "team_revenue": total_revenue,
        "team_cost": total_cost,
        "team_co2": total_co2,
        "team_demand": total_demand,
        "team_fulfilled": total_fulfilled,
        "team_unfulfilled": total_unfulfilled,
        "team_overflow": total_overflow,
        "team_fill_rate": fill_rate,
        "team_stockout_penalty": total_stockout_penalty,
    }
    return mean_reward, metrics


def merge_metric_sums(metric_sums, step_metrics):
    for key, value in step_metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
    return metric_sums
