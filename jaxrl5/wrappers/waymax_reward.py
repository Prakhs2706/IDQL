"""Offline-style reward (same as extract_waymax_expert_data / WaymoOfflineRL)."""

import numpy as np

# --- Reward weights (WaymoOfflineRL stage_3_1 reward_function.py) ---
W_ROUTE_FOLLOWING = 2.0
W_SAFETY = -5.0
W_COMFORT = -3.0
REWARD_SCALING_FACTOR = 10.0
MIN_TTC_THRESHOLD = 2.5
LATERAL_INFLUENCE_THRESHOLD = 2.0


def _compute_route_following_reward(ego, route):
    speed = ego[0]
    ego_vel = np.array([speed, 0.0])
    target_dir = route[0]
    norm = np.linalg.norm(target_dir)
    if norm < 1e-4:
        return 0.0
    unit_dir = target_dir / norm
    projected = np.dot(ego_vel, unit_dir)
    progress = max(0.0, projected)
    cross_track = abs(target_dir[1])
    return progress - cross_track ** 2


def _compute_safety_penalty(agents):
    ttc_values = []
    for agent in agents:
        if np.linalg.norm(agent[:2]) < 1e-4:
            continue
        rel_x, rel_y, rel_vx = agent[0], agent[1], agent[2]
        if rel_x > 0 and abs(rel_y) < LATERAL_INFLUENCE_THRESHOLD and rel_vx < -0.1:
            ttc = rel_x / (-rel_vx)
            ttc_values.append(ttc)
    if not ttc_values:
        return 0.0
    min_ttc = min(ttc_values)
    if min_ttc > MIN_TTC_THRESHOLD:
        return 0.0
    return (1.0 - min_ttc / MIN_TTC_THRESHOLD) ** 2


def _compute_comfort_penalty(ego):
    accel = ego[1]
    yaw_rate = ego[2]
    return (accel / 5.0) ** 2 + (yaw_rate / 0.8) ** 2


def compute_reward(state_dict: dict) -> float:
    """Multi-objective reward, tanh-normalized to [-1, 1]."""
    ego = state_dict['ego']
    agents = state_dict['agents']
    route = state_dict['route']
    raw = (
        W_ROUTE_FOLLOWING * _compute_route_following_reward(ego, route) +
        W_SAFETY * _compute_safety_penalty(agents) +
        W_COMFORT * _compute_comfort_penalty(ego)
    )
    return float(np.tanh(raw / REWARD_SCALING_FACTOR))
