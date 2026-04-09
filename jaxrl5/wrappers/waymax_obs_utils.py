"""
Observation utilities matching WaymoOfflineRL's feature format.

Produces the same structured observation dictionary as
WaymoOfflineRL/src/stage_3_offline_rl/preprocess.py and
WaymoOfflineRL/src/stage_4_evaluation/run_evaluation.py::state_to_feature_dict.

Observation layout (flat 302-D):
    ego       :   3  (speed, acceleration, yaw_rate)
    agents    : 150  (15 agents x 10: rel_x, rel_y, rel_vx, rel_vy, rel_heading,
                       length, width, type_onehot_3)
    lanes     : 100  (50 points x 2: rel_x, rel_y)
    crosswalks:  20  (10 points x 2: rel_x, rel_y)
    route     :  20  (10 waypoints x 2: rel_x, rel_y)
    rules     :   9  (dist_goal, dir_goal_x, dir_goal_y, dist_stop,
                       is_stop_controlled, tl_G, tl_Y, tl_R, tl_U)
    Total     : 302
"""

import numpy as np
import jax.numpy as jnp
from waymax import datatypes

NUM_CLOSEST_AGENTS = 15
NUM_CLOSEST_MAP_POINTS = 50
NUM_CLOSEST_CROSSWALK_POINTS = 10
NUM_FUTURE_WAYPOINTS = 10
MAX_DIST = 100.0

FEATURES_PER_AGENT = 10
OBS_DIM = 3 + NUM_CLOSEST_AGENTS * FEATURES_PER_AGENT + \
          NUM_CLOSEST_MAP_POINTS * 2 + NUM_CLOSEST_CROSSWALK_POINTS * 2 + \
          NUM_FUTURE_WAYPOINTS * 2 + 9  # = 302


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _find_current_lane_id(ego_pos, ego_heading, map_data):
    """Find the lane ID the ego is most likely driving in."""
    if not map_data['lane_polylines']:
        return None

    best_lane_id = None
    min_dist = float('inf')
    ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])

    for lane_id, polyline in map_data['lane_polylines'].items():
        distances = np.linalg.norm(polyline[:, :2] - ego_pos, axis=1)
        closest_idx = np.argmin(distances)
        dist = distances[closest_idx]
        if dist < min_dist:
            if closest_idx < len(polyline) - 1:
                lane_dir = polyline[closest_idx + 1, :2] - polyline[closest_idx, :2]
            else:
                lane_dir = polyline[closest_idx, :2] - polyline[closest_idx - 1, :2]
            if np.dot(lane_dir, ego_dir) > 0:
                min_dist = dist
                best_lane_id = lane_id

    return best_lane_id if min_dist < 5.0 else None


def _get_tl_state_enum(ts, current_lane_id, tl_lookup=None, state=None):
    """Get traffic light state for current lane."""
    if current_lane_id is None:
        return 0

    if tl_lookup is not None and ts in tl_lookup:
        return tl_lookup[ts].get(current_lane_id, 0)

    if state is not None and state.log_traffic_light is not None:
        lane_ids = state.log_traffic_light.lane_ids
        mask = (lane_ids == current_lane_id).any(axis=-1)
        if mask.any():
            light_idx = int(jnp.argmax(mask))
            t_idx = min(ts, state.log_traffic_light.state.shape[-1] - 1)
            return int(np.asarray(state.log_traffic_light.state[light_idx, t_idx]))

    return 0


def _tl_state_to_onehot(tl_state_enum):
    """Convert Waymo TL state enum to [G, Y, R, U] one-hot."""
    vec = np.zeros(4, dtype=np.float32)
    if tl_state_enum in [3, 6]:
        vec[0] = 1.0
    elif tl_state_enum in [2, 5, 8]:
        vec[1] = 1.0
    elif tl_state_enum in [1, 4, 7]:
        vec[2] = 1.0
    else:
        vec[3] = 1.0
    return vec


def state_to_feature_dict(
    state: datatypes.SimulatorState,
    ts: int,
    map_data: dict,
    tl_lookup: dict = None,
    use_sim_trajectory: bool = False,
) -> dict:
    """Convert a Waymax SimulatorState + map_data into the structured feature dict.

    Args:
        state: Waymax SimulatorState (unbatched).
        ts: Timestep index to observe.
        map_data: The map_data dict from scenario_to_waymax.extract_map_data.
        tl_lookup: Optional {timestep: {lane_id: state_enum}} for offline mode.
        use_sim_trajectory: If True read sim_trajectory; else log_trajectory.

    Returns:
        Dict with keys: ego, agents, lanes, crosswalks, route, rules.
    """
    traj = state.sim_trajectory if use_sim_trajectory else state.log_trajectory
    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))

    # --- Frame of reference ---
    ego_x = float(np.asarray(traj.x)[sdc_idx, ts])
    ego_y = float(np.asarray(traj.y)[sdc_idx, ts])
    ego_pos = np.array([ego_x, ego_y])
    ego_heading = float(np.asarray(traj.yaw)[sdc_idx, ts])

    c, s = np.cos(ego_heading), np.sin(ego_heading)
    rot = np.array([[c, s], [-s, c]])  # global → ego frame

    # --- 1. Ego features ---
    vx = float(np.asarray(traj.vel_x)[sdc_idx, ts])
    vy = float(np.asarray(traj.vel_y)[sdc_idx, ts])
    speed = np.sqrt(vx**2 + vy**2)

    if ts > 0:
        prev_vx = float(np.asarray(traj.vel_x)[sdc_idx, ts - 1])
        prev_vy = float(np.asarray(traj.vel_y)[sdc_idx, ts - 1])
        prev_speed = np.sqrt(prev_vx**2 + prev_vy**2)
        prev_heading = float(np.asarray(traj.yaw)[sdc_idx, ts - 1])
        acceleration = (speed - prev_speed) / 0.1
        yaw_rate = normalize_angle(ego_heading - prev_heading) / 0.1
    else:
        acceleration, yaw_rate = 0.0, 0.0

    ego_features = np.array([speed, acceleration, yaw_rate], dtype=np.float32)

    # --- 2. Agent features ---
    all_x = np.asarray(traj.x)[:, ts]
    all_y = np.asarray(traj.y)[:, ts]
    all_vx = np.asarray(traj.vel_x)[:, ts]
    all_vy = np.asarray(traj.vel_y)[:, ts]
    all_yaw = np.asarray(traj.yaw)[:, ts]
    all_valid = np.asarray(traj.valid)[:, ts].astype(bool)
    all_len = np.asarray(traj.length)[:, ts] if np.asarray(traj.length).ndim == 2 else np.asarray(traj.length)
    all_wid = np.asarray(traj.width)[:, ts] if np.asarray(traj.width).ndim == 2 else np.asarray(traj.width)
    obj_types = np.asarray(state.object_metadata.object_types)

    dist_to_ego = np.sqrt((all_x - ego_x)**2 + (all_y - ego_y)**2)
    dist_to_ego[sdc_idx] = np.inf
    sorted_indices = np.argsort(dist_to_ego)

    agent_features = np.zeros((NUM_CLOSEST_AGENTS, FEATURES_PER_AGENT), dtype=np.float32)
    added = 0
    for idx in sorted_indices:
        if added >= NUM_CLOSEST_AGENTS:
            break
        if idx == sdc_idx or not all_valid[idx]:
            continue
        rel_pos = rot @ (np.array([all_x[idx], all_y[idx]]) - ego_pos)
        rel_vel = rot @ np.array([all_vx[idx] - vx, all_vy[idx] - vy])
        rel_heading = normalize_angle(all_yaw[idx] - ego_heading)

        type_vec = np.zeros(3, dtype=np.float32)
        ot = int(obj_types[idx])
        if ot in [1, 2, 3]:
            type_vec[ot - 1] = 1.0

        agent_features[added] = [
            rel_pos[0], rel_pos[1],
            rel_vel[0], rel_vel[1],
            rel_heading,
            float(all_len[idx]), float(all_wid[idx]),
            type_vec[0], type_vec[1], type_vec[2],
        ]
        added += 1

    # --- 3. Lane features ---
    lane_points = np.zeros((NUM_CLOSEST_MAP_POINTS, 2), dtype=np.float32)
    lane_polylines = list(map_data['lane_polylines'].values())
    if lane_polylines:
        all_lane_pts = np.vstack(lane_polylines)[:, :2]
        dists = np.linalg.norm(all_lane_pts - ego_pos, axis=1)
        nearest = np.argsort(dists)[:NUM_CLOSEST_MAP_POINTS]
        transformed = (rot @ (all_lane_pts[nearest] - ego_pos).T).T
        lane_points[:transformed.shape[0]] = transformed

    # --- 4. Crosswalk features ---
    crosswalk_points = np.zeros((NUM_CLOSEST_CROSSWALK_POINTS, 2), dtype=np.float32)
    cw_polygons = list(map_data['crosswalk_polygons'].values())
    if cw_polygons:
        all_cw_pts = np.vstack(cw_polygons)[:, :2]
        dists = np.linalg.norm(all_cw_pts - ego_pos, axis=1)
        nearest = np.argsort(dists)[:NUM_CLOSEST_CROSSWALK_POINTS]
        transformed = (rot @ (all_cw_pts[nearest] - ego_pos).T).T
        crosswalk_points[:transformed.shape[0]] = transformed

    # --- 5. Route features (future waypoints from log) ---
    log_traj = state.log_trajectory
    sdc_log_x = np.asarray(log_traj.x)[sdc_idx]
    sdc_log_y = np.asarray(log_traj.y)[sdc_idx]
    future_indices = np.clip(np.arange(ts + 5, ts + 51, 5), 0, 90)
    future_global = np.stack([sdc_log_x[future_indices], sdc_log_y[future_indices]], axis=-1)
    route_features = (rot @ (future_global - ego_pos).T).T.astype(np.float32)

    # --- 6. Rules features ---
    final_dest = np.array([sdc_log_x[-1], sdc_log_y[-1]])
    dist_to_goal = np.linalg.norm(final_dest - ego_pos)
    dir_to_goal_global = final_dest - ego_pos
    dir_to_goal_ego = rot @ dir_to_goal_global
    dir_norm = np.linalg.norm(dir_to_goal_ego)
    if dir_norm > 1e-4:
        dir_to_goal_ego /= dir_norm

    dist_to_stop = MAX_DIST
    if map_data['stopsign_positions']:
        ss_pos = np.array([d['position'][:2] for d in map_data['stopsign_positions'].values()])
        dist_to_stop = float(np.min(np.linalg.norm(ss_pos - ego_pos, axis=1)))

    current_lane_id = _find_current_lane_id(ego_pos, ego_heading, map_data)

    stop_controlled_lanes = set()
    if map_data['stopsign_positions']:
        for ss_data in map_data['stopsign_positions'].values():
            stop_controlled_lanes.update(ss_data['controls_lanes'])
    is_stop_controlled = 1.0 if current_lane_id in stop_controlled_lanes else 0.0

    tl_state_enum = _get_tl_state_enum(ts, current_lane_id, tl_lookup, state)
    tl_vec = _tl_state_to_onehot(tl_state_enum)

    rules = np.concatenate([
        np.array([dist_to_goal, dir_to_goal_ego[0], dir_to_goal_ego[1],
                  dist_to_stop, is_stop_controlled]),
        tl_vec,
    ]).astype(np.float32)

    return {
        'ego': ego_features,
        'agents': agent_features,
        'lanes': lane_points,
        'crosswalks': crosswalk_points,
        'route': route_features,
        'rules': rules,
    }


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    """Flatten the structured state dict into a 1-D float32 vector."""
    return np.concatenate([
        state_dict['ego'].ravel(),
        state_dict['agents'].ravel(),
        state_dict['lanes'].ravel(),
        state_dict['crosswalks'].ravel(),
        state_dict['route'].ravel(),
        state_dict['rules'].ravel(),
    ]).astype(np.float32)


def flatten_state_to_obs(
    state: datatypes.SimulatorState,
    ts: int,
    map_data: dict,
    tl_lookup: dict = None,
    use_sim_trajectory: bool = False,
) -> np.ndarray:
    """One-call shorthand: state -> flat 302-D observation vector."""
    d = state_to_feature_dict(state, ts, map_data, tl_lookup, use_sim_trajectory)
    return flatten_state_dict(d)
