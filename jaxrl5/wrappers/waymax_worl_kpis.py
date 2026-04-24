"""WaymoOfflineRL-style episode KPIs (ported from stage_4_evaluation/metrics.py)."""

import jax
from jax import numpy as jnp

from waymax import datatypes
from waymax.utils import geometry

OFF_ROAD_THRESHOLD = 2.0
GOAL_THRESHOLD = 3.0


def check_collision(state: datatypes.SimulatorState, sdc_index: int) -> bool:
    sim_traj = state.sim_trajectory
    valid_mask = sim_traj.valid
    all_bboxes_5dof = sim_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
    sdc_bboxes_5dof = all_bboxes_5dof[sdc_index]

    def check_collision_at_timestep(t):
        sdc_bbox = sdc_bboxes_5dof[t]
        other_agents_bboxes = all_bboxes_5dof[:, t, :]
        other_agents_valid = valid_mask[:, t]
        all_overlaps_at_t = jax.vmap(geometry.has_overlap, in_axes=(None, 0))(
            sdc_bbox, other_agents_bboxes,
        )
        not_sdc_mask = (jnp.arange(state.num_objects) != sdc_index)
        valid_collisions_mask = all_overlaps_at_t & other_agents_valid & not_sdc_mask
        return jnp.any(valid_collisions_mask)

    num_timesteps = state.sim_trajectory.num_timesteps
    collisions_over_time = jax.vmap(check_collision_at_timestep)(jnp.arange(num_timesteps))
    return bool(jnp.any(collisions_over_time))


def check_off_road(state: datatypes.SimulatorState, sdc_index: int) -> bool:
    sdc_positions = state.sim_trajectory.xy[sdc_index]
    rg_points = state.roadgraph_points
    # WOMD / Waymax: 14=ROAD_EDGE_UNKNOWN, 15=BOUNDARY, 16=MEDIAN. Our
    # scenario_to_waymax.py tags parsed road edges as 14 only, so include 14
    # or this check always returns False (GIF can look off-road while KPI is 0%).
    ROAD_EDGE_TYPES = jnp.array([14, 15, 16])
    edge_mask = jnp.isin(rg_points.types, ROAD_EDGE_TYPES)
    if not bool(edge_mask.any()):
        return False
    road_edge_points = jnp.stack([rg_points.x[edge_mask], rg_points.y[edge_mask]], axis=-1)

    def min_dist_to_edges(point):
        return jnp.min(jnp.linalg.norm(road_edge_points - point, axis=1))

    min_distances_over_time = jax.vmap(min_dist_to_edges)(sdc_positions)
    sdc_valid_mask = state.sim_trajectory.valid[sdc_index]
    valid_distances = min_distances_over_time[sdc_valid_mask]
    if valid_distances.shape[0] == 0:
        return False
    max_dist_from_road = jnp.max(valid_distances)
    return bool(max_dist_from_road > OFF_ROAD_THRESHOLD)


def check_goal_completion(state: datatypes.SimulatorState, sdc_index: int) -> bool:
    final_sim_position = state.sim_trajectory.xy[sdc_index, -1]
    final_log_position = state.log_trajectory.xy[sdc_index, -1]
    is_final_state_valid = state.sim_trajectory.valid[sdc_index, -1]
    distance_to_goal = jnp.linalg.norm(final_sim_position - final_log_position)
    return bool(is_final_state_valid & (distance_to_goal < GOAL_THRESHOLD))


def compute_worl_episode_kpis(state: datatypes.SimulatorState, sdc_index: int) -> dict:
    """Binary KPIs aligned with WaymoOfflineRL run_evaluation success definition."""
    collided = check_collision(state, sdc_index)
    off_road = check_off_road(state, sdc_index)
    goal_completed = check_goal_completion(state, sdc_index)
    is_success = goal_completed and (not collided) and (not off_road)
    return {
        'collision': float(collided),
        'off_road': float(off_road),
        'goal_completed': float(goal_completed),
        'success': float(is_success),
    }
