"""
Bridge between raw Waymo Scenario proto TFRecords and Waymax SimulatorState.

Parses Scenario protobuf messages directly and builds:
  1. A Waymax SimulatorState (direct construction matching WaymoOfflineRL)
  2. A map_data_dict matching WaymoOfflineRL's parser_scenarios.py format
  3. A traffic-light lookup table

Requires:
    waymo_open_dataset.protos.scenario_pb2  (from waymo-open-dataset-tf pip
    package, or compiled from the cloned waymo-open-dataset repo)
"""

import glob
import os
from typing import Iterator, Optional

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymax import datatypes


NUM_TIMESTEPS = 91


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
#  Map data extraction (matches WaymoOfflineRL parser_scenarios.py)
# ---------------------------------------------------------------------------

def extract_map_data(scenario) -> dict:
    """Extract static map data from a Scenario proto.

    Returns a dict with keys:
        lane_polylines, road_line_polylines, road_edge_polylines,
        crosswalk_polygons, stopsign_positions, lane_connectivity
    """
    map_data = {
        'lane_polylines': {},
        'road_line_polylines': {},
        'road_edge_polylines': {},
        'crosswalk_polygons': {},
        'stopsign_positions': {},
        'lane_connectivity': {},
    }

    for feature in scenario.map_features:
        fid = feature.id
        if feature.HasField('lane'):
            pts = np.array(
                [[p.x, p.y, p.z] for p in feature.lane.polyline],
                dtype=np.float32,
            )
            map_data['lane_polylines'][fid] = pts
            map_data['lane_connectivity'][fid] = {
                'left_neighbors': [n.feature_id for n in feature.lane.left_neighbors],
                'right_neighbors': [n.feature_id for n in feature.lane.right_neighbors],
            }
        elif feature.HasField('road_line'):
            pts = np.array(
                [[p.x, p.y, p.z] for p in feature.road_line.polyline],
                dtype=np.float32,
            )
            map_data['road_line_polylines'][fid] = pts
        elif feature.HasField('road_edge'):
            pts = np.array(
                [[p.x, p.y, p.z] for p in feature.road_edge.polyline],
                dtype=np.float32,
            )
            map_data['road_edge_polylines'][fid] = pts
        elif feature.HasField('crosswalk'):
            pts = np.array(
                [[p.x, p.y, p.z] for p in feature.crosswalk.polygon],
                dtype=np.float32,
            )
            map_data['crosswalk_polygons'][fid] = pts
        elif feature.HasField('stop_sign'):
            pos = feature.stop_sign.position
            map_data['stopsign_positions'][fid] = {
                'position': np.array([pos.x, pos.y, pos.z], dtype=np.float32),
                'controls_lanes': list(feature.stop_sign.lane),
            }

    return map_data


def extract_tl_lookup(scenario) -> dict:
    """Build a {timestep: {lane_id: state_enum}} traffic-light lookup table."""
    tl_lookup = {}
    for ts_idx, dms in enumerate(scenario.dynamic_map_states):
        ts_map = {}
        for ls in dms.lane_states:
            if ls.lane > 0:
                ts_map[ls.lane] = ls.state
        tl_lookup[ts_idx] = ts_map
    return tl_lookup


# ---------------------------------------------------------------------------
#  Roadgraph construction (matches WaymoOfflineRL construct_state_from_npz)
# ---------------------------------------------------------------------------

def _build_roadgraph(map_data: dict) -> datatypes.RoadgraphPoints:
    """Build Waymax RoadgraphPoints from the extracted map_data dict."""
    WAYMAX_LANE_SURFACE_STREET = 2
    WAYMAX_ROAD_LINE_UNKNOWN = 5
    WAYMAX_ROAD_EDGE_UNKNOWN = 14
    WAYMAX_STOP_SIGN = 17
    WAYMAX_CROSSWALK = 18

    points_x, points_y, points_z = [], [], []
    points_ids, points_types = [], []

    for fid, polyline in map_data['lane_polylines'].items():
        n = polyline.shape[0]
        points_x.append(polyline[:, 0])
        points_y.append(polyline[:, 1])
        points_z.append(polyline[:, 2])
        points_ids.extend([fid] * n)
        points_types.extend([WAYMAX_LANE_SURFACE_STREET] * n)

    for fid, polyline in map_data['road_line_polylines'].items():
        n = polyline.shape[0]
        points_x.append(polyline[:, 0])
        points_y.append(polyline[:, 1])
        points_z.append(polyline[:, 2])
        points_ids.extend([fid] * n)
        points_types.extend([WAYMAX_ROAD_LINE_UNKNOWN] * n)

    for fid, polyline in map_data['road_edge_polylines'].items():
        n = polyline.shape[0]
        points_x.append(polyline[:, 0])
        points_y.append(polyline[:, 1])
        points_z.append(polyline[:, 2])
        points_ids.extend([fid] * n)
        points_types.extend([WAYMAX_ROAD_EDGE_UNKNOWN] * n)

    for fid, ss_data in map_data['stopsign_positions'].items():
        pos = ss_data['position']
        points_x.append(np.array([pos[0]]))
        points_y.append(np.array([pos[1]]))
        points_z.append(np.array([pos[2]]))
        points_ids.append(fid)
        points_types.append(WAYMAX_STOP_SIGN)

    for fid, polygon in map_data['crosswalk_polygons'].items():
        n = polygon.shape[0]
        points_x.append(polygon[:, 0])
        points_y.append(polygon[:, 1])
        points_z.append(polygon[:, 2])
        points_ids.extend([fid] * n)
        points_types.extend([WAYMAX_CROSSWALK] * n)

    if points_x:
        final_x = np.concatenate(points_x)
        final_y = np.concatenate(points_y)
        final_z = np.concatenate(points_z)
        final_ids = np.array(points_ids, dtype=np.int32)
        final_types = np.array(points_types, dtype=np.int32)
        num_pts = len(final_x)
    else:
        final_x = np.zeros(1, dtype=np.float32)
        final_y = np.zeros(1, dtype=np.float32)
        final_z = np.zeros(1, dtype=np.float32)
        final_ids = np.zeros(1, dtype=np.int32)
        final_types = np.zeros(1, dtype=np.int32)
        num_pts = 1

    return datatypes.RoadgraphPoints(
        x=jnp.array(final_x),
        y=jnp.array(final_y),
        z=jnp.array(final_z),
        dir_x=jnp.zeros(num_pts, dtype=jnp.float32),
        dir_y=jnp.zeros(num_pts, dtype=jnp.float32),
        dir_z=jnp.zeros(num_pts, dtype=jnp.float32),
        valid=jnp.ones(num_pts, dtype=bool),
        ids=jnp.array(final_ids),
        types=jnp.array(final_types),
    )


# ---------------------------------------------------------------------------
#  Main: Scenario proto  -->  (SimulatorState, map_data, tl_lookup)
# ---------------------------------------------------------------------------

def scenario_proto_to_state(scenario):
    """Parse a single Scenario proto into a Waymax SimulatorState + auxiliaries.

    Returns:
        (SimulatorState, map_data_dict, tl_lookup)
    """
    num_agents = len(scenario.tracks)
    n_steps = min(NUM_TIMESTEPS, len(scenario.timestamps_seconds))

    # --- Trajectories (matches WaymoOfflineRL parser_scenarios.py) ---
    all_traj = np.zeros((num_agents, NUM_TIMESTEPS, 10), dtype=np.float32)
    object_ids = np.zeros(num_agents, dtype=np.int32)
    object_types = np.zeros(num_agents, dtype=np.int32)

    for i, track in enumerate(scenario.tracks):
        object_ids[i] = track.id
        object_types[i] = track.object_type
        for t, s in enumerate(track.states):
            if t >= NUM_TIMESTEPS:
                break
            all_traj[i, t, 0] = s.center_x
            all_traj[i, t, 1] = s.center_y
            all_traj[i, t, 2] = s.center_z
            all_traj[i, t, 3] = s.length
            all_traj[i, t, 4] = s.width
            all_traj[i, t, 5] = s.height
            all_traj[i, t, 6] = s.heading
            all_traj[i, t, 7] = s.velocity_x
            all_traj[i, t, 8] = s.velocity_y
            all_traj[i, t, 9] = float(s.valid)

    # --- Timestamps ---
    timestamps = np.array(scenario.timestamps_seconds[:n_steps], dtype=np.float64)
    if len(timestamps) < NUM_TIMESTEPS:
        timestamps = np.pad(timestamps, (0, NUM_TIMESTEPS - len(timestamps)))
    ts_micros = (timestamps * 1e6).astype(np.int64)
    ts_micros_2d = np.tile(ts_micros[None, :], (num_agents, 1))

    # --- Build Trajectory (matches construct_state_from_npz) ---
    log_trajectory = datatypes.Trajectory(
        x=jnp.array(all_traj[:, :, 0]),
        y=jnp.array(all_traj[:, :, 1]),
        z=jnp.array(all_traj[:, :, 2]),
        length=jnp.array(all_traj[:, :, 3]),
        width=jnp.array(all_traj[:, :, 4]),
        height=jnp.array(all_traj[:, :, 5]),
        yaw=jnp.array(all_traj[:, :, 6]),
        vel_x=jnp.array(all_traj[:, :, 7]),
        vel_y=jnp.array(all_traj[:, :, 8]),
        valid=jnp.array(all_traj[:, :, 9], dtype=bool),
        timestamp_micros=jnp.array(ts_micros_2d, dtype=jnp.int64),
    )

    # --- Object Metadata ---
    sdc_idx = scenario.sdc_track_index if scenario.HasField('sdc_track_index') else 0
    is_sdc = jnp.zeros(num_agents, dtype=bool).at[sdc_idx].set(True)

    difficulty_map = {p.track_index: p.difficulty for p in scenario.tracks_to_predict}
    agent_diff = np.array([difficulty_map.get(i, 0) for i in range(num_agents)], dtype=np.int32)

    object_metadata = datatypes.ObjectMetadata(
        ids=jnp.array(object_ids),
        object_types=jnp.array(object_types),
        is_sdc=is_sdc,
        is_modeled=jnp.array(agent_diff > 0, dtype=bool),
        is_valid=jnp.ones(num_agents, dtype=bool),
        objects_of_interest=jnp.zeros(num_agents, dtype=bool),
        is_controlled=jnp.zeros(num_agents, dtype=bool),
    )

    # --- Map / Roadgraph ---
    map_data = extract_map_data(scenario)
    roadgraph = _build_roadgraph(map_data)

    # --- Traffic Lights ---
    tl_lookup = extract_tl_lookup(scenario)

    if scenario.dynamic_map_states:
        tl_lane_to_idx = {
            ls.lane: i
            for i, ls in enumerate(scenario.dynamic_map_states[0].lane_states)
        }
        num_tl_lanes = len(tl_lane_to_idx)
    else:
        tl_lane_to_idx = {}
        num_tl_lanes = 0

    tl_states_raw = np.zeros((NUM_TIMESTEPS, max(num_tl_lanes, 1), 4), np.float32)
    if num_tl_lanes > 0:
        tl_states_raw = np.zeros((NUM_TIMESTEPS, num_tl_lanes, 4), np.float32)
        for t_idx, dms in enumerate(scenario.dynamic_map_states):
            if t_idx >= NUM_TIMESTEPS:
                break
            for ls in dms.lane_states:
                if ls.lane in tl_lane_to_idx:
                    j = tl_lane_to_idx[ls.lane]
                    tl_states_raw[t_idx, j, 0] = ls.lane
                    tl_states_raw[t_idx, j, 1] = ls.state
                    if ls.HasField('stop_point'):
                        tl_states_raw[t_idx, j, 2] = ls.stop_point.x
                        tl_states_raw[t_idx, j, 3] = ls.stop_point.y

    n_tl = tl_states_raw.shape[1]
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((n_tl, NUM_TIMESTEPS)),
        y=jnp.zeros((n_tl, NUM_TIMESTEPS)),
        z=jnp.zeros((n_tl, NUM_TIMESTEPS)),
        state=jnp.array(tl_states_raw[:, :, 1].T, dtype=jnp.int32),
        valid=jnp.array(tl_states_raw[:, :, 1] > 0, dtype=bool).T,
        lane_ids=jnp.array(tl_states_raw[0, :, 0], dtype=jnp.int32)[:, None],
    )

    # --- Assemble SimulatorState ---
    state = datatypes.SimulatorState(
        log_trajectory=log_trajectory,
        sim_trajectory=log_trajectory,
        object_metadata=object_metadata,
        roadgraph_points=roadgraph,
        log_traffic_light=traffic_lights,
        sdc_paths=None,
        timestep=jnp.array(10, dtype=jnp.int32),
    )

    return state, map_data, tl_lookup


# ---------------------------------------------------------------------------
#  Generator: iterate raw TFRecords
# ---------------------------------------------------------------------------

def scenario_generator(
    tfrecord_dir: str,
    max_scenarios: Optional[int] = None,
    start_scenario: int = 0,
) -> Iterator:
    """Yield (scenario_id, SimulatorState, map_data_dict, tl_lookup) tuples.

    Args:
        tfrecord_dir: Directory containing *.tfrecord* files.
        max_scenarios: Stop after yielding this many (None = all).
        start_scenario: Skip this many scenarios first.
    """
    pattern = os.path.join(tfrecord_dir, 'training_20s.tfrecord-*-of-*')
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = os.path.join(tfrecord_dir, '*.tfrecord*')
        paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No TFRecord files found in {tfrecord_dir}')

    global_idx = 0
    yielded = 0

    for path in paths:
        ds = tf.data.TFRecordDataset([path])
        for raw in ds:
            if global_idx < start_scenario:
                global_idx += 1
                continue

            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw.numpy())

            state, map_data, tl_lookup = scenario_proto_to_state(scenario)
            yield scenario.scenario_id, state, map_data, tl_lookup

            global_idx += 1
            yielded += 1
            if max_scenarios is not None and yielded >= max_scenarios:
                return
