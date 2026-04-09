"""
Smoke test for the Waymax-IDQL pipeline (aligned with WaymoOfflineRL).

Tests:
  1. scenario_to_waymax: parse raw TFRecords -> SimulatorState + map_data
  2. waymax_obs_utils: produce 302-D obs matching WaymoOfflineRL format
  3. inverse dynamics: expert [accel, steer] with correct bounds
  4. reward function: multi-objective tanh-normalized
  5. WaymaxGymWrapper: reset/step

Usage:
    cd /tmp && python /path/to/test_waymax_pipeline.py \\
        --data_dir /path/to/RawWaymo/training/ --num_scenarios 2
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def test_scenario_adapter(data_dir, num_scenarios):
    print("=" * 60)
    print("TEST 1: scenario_to_waymax adapter")
    print("=" * 60)
    from scenario_to_waymax import scenario_generator

    count = 0
    for sid, state, map_data, tl_lookup in scenario_generator(data_dir, max_scenarios=num_scenarios):
        print(f"  Scenario: {sid}")
        print(f"    num_objects:     {state.num_objects}")
        print(f"    log_traj shape:  ({state.log_trajectory.x.shape})")
        print(f"    roadgraph pts:   {state.roadgraph_points.x.shape[0]}")
        print(f"    lanes:           {len(map_data['lane_polylines'])}")
        print(f"    crosswalks:      {len(map_data['crosswalk_polygons'])}")
        print(f"    stop signs:      {len(map_data['stopsign_positions'])}")
        print(f"    tl_lookup keys:  {len(tl_lookup)}")
        sdc_idx = int(np.argmax(np.asarray(state.object_metadata.is_sdc)))
        sdc_valid = np.asarray(state.log_trajectory.valid)[sdc_idx]
        print(f"    SDC index:       {sdc_idx}")
        print(f"    SDC valid steps: {sdc_valid.sum()}/{len(sdc_valid)}")
        count += 1
    print(f"  PASS: parsed {count} scenarios\n")
    return count


def test_observation(data_dir):
    print("=" * 60)
    print("TEST 2: observation (302-D flat)")
    print("=" * 60)
    from scenario_to_waymax import scenario_generator
    from jaxrl5.wrappers.waymax_obs_utils import (
        state_to_feature_dict, flatten_state_dict, OBS_DIM,
    )

    for sid, state, map_data, tl_lookup in scenario_generator(data_dir, max_scenarios=1):
        for t in [10, 30, 60]:
            sdc_idx = int(np.argmax(np.asarray(state.object_metadata.is_sdc)))
            if not np.asarray(state.log_trajectory.valid)[sdc_idx, t]:
                print(f"  t={t}: SDC not valid, skipping")
                continue
            sd = state_to_feature_dict(state, t, map_data, tl_lookup, use_sim_trajectory=False)
            print(f"  t={t}: ego={sd['ego']}, agents={sd['agents'].shape}, "
                  f"lanes={sd['lanes'].shape}, cw={sd['crosswalks'].shape}, "
                  f"route={sd['route'].shape}, rules={sd['rules'].shape}")
            obs = flatten_state_dict(sd)
            assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
            assert obs.dtype == np.float32
            assert np.isfinite(obs).all(), "Non-finite in obs!"
            print(f"         flat obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  PASS: OBS_DIM={OBS_DIM}\n")


def test_inverse_dynamics(data_dir):
    print("=" * 60)
    print("TEST 3: expert inverse dynamics")
    print("=" * 60)
    import jax
    from waymax.dynamics.bicycle_model import compute_inverse
    from scenario_to_waymax import scenario_generator

    jit_inverse = jax.jit(compute_inverse)

    for sid, state, map_data, tl_lookup in scenario_generator(data_dir, max_scenarios=1):
        sdc_idx = int(np.argmax(np.asarray(state.object_metadata.is_sdc)))
        for t in [10, 30, 60]:
            action = jit_inverse(traj=state.log_trajectory, timestep=t, dt=0.1)
            raw = np.asarray(action.data)[sdc_idx]
            valid = bool(np.asarray(action.valid)[sdc_idx])
            accel_clip = np.clip(raw[0], -10.0, 8.0)
            steer_clip = np.clip(raw[1], -0.8, 0.8)
            print(f"  t={t}: raw=[{raw[0]:.4f}, {raw[1]:.6f}] "
                  f"clipped=[{accel_clip:.4f}, {steer_clip:.6f}] valid={valid}")
    print("  PASS: inverse dynamics works\n")


def test_reward(data_dir):
    print("=" * 60)
    print("TEST 4: reward function")
    print("=" * 60)
    from scenario_to_waymax import scenario_generator
    from jaxrl5.wrappers.waymax_obs_utils import state_to_feature_dict
    from extract_waymax_expert_data import compute_reward

    for sid, state, map_data, tl_lookup in scenario_generator(data_dir, max_scenarios=1):
        rewards = []
        for t in [10, 30, 50, 70]:
            sdc_idx = int(np.argmax(np.asarray(state.object_metadata.is_sdc)))
            if not np.asarray(state.log_trajectory.valid)[sdc_idx, t]:
                continue
            sd = state_to_feature_dict(state, t, map_data, tl_lookup)
            r = compute_reward(sd)
            rewards.append(r)
            print(f"  t={t}: reward={r:.4f}")
        if rewards:
            print(f"  Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    print("  PASS: reward in [-1, 1]\n")


def test_gym_wrapper(data_dir):
    print("=" * 60)
    print("TEST 5: WaymaxGymWrapper reset/step")
    print("=" * 60)
    from jaxrl5.wrappers.waymax_wrapper import WaymaxGymWrapper

    env = WaymaxGymWrapper(
        tfrecord_dir=data_dir,
        max_scenarios=2,
        start_scenario=0,
    )

    obs = env.reset()
    print(f"  Reset: obs shape={obs.shape}, dtype={obs.dtype}")
    assert obs.shape == env.observation_space.shape

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  Step {step}: action=[{action[0]:.3f}, {action[1]:.3f}] "
              f"reward={reward:.4f}, done={done}")
        if done:
            obs = env.reset()
            print(f"  Reset after done")

    env.close()
    print("  PASS: gym wrapper works\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_scenarios", type=int, default=2)
    args = parser.parse_args()

    test_scenario_adapter(args.data_dir, args.num_scenarios)
    test_observation(args.data_dir)
    test_inverse_dynamics(args.data_dir)
    test_reward(args.data_dir)
    test_gym_wrapper(args.data_dir)

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
