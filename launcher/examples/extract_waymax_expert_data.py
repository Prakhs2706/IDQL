"""
Extract expert offline RL data from raw Waymo TFRecords using Waymax.

Reads Scenario proto TFRecords, converts to Waymax SimulatorState,
then extracts:
  - Flat observations  (302-D, matching WaymoOfflineRL format)
  - Expert actions     (Waymax inverse bicycle model, clipped to WaymoOfflineRL bounds)
  - Rewards            (multi-objective, tanh-normalized to [-1,1])
  - Done / mask signals

Outputs a .npz compatible with IDQL's Dataset.

Usage:
    python launcher/examples/extract_waymax_expert_data.py \\
        --data_dir /path/to/RawWaymo/training/ \\
        --max_scenarios 10000 --checkpoint_every 1000

    Default --output_path: ~/scratch/data/waymax_expert/waymo_train_10k.npz
"""

import argparse
import os

# Expert .npz files live under scratch (not inside the repo clone).
DEFAULT_WAYMAX_EXPERT_DIR = os.path.expanduser('~/scratch/data/waymax_expert')
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_WAYMAX_EXPERT_DIR, 'waymo_train_10k.npz')
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from waymax.dynamics import bicycle_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from scenario_to_waymax import scenario_generator, NUM_TIMESTEPS

from jaxrl5.wrappers.waymax_obs_utils import (
    state_to_feature_dict, flatten_state_dict, OBS_DIM,
)
from jaxrl5.wrappers.waymax_reward import compute_reward

# --- Action bounds (matching WaymoOfflineRL TrainingConfig) ---
MAX_ACCELERATION = 8.0
MIN_ACCELERATION = -10.0
MAX_STEERING_ANGLE = 0.8


def _compute_expert_actions(state, t, jit_inverse):
    """Compute clipped [accel, steering] from log_trajectory at timestep t."""
    action = jit_inverse(traj=state.log_trajectory, timestep=t, dt=0.1)
    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
    raw = np.asarray(action.data)[sdc_idx]
    v = bool(np.asarray(action.valid)[sdc_idx])
    if not v:
        return np.zeros(2, dtype=np.float32), False
    accel = float(np.clip(raw[0], MIN_ACCELERATION, MAX_ACCELERATION))
    steer = float(np.clip(raw[1], -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))
    return np.array([accel, steer], dtype=np.float32), True


def _save_checkpoint(path, ep_done, obs, acts, rews, nxt, dones, masks):
    base, ext = os.path.splitext(path)
    ext = ext or '.npz'
    ckpt = f'{base}_ckpt{ep_done}{ext}'
    np.savez(
        ckpt,
        observations=np.array(obs, dtype=np.float32),
        actions=np.array(acts, dtype=np.float32),
        rewards=np.array(rews, dtype=np.float32),
        next_observations=np.array(nxt, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
        masks=np.array(masks, dtype=np.float32),
    )
    actual = ckpt if ckpt.endswith('.npz') else ckpt + '.npz'
    sz = os.path.getsize(actual) / (1024 * 1024)
    print(f'  [CKPT] {ep_done} scenarios -> {actual} ({sz:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(
        description='Extract Waymax expert data for offline RL')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument(
        '--output_path', type=str, default=DEFAULT_OUTPUT_PATH,
        help=f'Output .npz path (default: {DEFAULT_OUTPUT_PATH})',
    )
    parser.add_argument('--max_scenarios', type=int, default=10000)
    parser.add_argument('--start_scenario', type=int, default=0)
    parser.add_argument('--checkpoint_every', type=int, default=1000)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    jit_inverse = jax.jit(bicycle_model.compute_inverse)

    all_obs, all_act, all_rew, all_nxt, all_done, all_mask = [], [], [], [], [], []
    ep_count = 0
    failed = 0
    t0 = time.time()

    gen = scenario_generator(
        tfrecord_dir=args.data_dir,
        max_scenarios=args.max_scenarios,
        start_scenario=args.start_scenario,
    )

    for scenario_id, state, map_data, tl_lookup in gen:
        try:
            sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
            sdc_valid = np.asarray(state.log_trajectory.valid)[sdc_idx]

            last_valid_t = 0
            for t in range(NUM_TIMESTEPS - 1, -1, -1):
                if sdc_valid[t]:
                    last_valid_t = t
                    break

            end_t = min(last_valid_t, NUM_TIMESTEPS - 2)
            ep_transitions = 0

            for t in range(90):
                if t > end_t:
                    break
                if not sdc_valid[t] or not sdc_valid[t + 1]:
                    continue

                action, act_valid = _compute_expert_actions(state, t, jit_inverse)
                if not act_valid:
                    continue

                sd = state_to_feature_dict(state, t, map_data, tl_lookup,
                                           use_sim_trajectory=False)
                sd_next = state_to_feature_dict(state, t + 1, map_data, tl_lookup,
                                                use_sim_trajectory=False)
                obs = flatten_state_dict(sd)
                next_obs = flatten_state_dict(sd_next)

                reward = compute_reward(sd)
                is_done = (t >= end_t)
                mask = 0.0 if is_done else 1.0

                all_obs.append(obs)
                all_act.append(action)
                all_rew.append(reward)
                all_nxt.append(next_obs)
                all_done.append(is_done)
                all_mask.append(mask)
                ep_transitions += 1

            ep_count += 1

            if ep_count % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f'Scenario {ep_count}/{args.max_scenarios} '
                    f'({scenario_id}) | '
                    f'steps={ep_transitions} | '
                    f'total={len(all_obs)} | '
                    f'{elapsed:.0f}s'
                )

            if args.checkpoint_every > 0 and ep_count % args.checkpoint_every == 0:
                _save_checkpoint(
                    args.output_path, ep_count,
                    all_obs, all_act, all_rew, all_nxt, all_done, all_mask,
                )

        except Exception as e:
            print(f'Scenario {scenario_id} failed: {e}')
            failed += 1

    if not all_obs:
        print('NO DATA COLLECTED.')
        return

    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_act, dtype=np.float32)
    rewards = np.array(all_rew, dtype=np.float32)
    next_observations = np.array(all_nxt, dtype=np.float32)
    dones = np.array(all_done, dtype=bool)
    masks = np.array(all_mask, dtype=np.float32)

    print(f'\n{"=" * 60}')
    print(f'Extraction complete!')
    print(f'  Scenarios: {ep_count} ok, {failed} failed')
    print(f'  Transitions: {len(observations)}')
    print(f'  Obs shape: {observations.shape} (expected dim={OBS_DIM})')
    print(f'  Act shape: {actions.shape}')
    print(f'  Act range: [{actions.min():.4f}, {actions.max():.4f}]')
    print(f'  Act std per dim: {actions.std(axis=0)}')
    print(f'  Reward range: [{rewards.min():.4f}, {rewards.max():.4f}]')
    print(f'  Reward mean: {rewards.mean():.4f}')
    print(f'  Episodes (dones): {dones.sum()}')
    print(f'  Elapsed: {time.time() - t0:.0f}s')
    print(f'{"=" * 60}')

    np.savez(
        args.output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        masks=masks,
    )
    actual = args.output_path if args.output_path.endswith('.npz') else args.output_path + '.npz'
    sz = os.path.getsize(actual) / (1024 * 1024)
    print(f'Saved -> {actual} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
