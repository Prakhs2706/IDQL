"""
Gym-compatible wrapper around Waymax MultiAgentEnvironment.

Uses the same observation/action format as the expert data extraction
pipeline and WaymoOfflineRL evaluation, presenting a standard gym.Env
interface for the jaxrl5 evaluation loop.

Action: 2-D continuous [acceleration, steering] clipped to
        [-10, 8] m/s^2 and [-0.8, 0.8] rad.
Obs:    flat 302-D vector from waymax_obs_utils.
"""

import sys
import os

import gym
import jax
import jax.numpy as jnp
import numpy as np
from gym import spaces

from waymax import config as wmx_config
from waymax import datatypes
from waymax import dynamics
from waymax import env as wmx_env

from jaxrl5.wrappers.waymax_obs_utils import (
    state_to_feature_dict, flatten_state_dict, OBS_DIM,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'launcher', 'examples'))
from scenario_to_waymax import scenario_generator

MAX_ACCELERATION = 8.0
MIN_ACCELERATION = -10.0
MAX_STEERING_ANGLE = 0.8


class WaymaxGymWrapper(gym.Env):
    """Wraps Waymax MultiAgentEnvironment for IDQL evaluation."""

    def __init__(
        self,
        tfrecord_dir: str,
        max_scenarios: int = 1000,
        start_scenario: int = 0,
        max_episode_steps: int = 80,
    ):
        super().__init__()

        self._tfrecord_dir = tfrecord_dir
        self._max_scenarios = max_scenarios
        self._start_scenario = start_scenario
        self._max_episode_steps = max_episode_steps

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([MIN_ACCELERATION, -MAX_STEERING_ANGLE], dtype=np.float32),
            high=np.array([MAX_ACCELERATION, MAX_STEERING_ANGLE], dtype=np.float32),
            dtype=np.float32,
        )

        self._dynamics_model = dynamics.InvertibleBicycleModel()
        self._gen = None
        self._state = None
        self._env = None
        self._map_data = None
        self._tl_lookup = None
        self._step_count = 0
        self._scenario_id = None

    def _ensure_generator(self):
        if self._gen is None:
            self._gen = scenario_generator(
                tfrecord_dir=self._tfrecord_dir,
                max_scenarios=self._max_scenarios,
                start_scenario=self._start_scenario,
            )

    def _next_scenario(self):
        self._ensure_generator()
        try:
            sid, raw_state, map_data, tl_lookup = next(self._gen)
        except StopIteration:
            self._gen = scenario_generator(
                tfrecord_dir=self._tfrecord_dir,
                max_scenarios=self._max_scenarios,
                start_scenario=self._start_scenario,
            )
            sid, raw_state, map_data, tl_lookup = next(self._gen)

        self._map_data = map_data
        self._tl_lookup = tl_lookup
        self._scenario_id = sid

        env_config = wmx_config.EnvironmentConfig(
            max_num_objects=raw_state.num_objects,
            controlled_object=wmx_config.ObjectType.SDC,
            metrics=wmx_config.MetricsConfig(),
            rewards=wmx_config.LinearCombinationRewardConfig(rewards={}),
        )
        self._env = wmx_env.MultiAgentEnvironment(
            dynamics_model=self._dynamics_model,
            config=env_config,
        )
        self._state = self._env.reset(raw_state)
        self._jit_step = jax.jit(self._env.step)

    def reset(self, **kwargs):
        self._next_scenario()
        self._step_count = 0
        ts = int(np.asarray(self._state.timestep))
        obs = flatten_state_dict(
            state_to_feature_dict(
                self._state, ts, self._map_data, self._tl_lookup,
                use_sim_trajectory=True,
            )
        )
        return obs.astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action[0] = np.clip(action[0], MIN_ACCELERATION, MAX_ACCELERATION)
        action[1] = np.clip(action[1], -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)

        sdc_idx = int(jnp.argmax(self._state.object_metadata.is_sdc))
        num_obj = self._state.num_objects

        action_data = jnp.zeros((num_obj, 2))
        action_data = action_data.at[sdc_idx].set(jnp.array(action))
        action_valid = jnp.zeros((num_obj, 1), dtype=bool)
        action_valid = action_valid.at[sdc_idx].set(True)

        waymax_action = datatypes.Action(data=action_data, valid=action_valid)
        self._state = self._jit_step(self._state, waymax_action)
        self._step_count += 1

        ts = int(np.asarray(self._state.timestep))
        obs = flatten_state_dict(
            state_to_feature_dict(
                self._state, ts, self._map_data, self._tl_lookup,
                use_sim_trajectory=True,
            )
        )

        sdc_valid = bool(np.asarray(self._state.sim_trajectory.valid)[sdc_idx, ts])
        terminated = not sdc_valid
        truncated = self._step_count >= self._max_episode_steps or ts >= 90
        done = terminated or truncated

        reward = 0.0
        info = {
            'scenario_id': self._scenario_id,
            'timestep': ts,
            'step_count': self._step_count,
        }
        return obs.astype(np.float32), float(reward), done, info

    def seed(self, seed=None):
        return [seed]

    def close(self):
        self._gen = None
        self._state = None
        self._env = None
