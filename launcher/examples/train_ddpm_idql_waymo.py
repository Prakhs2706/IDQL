"""
Train DDPM-IQL on Waymo offline dataset extracted via Waymax.

Loads a .npz dataset (from extract_waymax_expert_data.py), trains a
DDPMIQLLearner, and periodically evaluates in Waymax.

Usage:
    python launcher/examples/train_ddpm_idql_waymo.py \\
        --dataset_path waymax_expert/waymo_train_10k.npz \\
        --eval_data_dir /path/to/RawWaymo/training/ \\
        --eval_num_scenarios 50 --eval_start_scenario 10000 \\
        --eval_episodes 20 --max_steps 1000000 --eval_interval 100000 \\
        --batch_size 256
"""

import os
import argparse
import time
import numpy as np
import jax
import wandb
from tqdm import tqdm

from jaxrl5.agents import DDPMIQLLearner
from jaxrl5.data.dataset import Dataset
from jaxrl5.wrappers.waymax_wrapper import WaymaxGymWrapper
from jaxrl5.wrappers import wrap_gym


# --- Action normalization (raw <-> [-1, 1]) ---
ACT_LOW = np.array([-10.0, -0.8], dtype=np.float32)
ACT_HIGH = np.array([8.0, 0.8], dtype=np.float32)


def normalize_actions(acts):
    """Map raw [accel, steer] -> [-1, 1]."""
    return 2.0 * (acts - ACT_LOW) / (ACT_HIGH - ACT_LOW) - 1.0


def denormalize_action(norm_action):
    """Map [-1, 1] -> raw [accel, steer]."""
    norm_action = np.asarray(norm_action, dtype=np.float32)
    return ACT_LOW + (norm_action + 1.0) * 0.5 * (ACT_HIGH - ACT_LOW)


def load_waymo_dataset(path, do_normalize_actions=True):
    """Load a .npz dataset into a jaxrl5 Dataset object.

    Actions are stored in raw [acceleration, steering] space.
    When do_normalize_actions=True, they are mapped to [-1,1] for DDPM sampling.
    """
    data = np.load(path)
    dataset_dict = {k: data[k] for k in data.files}

    for k in ["observations", "actions", "rewards", "next_observations", "masks"]:
        dataset_dict[k] = dataset_dict[k].astype(np.float32)

    if "dones" in dataset_dict:
        dones_raw = dataset_dict["dones"]
        if dones_raw.dtype != bool:
            dataset_dict["dones"] = dones_raw.astype(bool)

    if do_normalize_actions:
        dataset_dict["actions"] = normalize_actions(dataset_dict["actions"])
        eps = 1e-5
        dataset_dict["actions"] = np.clip(dataset_dict["actions"], -1 + eps, 1 - eps)

    n = len(dataset_dict["observations"])
    print(f"Loaded dataset: {n} transitions")
    print(f"  obs shape:  {dataset_dict['observations'].shape}")
    print(f"  act shape:  {dataset_dict['actions'].shape}")
    print(f"  act range:  [{dataset_dict['actions'].min():.4f}, {dataset_dict['actions'].max():.4f}]")
    print(f"  reward range: [{dataset_dict['rewards'].min():.4f}, {dataset_dict['rewards'].max():.4f}]")

    return Dataset(dataset_dict)


def evaluate_waymax(agent, env, num_episodes):
    """Evaluate the IDQL agent in Waymax."""
    all_returns = []
    all_lengths = []

    for ep in tqdm(range(num_episodes), desc="Eval", leave=False):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done and ep_length < 80:
            norm_action, agent = agent.eval_actions(obs)
            raw_action = denormalize_action(norm_action)
            obs, reward, done, info = env.step(raw_action)
            ep_reward += reward
            ep_length += 1

        all_returns.append(ep_reward)
        all_lengths.append(ep_length)

    return {
        "return_mean": np.mean(all_returns),
        "return_std": np.std(all_returns),
        "episode_length_mean": np.mean(all_lengths),
        "episode_length_std": np.std(all_lengths),
        "num_episodes": num_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DDPM-IQL on Waymo offline data")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to .npz dataset from extract_waymax_expert_data.py")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Directory with raw Waymo TFRecord files for eval")
    parser.add_argument("--eval_num_scenarios", type=int, default=50)
    parser.add_argument("--eval_start_scenario", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--save_interval", type=int, default=100000)
    parser.add_argument("--normalize_returns", action="store_true", default=False)
    parser.add_argument("--no_normalize_returns", action="store_true", default=False)
    parser.add_argument("--project", type=str, default="idql_waymo")
    parser.add_argument("--experiment_name", type=str, default="ddpm_iql_waymax_302d")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--no_wandb", action="store_true", default=False)

    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--M", type=int, default=0)
    parser.add_argument("--critic_hyperparam", type=float, default=0.9)
    parser.add_argument("--actor_dropout_rate", type=float, default=0.1)
    parser.add_argument("--actor_num_blocks", type=int, default=3)
    parser.add_argument("--actor_tau", type=float, default=0.001)
    parser.add_argument("--beta_schedule", type=str, default="vp")
    args = parser.parse_args()

    if args.no_normalize_returns:
        args.normalize_returns = False

    # --- Load dataset ---
    print(f"Loading dataset from {args.dataset_path}")
    ds = load_waymo_dataset(args.dataset_path)

    if args.normalize_returns:
        print("Normalizing returns ...")
        ds.normalize_returns(scaling=1000)
        print(f"  Reward range after normalize: "
              f"[{ds.dataset_dict['rewards'].min():.4f}, {ds.dataset_dict['rewards'].max():.4f}]")

    ds, ds_val = ds.split(0.95)
    print(f"Train: {len(ds)} transitions, Val: {len(ds_val)} transitions")

    # --- Create eval environment ---
    eval_env = None
    if args.eval_data_dir is not None:
        print(f"Creating eval env from {args.eval_data_dir}")
        eval_env = WaymaxGymWrapper(
            tfrecord_dir=args.eval_data_dir,
            max_scenarios=args.eval_num_scenarios,
            start_scenario=args.eval_start_scenario,
        )
        eval_env = wrap_gym(eval_env, rescale_actions=False)
        print(f"Eval env: obs={eval_env.observation_space}, act={eval_env.action_space}")

    # --- Create agent ---
    obs_space = eval_env.observation_space if eval_env is not None else None
    act_space = eval_env.action_space if eval_env is not None else None

    if obs_space is None:
        import gym
        obs_dim = ds.dataset_dict["observations"].shape[1]
        act_dim = ds.dataset_dict["actions"].shape[1]
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    decay_steps = int(args.max_steps)

    rl_config = dict(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        T=args.T,
        N=args.N,
        M=args.M,
        actor_dropout_rate=args.actor_dropout_rate,
        actor_num_blocks=args.actor_num_blocks,
        decay_steps=decay_steps,
        actor_layer_norm=True,
        value_layer_norm=True,
        actor_tau=args.actor_tau,
        beta_schedule=args.beta_schedule,
        critic_hyperparam=args.critic_hyperparam,
    )

    print(f"Creating DDPMIQLLearner (seed={args.seed})")
    print(f"  obs_space: {obs_space}")
    print(f"  act_space: {act_space}")
    print(f"  config: {rl_config}")

    agent = DDPMIQLLearner.create(
        seed=args.seed,
        observation_space=obs_space,
        action_space=act_space,
        **rl_config,
    )

    # --- W&B ---
    if not args.no_wandb:
        wandb.init(project=args.project, name=args.experiment_name)
        wandb.config.update(vars(args))
        wandb.config.update(rl_config)

    # --- Training loop ---
    print(f"\nStarting training for {args.max_steps} steps ...")
    os.makedirs(args.save_dir, exist_ok=True)

    training_time_inference_params = dict(N=args.N, clip_sampler=True, M=args.M)

    t_start = time.time()
    sample = ds.sample_jax(args.batch_size, keys=None)

    for i in tqdm(range(args.max_steps), smoothing=0.1):
        sample = ds.sample_jax(args.batch_size, keys=None)
        agent, info = agent.update(sample)

        if i % args.log_interval == 0:
            val_sample = ds_val.sample(args.batch_size, keys=None)
            _, val_info = agent.update(val_sample)

            info = jax.device_get(info)
            val_info = jax.device_get(val_info)

            if not args.no_wandb:
                wandb.log({f"train/{k}": float(v) for k, v in info.items()}, step=i)
                wandb.log({f"val/{k}": float(v) for k, v in val_info.items()}, step=i)

            if i % (args.log_interval * 10) == 0:
                elapsed = time.time() - t_start
                steps_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"Step {i} | "
                    f"actor_loss={info.get('actor_loss', 0):.4f} | "
                    f"critic_loss={info.get('critic_loss', 0):.4f} | "
                    f"value_loss={info.get('value_loss', 0):.4f} | "
                    f"v={info.get('v', 0):.4f} | "
                    f"q={info.get('q', 0):.4f} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

        if i % args.eval_interval == 0 and i > 0 and eval_env is not None:
            agent = agent.replace(N=args.N, clip_sampler=True, M=0)
            eval_info = evaluate_waymax(agent, eval_env, args.eval_episodes)

            if not args.no_wandb:
                wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)

            print(
                f"  Eval (N={args.N}): "
                f"return={eval_info['return_mean']:.2f}±{eval_info['return_std']:.2f} | "
                f"ep_len={eval_info['episode_length_mean']:.1f}"
            )

            agent = agent.replace(**training_time_inference_params)

        if i % args.save_interval == 0 and i > 0:
            ckpt_path = os.path.join(args.save_dir, f"agent_step_{i}")
            os.makedirs(ckpt_path, exist_ok=True)
            params_to_save = jax.device_get(agent.score_model.params)
            critic_params = jax.device_get(agent.critic.params)
            value_params = jax.device_get(agent.value.params)
            np.savez(
                os.path.join(ckpt_path, "params.npz"),
                score_model=params_to_save,
                critic=critic_params,
                value=value_params,
            )
            print(f"  Saved checkpoint to {ckpt_path}")

    # --- Final save ---
    ckpt_path = os.path.join(args.save_dir, "agent_final")
    os.makedirs(ckpt_path, exist_ok=True)
    params_to_save = jax.device_get(agent.score_model.params)
    critic_params = jax.device_get(agent.critic.params)
    value_params = jax.device_get(agent.value.params)
    np.savez(
        os.path.join(ckpt_path, "params.npz"),
        score_model=params_to_save,
        critic=critic_params,
        value=value_params,
    )
    print(f"Saved final checkpoint to {ckpt_path}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete! Total time: {elapsed / 3600:.2f} hours")

    if eval_env is not None:
        eval_env.close()

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
