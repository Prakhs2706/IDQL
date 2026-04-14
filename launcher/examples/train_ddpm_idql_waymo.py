"""
Train DDPM-IQL on Waymo offline dataset extracted via Waymax.

Loads a .npz dataset (from extract_waymax_expert_data.py), trains a
DDPMIQLLearner, and periodically evaluates in Waymax.

Usage:
    python launcher/examples/train_ddpm_idql_waymo.py \\
        --eval_data_dir ~/scratch/data/raw/validation \\
        --max_steps 500000 --eval_interval 100000 --batch_size 512

    Defaults: training data from ~/scratch/data/waymax_expert/waymo_train_10k.npz
    (10k scenarios from extract_waymax_expert_data.py); eval uses up to 1k
    validation scenarios (eval_num_scenarios=1000, eval_episodes=1000,
    eval_start_scenario=0).

    GPU (JAX): install a CUDA jaxlib matching the machine (e.g. pip install jax[cuda12]
    per JAX docs), set CUDA_VISIBLE_DEVICES, then verify:
        python -c "import jax; print(jax.default_backend(), jax.devices())"
    Eval uses JAX for Waymax dynamics and the DDPM actor; NumPy featurization stays
    on the host.

    Each eval logs eval_profile_* scalars (and a profile line on stdout): time in
    reset, agent.eval_actions, env.step, collect_episode_eval_stats, plus ms/step.

    First timed eval can look much slower than later ones: XLA compiles `env.step`,
    Waymax metric kernels, and the DDPM+Q `eval_actions` path on first use. A short
    JIT warmup runs once after creating the agent (see warmup_waymax_eval_jit) so
    the first full eval is closer to steady state.

    Optional headless GIFs: set --eval_gif_dir to save matplotlib/Waymax frames
    (no display; use MPLBACKEND=Agg). Requires imageio (already in requirements.txt).
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

DEFAULT_WAYMAX_EXPERT_DIR = os.path.expanduser('~/scratch/data/waymax_expert')
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_WAYMAX_EXPERT_DIR, 'waymo_train_10k.npz')

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


def warmup_waymax_eval_jit(eval_env, agent, denormalize_fn):
    """Compile JAX paths (env.step, Waymax metrics, eval_actions) before timed eval."""
    base = eval_env.unwrapped
    for _ in range(2):
        obs = eval_env.reset()
        done = False
        steps = 0
        while not done and steps < 80:
            norm_action, agent = agent.eval_actions(obs)
            obs, _, done, _ = eval_env.step(denormalize_fn(norm_action))
            steps += 1
        if hasattr(base, "collect_episode_eval_stats"):
            base.collect_episode_eval_stats()
    eval_env.reset()
    return agent


def record_eval_rollout_gif(agent, eval_env, gif_path, denormalize_fn, frame_stride=2, max_steps=80):
    """Save one rollout as GIF using Waymax matplotlib viz (headless-safe).

    Returns ``(agent, True)`` if a GIF was written, ``(agent, False)`` if imports
    or rendering failed (training should continue). ``agent`` is always the
    post-rollout handle when the rollout loop ran, else unchanged on import failure.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import imageio.v2 as imageio
        from waymax.visualization.viz import plot_simulator_state
    except ImportError as e:
        print(
            "  Eval GIF skipped: matplotlib/imageio import failed "
            f"({e}). Fix env e.g. `pip install -U 'pyparsing>=3' matplotlib` "
            "or unset --eval_gif_dir.",
            flush=True,
        )
        return agent, False

    try:
        base = eval_env.unwrapped
        frames = []
        obs = eval_env.reset()
        for t in range(max_steps):
            state = jax.device_get(base._state)
            if t % frame_stride == 0:
                frames.append(plot_simulator_state(state, use_log_traj=False))
            norm_action, agent = agent.eval_actions(obs)
            obs, _, done, _ = eval_env.step(denormalize_fn(norm_action))
            if done:
                state = jax.device_get(base._state)
                if t % frame_stride != 0:
                    frames.append(plot_simulator_state(state, use_log_traj=False))
                break
        if not frames:
            return agent, False
        parent = os.path.dirname(os.path.abspath(gif_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
        return agent, True
    except Exception as e:
        print(f"  Eval GIF skipped: {e}", flush=True)
        return agent, False


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
    """Evaluate the IDQL agent in Waymax; aggregate returns, lengths, Waymax metrics, offline KPIs.

    Also records wall time in coarse sections (perf_counter): reset, policy forward,
    env.step (includes denormalize + Waymax step + featurization in the wrapper),
    and collect_episode_eval_stats. Use these to see where eval time goes.
    """
    all_returns = []
    all_lengths = []
    episode_stat_dicts = []

    base = env.unwrapped
    t_eval0 = time.time()

    t_reset_total = 0.0
    t_policy_total = 0.0
    t_step_total = 0.0
    t_collect_total = 0.0
    n_env_steps = 0

    for ep in tqdm(range(num_episodes), desc="Eval", leave=False):
        t0 = time.perf_counter()
        obs = env.reset()
        t_reset_total += time.perf_counter() - t0
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done and ep_length < 80:
            t0 = time.perf_counter()
            norm_action, agent = agent.eval_actions(obs)
            t_policy_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            raw_action = denormalize_action(norm_action)
            obs, reward, done, info = env.step(raw_action)
            t_step_total += time.perf_counter() - t0
            n_env_steps += 1

            ep_reward += reward
            ep_length += 1

        all_returns.append(ep_reward)
        all_lengths.append(ep_length)

        if hasattr(base, "collect_episode_eval_stats"):
            t0 = time.perf_counter()
            episode_stat_dicts.append(base.collect_episode_eval_stats())
            t_collect_total += time.perf_counter() - t0

    eval_elapsed_s = time.time() - t_eval0
    accounted_s = t_reset_total + t_policy_total + t_step_total + t_collect_total

    out = {
        "return_mean": float(np.mean(all_returns)),
        "return_std": float(np.std(all_returns)),
        "episode_length_mean": float(np.mean(all_lengths)),
        "episode_length_std": float(np.std(all_lengths)),
        "num_episodes": int(num_episodes),
        "eval_wall_time_s": float(eval_elapsed_s),
        "eval_profile_reset_s": float(t_reset_total),
        "eval_profile_policy_s": float(t_policy_total),
        "eval_profile_env_step_s": float(t_step_total),
        "eval_profile_collect_stats_s": float(t_collect_total),
        "eval_profile_n_env_steps": int(n_env_steps),
        "eval_profile_accounted_s": float(accounted_s),
        "eval_profile_unaccounted_s": float(max(0.0, eval_elapsed_s - accounted_s)),
    }
    if n_env_steps > 0:
        out["eval_profile_policy_ms_per_env_step"] = float(
            1000.0 * t_policy_total / n_env_steps
        )
        out["eval_profile_env_step_ms_per_env_step"] = float(
            1000.0 * t_step_total / n_env_steps
        )
    if num_episodes > 0:
        out["eval_profile_reset_ms_per_episode"] = float(
            1000.0 * t_reset_total / num_episodes
        )
        out["eval_profile_collect_stats_ms_per_episode"] = float(
            1000.0 * t_collect_total / num_episodes
        )

    if episode_stat_dicts:
        keys = set()
        for d in episode_stat_dicts:
            keys.update(d.keys())
        for k in sorted(keys):
            vals = [float(d[k]) for d in episode_stat_dicts if k in d and np.isfinite(d[k])]
            if not vals:
                continue
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
        if "success_mean" in out:
            out["success_rate_pct"] = float(100.0 * out["success_mean"])
        if "collision_mean" in out:
            out["collision_rate_pct"] = float(100.0 * out["collision_mean"])
        if "off_road_mean" in out:
            out["off_road_rate_pct"] = float(100.0 * out["off_road_mean"])
        if "goal_completed_mean" in out:
            out["goal_completion_rate_pct"] = float(100.0 * out["goal_completed_mean"])

    return out


def main():
    parser = argparse.ArgumentParser(description="Train DDPM-IQL on Waymo offline data")
    parser.add_argument(
        "--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
        help=f"Path to .npz from extract_waymax_expert_data.py (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Directory with raw Waymo TFRecord files for eval")
    parser.add_argument(
        "--eval_num_scenarios", type=int, default=1000,
        help="Max scenarios in the eval TFRecord iterator (use 1000 for 1k-scenario eval pool).",
    )
    parser.add_argument(
        "--eval_start_scenario", type=int, default=0,
        help="Skip this many scenarios in eval_dir before the eval pool (use 0 for validation/).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--eval_episodes", type=int, default=1000,
        help="Rollouts per eval call; set equal to eval_num_scenarios for one episode per scenario.",
    )
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=50000)
    parser.add_argument("--save_interval", type=int, default=100000)
    parser.add_argument("--normalize_returns", action="store_true", default=False)
    parser.add_argument("--no_normalize_returns", action="store_true", default=False)
    parser.add_argument("--project", type=str, default="idql_waymo")
    parser.add_argument("--experiment_name", type=str, default="ddpm_iql_waymax_302d")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument(
        "--eval_gif_dir",
        type=str,
        default=None,
        help=(
            "If set, after each eval save one rollout GIF here (matplotlib Agg; no display). "
            "Uses one extra scenario (one reset + rollout after the main eval). "
            "Requires a working matplotlib stack (e.g. pip install -U 'pyparsing>=3' matplotlib); "
            "if imports fail, GIF is skipped and training continues."
        ),
    )
    parser.add_argument(
        "--eval_gif_stride",
        type=int,
        default=2,
        help="Save every k-th timestep as a frame (smaller GIFs).",
    )

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

    print(
        f"JAX default_backend={jax.default_backend()} | "
        f"devices={jax.devices()}",
        flush=True,
    )

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

    if eval_env is not None:
        print(
            "Warmup: compiling Waymax env.step + metrics + eval policy (not counted in eval) ...",
            flush=True,
        )
        t_w = time.perf_counter()
        agent = warmup_waymax_eval_jit(eval_env, agent, denormalize_action)
        print(f"  Warmup done in {time.perf_counter() - t_w:.1f}s", flush=True)

    if args.eval_gif_dir:
        os.makedirs(args.eval_gif_dir, exist_ok=True)

    # --- W&B ---
    if not args.no_wandb:
        wandb.init(project=args.project, name=args.experiment_name)
        wandb.config.update(vars(args))
        wandb.config.update(rl_config)
        wandb.config.update({
            "jax_default_backend": jax.default_backend(),
            "jax_devices_repr": str(jax.devices()),
        })

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
                eval_log = {}
                for k, v in eval_info.items():
                    try:
                        x = float(v)
                        if np.isfinite(x):
                            eval_log[f"eval/{k}"] = x
                    except (TypeError, ValueError):
                        pass
                wandb.log(eval_log, step=i)

            extra = " ".join(
                f"{k}={eval_info[k]:.4f}"
                for k in sorted(eval_info)
                if k.startswith("waymax_") and k.endswith("_mean")
            )
            kpi = " ".join(
                f"{k.replace('_rate_pct', '')}={eval_info[k]:.2f}%"
                for k in (
                    "success_rate_pct",
                    "collision_rate_pct",
                    "off_road_rate_pct",
                    "goal_completion_rate_pct",
                )
                if k in eval_info
            )
            wall = float(eval_info.get("eval_wall_time_s", 0.0))
            pol = float(eval_info.get("eval_profile_policy_s", 0.0))
            stp = float(eval_info.get("eval_profile_env_step_s", 0.0))
            rst = float(eval_info.get("eval_profile_reset_s", 0.0))
            col = float(eval_info.get("eval_profile_collect_stats_s", 0.0))
            unacc = float(eval_info.get("eval_profile_unaccounted_s", 0.0))
            pol_ms = float(eval_info.get("eval_profile_policy_ms_per_env_step", 0.0))
            stp_ms = float(eval_info.get("eval_profile_env_step_ms_per_env_step", 0.0))
            pct = lambda x: (100.0 * x / wall) if wall > 0 else 0.0
            profile_str = (
                f"profile wall={wall:.1f}s "
                f"(policy {pct(pol):.0f}% {pol_ms:.1f}ms/step | "
                f"env_step {pct(stp):.0f}% {stp_ms:.1f}ms/step | "
                f"reset {pct(rst):.0f}% | collect {pct(col):.0f}% | "
                f"unaccounted {pct(unacc):.0f}%)"
            )
            print(
                f"  Eval (N={args.N}): "
                f"return={eval_info['return_mean']:.3f}±{eval_info['return_std']:.3f} | "
                f"ep_len={eval_info['episode_length_mean']:.1f} | "
                f"{profile_str} | "
                f"{extra} | {kpi}",
                flush=True,
            )

            if args.eval_gif_dir:
                gif_path = os.path.join(args.eval_gif_dir, f"eval_step_{i}.gif")
                agent, gif_ok = record_eval_rollout_gif(
                    agent,
                    eval_env,
                    gif_path,
                    denormalize_action,
                    frame_stride=args.eval_gif_stride,
                )
                if gif_ok:
                    print(f"  Eval GIF saved: {gif_path}", flush=True)
                    if not args.no_wandb:
                        wandb.log({"eval/rollout_gif_path": gif_path}, step=i)

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
