"""
Ego_Agent/Trainer.py
────────────────────
在 highway-v0 上训练 Ego 代理（DQN / PPO / A2C）。

A2C 修改说明（PPO / DQN 流程完全不变）
──────────────────────────────────────
1. VecEnv   → SubprocVecEnv（默认 n_envs=8）
   参考：SB3 RL Tips「Vectorized Environments」
   原因：DummyVecEnv 串行采样导致样本高度相关，on-policy A2C 梯度方差大；
         SubprocVecEnv 多进程并行，样本去相关，梯度更稳定。

2. VecNormalize（norm_obs=True, norm_reward=True, clip_obs=10）
   参考：SB3 RL Tips「Normalization」
   原因：Kinematics obs 混合了位置（米）与速度（m/s），尺度差异大；
         未归一化的 obs 使 critic 值函数方差高，A2C advantage 估计噪声强。
         训练统计保存至 vecnorm.pkl，测试时加载冻结参数保证一致性。

3. A2C 超参（RL Zoo 风格）
   参考：rl-baselines3-zoo / hyperparams/a2c.yml + SB3 文档
   - RMSpropTFLike(eps=1e-5)：比默认 eps=1e-8 更稳定，TF A2C 官方实现匹配。
   - net_arch [256,256]：Kinematics 含关系结构，需要更大容量。
   - ortho_init=False：RL Zoo 对 MLP policy 的推荐，避免 tanh 饱和。
   - gamma=0.995：highway episode ≈50步，需更长远信用分配。
   - ent_coef=0.02：防止策略过早收敛到确定性 SLOWER。
   - n_steps=64：8 env × 64 = 512 samples/update，方差远低于 n_steps=5。

4. 训练预算
   参考：SB3 RL Tips「Train for as many steps as possible」
   建议 ≥ 2M，推荐 3M-5M。

5. A2C 固定使用 device="cpu"
   原因：MPS 后端在 macOS 上与 SB3 A2C 存在死锁风险。
"""

import argparse
import os
import warnings

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def auto_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_class(model_name):
    return {"DQN": DQN, "PPO": PPO, "A2C": A2C}.get(model_name)


def make_env_fn(lanes_count, vehicles_count, duration, render=False):
    """Returns a callable that creates and configures one highway-v0 env."""
    def _init():
        mode = "rgb_array" if render else None
        e = gym.make("highway-v0", render_mode=mode)
        e.configure({
            "lanes_count": lanes_count,
            "vehicles_count": vehicles_count,
            "duration": duration,
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "collision_reward": -1.0,
            "high_speed_reward": 0.8,
            "right_lane_reward": 0.4,
            "reward_speed_range": [20, 35],
            "normalize_reward": True,
        })
        return e
    return _init


def _build_a2c(env, tb_log: str, seed: int) -> A2C:
    """
    构建使用 RL Zoo 风格超参数的 A2C 模型。
    所有参数集中于此，方便调整，与 PPO/DQN 路径完全隔离。
    """
    policy_kwargs = dict(
        optimizer_class=RMSpropTFLike,
        optimizer_kwargs=dict(eps=1e-5),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ortho_init=False,
    )
    return A2C(
        "MlpPolicy", env,
        device="cpu",           # A2C 固定 CPU，避免 MPS 死锁
        learning_rate=7e-4,
        n_steps=64,             # 8 env × 64 = 512 samples/update
        gamma=0.995,            # 长视野，适合 50 步 episode
        gae_lambda=0.95,        # 标准 λ，降低方差
        ent_coef=0.02,          # 熵正则，防止策略退化为纯 SLOWER
        vf_coef=0.4,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log,
        seed=seed,
    )


def main(args):
    # A2C 强制 cpu；PPO/DQN 保持原来的自动检测
    if args.ego_type == "A2C":
        device = "cpu"
    else:
        device = args.device if args.device else auto_device()

    print(f">> Device : {device}")
    print(f">> n_envs : {args.n_envs}")

    model_type   = get_model_class(args.ego_type)
    save_dir     = f"Ego_Agent/Ego_Agent_model/{args.ego_type}"
    save_path    = f"{save_dir}/model"
    vecnorm_path = f"{save_dir}/vecnorm.pkl"
    tb_log       = f"{save_dir}/"
    # 每个 (lane, seed) 组合独立命名，避免不同配置的曲线混进同一个 Run
    # （SB3 会在此基础上自动追加 _1/_2/... 后缀，不会覆盖旧日志）
    run_name     = f"{args.ego_type}_lane{args.lanes_count}_seed{args.seed}"
    os.makedirs(save_dir, exist_ok=True)

    # ── 训练 ─────────────────────────────────────────────────────────────────
    if args.train:
        print(f">> Ego Model type : {args.ego_type}")
        print(f">> Steps          : {args.total_timesteps:,}")
        print(f">> Seed           : {args.seed}")
        print(f">> TB run name    : {run_name}")

        if args.ego_type == "A2C":
            # ── A2C 专用路径 ────────────────────────────────────────────────
            # SubprocVecEnv：多进程并行，样本去相关
            env = make_vec_env(
                make_env_fn(args.lanes_count, args.vehicles_count, args.duration),
                n_envs=args.n_envs,
                seed=args.seed,
                vec_env_cls=SubprocVecEnv,   # 关键：替换 DummyVecEnv
            )
            # VecNormalize：对 obs 和 reward 做 running mean/var 归一化
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
            )
            model = _build_a2c(env, tb_log, args.seed)
            print(">> A2C RL Zoo config: SubprocVecEnv / VecNormalize / RMSpropTFLike")

        else:
            # ── PPO / DQN 原有路径，完全不变 ────────────────────────────────
            env = make_vec_env(
                make_env_fn(args.lanes_count, args.vehicles_count, args.duration),
                n_envs=args.n_envs,
                seed=args.seed,
            )
            supports_batch_size = args.ego_type in ("PPO", "DQN")
            batch_size = min(256 * args.n_envs, 1024) if supports_batch_size else None
            if batch_size:
                print(f">> batch_size     : {batch_size}")

            model_kwargs = dict(verbose=1, device=device, tensorboard_log=tb_log, seed=args.seed)
            if batch_size:
                model_kwargs["batch_size"] = batch_size
            model = model_type("MlpPolicy", env, **model_kwargs)

        model.learn(total_timesteps=args.total_timesteps, log_interval=1, tb_log_name=run_name)
        model.save(save_path)
        print(f"\n>> Model saved → {save_path}.zip")

        # VecNormalize 统计只针对 A2C 保存，供测试时加载
        if args.ego_type == "A2C":
            env.save(vecnorm_path)
            print(f">> VecNormalize stats → {vecnorm_path}")

        del model
        env.close()

    # ── 测试 + 录像 ───────────────────────────────────────────────────────────
    if args.test:
        video_dir = f"{save_dir}/video"

        if args.ego_type == "A2C":
            # 测试时加载 VecNormalize 统计（freeze running stats）
            raw_env = make_env_fn(
                args.lanes_count, args.vehicles_count, args.duration, render=True
            )()
            raw_env = RecordVideo(
                raw_env, video_folder=video_dir, episode_trigger=lambda e: True
            )
            raw_env.unwrapped.set_record_video_wrapper(raw_env)
            raw_env.configure({"simulation_frequency": 15})

            vec_env = DummyVecEnv([lambda: raw_env])
            if os.path.exists(vecnorm_path):
                vec_env = VecNormalize.load(vecnorm_path, vec_env)
                vec_env.training   = False   # 冻结 running stats
                vec_env.norm_reward = False  # 测试时不归一化 reward
                print(f">> VecNormalize loaded from {vecnorm_path}")
            else:
                print("[WARNING] vecnorm.pkl not found — obs not normalised")

            model = A2C.load(save_path, env=vec_env, device="cpu")
            for _ in range(10):
                obs      = vec_env.reset()
                done     = [False]
                while not done[0]:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _ = vec_env.step(action)
            vec_env.close()

        else:
            # PPO / DQN 测试路径，完全不变
            single_env = make_env_fn(
                args.lanes_count, args.vehicles_count, args.duration, render=True
            )()
            model = model_type.load(save_path, env=single_env, device=device)
            single_env = RecordVideo(
                single_env,
                video_folder=video_dir,
                episode_trigger=lambda e: True,
            )
            single_env.unwrapped.set_record_video_wrapper(single_env)
            single_env.configure({"simulation_frequency": 15})

            for _ in range(10):
                done = truncated = False
                obs, _ = single_env.reset()
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, truncated, _ = single_env.step(action)
            single_env.close()

        print(f"\n>> Videos saved → {video_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train / test ego agent on highway-v0')
    parser.add_argument('--train',    action='store_true', help="run training")
    parser.add_argument('--test',     action='store_true', help="run test + record video")
    parser.add_argument('--vehicles_count', type=int, default=5)
    parser.add_argument('--duration',       type=int, default=50)
    parser.add_argument('--lanes_count',    type=int, default=2)
    parser.add_argument('--ego_type', type=str, default='DQN',
                        choices=['PPO', 'DQN', 'A2C'])
    parser.add_argument('--total_timesteps', type=int, default=500_000,
                        help="Training steps. A2C needs ≥ 2M; recommend 3M-5M.")
    parser.add_argument('--n_envs', type=int, default=8,
                        help="Parallel envs. A2C: 8 (SubprocVecEnv). PPO/DQN: 4 fine.")
    parser.add_argument('--device', type=str, default='',
                        help="PPO/DQN device. A2C always uses cpu regardless.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for env + model. Also used in the TensorBoard run name "
                             "so that different seeds/lane configs never share one Run.")
    args = parser.parse_args()
    main(args)
